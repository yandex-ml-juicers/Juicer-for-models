"""Лоссы сегментации, добавленные поверх CE: отбор трудных пикселей,
прямая оптимизация IoU, граничная и разноархитектурная дистилляция,
а также их композиция.

Сети здесь нет: всё считается на маленьких случайных тензорах.
"""

import pytest
import torch
import torch.nn.functional as F

from src.losses import (
    BPKDLoss,
    CompositeLoss,
    CrossEntropy,
    DiceLoss,
    FitNetsKD,
    FocalLoss,
    HeteroAKDLoss,
    LovaszSoftmax,
    OhemCrossEntropy,
)
from src.losses.bpkd import boundary_mask

NUM_CLASSES = 4
IGNORE_INDEX = 255


@pytest.fixture()
def batch():
    """Логиты [B, C, H, W], логиты учителя и маска с void-пикселями."""
    generator = torch.Generator().manual_seed(0)
    student = torch.randn(2, NUM_CLASSES, 16, 24, generator=generator)
    teacher = torch.randn(2, NUM_CLASSES, 16, 24, generator=generator)
    labels = torch.randint(0, NUM_CLASSES, (2, 16, 24), generator=generator)
    labels[0, 0, :6] = IGNORE_INDEX
    return student, teacher, labels


class TestOhemCrossEntropy:
    def test_keeps_only_the_hard_pixels(self, batch):
        """Лосс по отобранным пикселям обязан быть выше среднего по всем:
        в этом и смысл отбора."""
        student, _, labels = batch
        criterion = OhemCrossEntropy(thresh=0.7, keep_ratio=0.1, ignore_index=IGNORE_INDEX)
        result = criterion(student, None, labels)
        assert result["ohem"] > result["ce"]

    def test_full_keep_ratio_without_threshold_is_plain_ce(self, batch):
        student, _, labels = batch
        criterion = OhemCrossEntropy(thresh=None, keep_ratio=1.0, ignore_index=IGNORE_INDEX)
        result = criterion(student, None, labels)
        expected = F.cross_entropy(student, labels, ignore_index=IGNORE_INDEX)
        assert torch.allclose(result["total"], expected, atol=1e-6)

    def test_void_pixels_never_enter_the_selection(self, batch):
        """У void-пикселей лосс нулевой, но порог строго положителен —
        значит, попасть в отбор они не могут ни при каком keep_ratio."""
        student, _, labels = batch
        criterion = OhemCrossEntropy(thresh=0.7, keep_ratio=1.0, ignore_index=IGNORE_INDEX)
        before = criterion(student, None, labels)["total"]

        corrupted = student.clone()
        corrupted[0, :, 0, :6] = 50.0
        after = criterion(corrupted, None, labels)["total"]
        assert torch.allclose(before, after)


class TestFocalLoss:
    def test_gamma_zero_is_plain_ce(self, batch):
        student, _, labels = batch
        criterion = FocalLoss(gamma=0.0, ignore_index=IGNORE_INDEX)
        result = criterion(student, None, labels)
        expected = F.cross_entropy(student, labels, ignore_index=IGNORE_INDEX)
        assert torch.allclose(result["total"], expected, atol=1e-6)

    def test_easy_pixels_are_downweighted(self):
        """Уверенно верный пиксель должен почти выпасть из лосса,
        а ошибочный — остаться."""
        labels = torch.zeros(1, 1, 2, dtype=torch.long)
        logits = torch.zeros(1, NUM_CLASSES, 1, 2)
        logits[0, 0, 0, 0] = 10.0  # угадан уверенно
        logits[0, 1, 0, 1] = 10.0  # уверенная ошибка

        focal = FocalLoss(gamma=2.0, ignore_index=IGNORE_INDEX)(logits, None, labels)["total"]
        ce = F.cross_entropy(logits, labels)
        # Лёгкий пиксель придавлен, тяжёлый — нет, поэтому среднее падает,
        # но остаётся заметно больше нуля.
        assert 0.0 < focal.item() < ce.item()

    def test_class_weights_scale_the_loss(self, batch):
        student, _, labels = batch
        plain = FocalLoss(gamma=2.0, ignore_index=IGNORE_INDEX)(student, None, labels)["total"]
        weighted = FocalLoss(
            gamma=2.0, alpha=[2.0] * NUM_CLASSES, ignore_index=IGNORE_INDEX
        )(student, None, labels)["total"]
        assert torch.allclose(weighted, 2.0 * plain, atol=1e-6)


class TestLovaszSoftmax:
    def test_perfect_prediction_gives_zero(self):
        labels = torch.randint(0, NUM_CLASSES, (2, 8, 8))
        logits = F.one_hot(labels, NUM_CLASSES).permute(0, 3, 1, 2).float() * 50.0

        criterion = LovaszSoftmax(ce_weight=0.0, ignore_index=IGNORE_INDEX, pixel_stride=1)
        assert criterion(logits, None, labels)["lovasz"].item() < 1e-4

    def test_worse_prediction_gives_larger_loss(self, batch):
        student, _, labels = batch
        criterion = LovaszSoftmax(ce_weight=0.0, ignore_index=IGNORE_INDEX, pixel_stride=1)

        good = F.one_hot(labels.clamp(max=NUM_CLASSES - 1), NUM_CLASSES)
        good = good.permute(0, 3, 1, 2).float() * 10.0

        assert criterion(student, None, labels)["lovasz"] > criterion(good, None, labels)["lovasz"]

    def test_gradient_flows(self, batch):
        student, _, labels = batch
        student = student.clone().requires_grad_(True)
        LovaszSoftmax(ignore_index=IGNORE_INDEX, pixel_stride=2)(student, None, labels)[
            "total"
        ].backward()
        assert student.grad is not None and torch.isfinite(student.grad).all()


class TestBoundaryMask:
    def test_marks_only_the_band_around_a_class_change(self):
        labels = torch.zeros(1, 9, 9, dtype=torch.long)
        labels[:, :, 5:] = 1

        mask = boundary_mask(labels, width=3)[0, 0]
        # Полоса шириной 3 вокруг стыка столбцов 4/5.
        assert mask[:, 4].all() and mask[:, 5].all()
        assert not mask[:, :3].any() and not mask[:, 7:].any()

    def test_uniform_label_has_no_boundary(self):
        labels = torch.full((1, 8, 8), 3, dtype=torch.long)
        assert boundary_mask(labels, width=5).sum() == 0


class TestBPKD:
    def test_both_terms_vanish_when_student_equals_teacher(self, batch):
        student, _, labels = batch
        criterion = BPKDLoss(ce_weight=0.0, ignore_index=IGNORE_INDEX)
        result = criterion(student, student.clone(), labels)
        assert result["edge"].abs().item() < 1e-5
        assert result["body"].abs().item() < 1e-5

    def test_terms_are_non_negative(self, batch):
        student, teacher, labels = batch
        result = BPKDLoss(ignore_index=IGNORE_INDEX)(student, teacher, labels)
        assert result["edge"] >= 0 and result["body"] >= 0

    def test_edge_term_ignores_what_happens_inside_the_body(self):
        """Ошибка ученика вдали от границы не должна двигать edge-член —
        иначе он ничем не отличался бы от обычной попиксельной KD."""
        labels = torch.zeros(1, 16, 16, dtype=torch.long)
        labels[:, :, 8:] = 1
        student = torch.randn(1, NUM_CLASSES, 16, 16, generator=torch.Generator().manual_seed(1))
        teacher = torch.randn(1, NUM_CLASSES, 16, 16, generator=torch.Generator().manual_seed(2))

        criterion = BPKDLoss(ce_weight=0.0, edge_width=3, ignore_index=IGNORE_INDEX)
        before = criterion(student, teacher, labels)["edge"]

        corrupted = student.clone()
        corrupted[:, :, :, 0] = 50.0  # столбец далеко от стыка 7/8
        after = criterion(corrupted, teacher, labels)["edge"]

        assert torch.allclose(before, after)


class TestHeteroAKD:
    @pytest.fixture()
    def features(self):
        generator = torch.Generator().manual_seed(3)
        return (
            {"taps.stage4": torch.randn(2, 8, 4, 6, generator=generator)},
            {"taps.stage4": torch.randn(2, 12, 2, 3, generator=generator)},
        )

    def make(self, **kwargs):
        return HeteroAKDLoss(
            student_channels=8,
            teacher_channels=12,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
            **kwargs,
        )

    def test_declares_the_tap_it_needs(self):
        assert self.make().required_features == ("taps.stage4",)

    def test_projectors_are_trainable_and_get_gradients(self, batch, features):
        student, teacher, labels = batch
        student_features, teacher_features = features
        criterion = self.make()

        result = criterion(
            student, teacher, labels,
            student_features=student_features, teacher_features=teacher_features,
        )
        result["total"].backward()

        assert list(criterion.parameters()), "проекторы обязаны быть обучаемыми"
        for name, parameter in criterion.named_parameters():
            assert parameter.grad is not None, f"{name} остался без градиента"

    def test_distillation_term_is_non_negative(self, batch, features):
        student, teacher, labels = batch
        result = self.make()(
            student, teacher, labels,
            student_features=features[0], teacher_features=features[1],
        )
        assert result["hakd"] >= 0

    def test_features_of_different_size_are_aligned(self, batch, features):
        """Учитель и ученик снимают признаки на разных страйдах — это и есть
        обычный случай разнородной пары, падать на нём нельзя."""
        student, teacher, labels = batch
        assert features[0]["taps.stage4"].shape[2:] != features[1]["taps.stage4"].shape[2:]
        self.make()(
            student, teacher, labels,
            student_features=features[0], teacher_features=features[1],
        )

    def test_missing_tap_is_reported(self, batch):
        student, teacher, labels = batch
        with pytest.raises(KeyError):
            self.make()(student, teacher, labels, student_features={}, teacher_features={})


class TestCompositeLoss:
    def make(self):
        return CompositeLoss(
            {
                "kd": FitNetsKD(
                    layers={
                        "taps.stage3": {
                            "student_channels": 8,
                            "teacher_channels": 12,
                            "weight": 1.0,
                        }
                    },
                    ce_weight=0.0,
                    hint_weight=1.0,
                    ignore_index=IGNORE_INDEX,
                ),
                "seg": DiceLoss(ce_weight=1.0, dice_weight=1.0, ignore_index=IGNORE_INDEX),
            },
            weights={"kd": 0.7, "seg": 0.3},
        )

    @pytest.fixture()
    def features(self):
        generator = torch.Generator().manual_seed(4)
        return (
            {"taps.stage3": torch.randn(2, 8, 4, 6, generator=generator)},
            {"taps.stage3": torch.randn(2, 12, 4, 6, generator=generator)},
        )

    def test_requirements_are_the_union_of_the_terms(self):
        criterion = self.make()
        assert criterion.requires_teacher is True
        assert criterion.required_features == ("taps.stage3",)

    def test_total_is_the_weighted_sum(self, batch, features):
        student, teacher, labels = batch
        criterion = self.make()
        result = criterion(
            student, teacher, labels,
            student_features=features[0], teacher_features=features[1],
        )
        expected = 0.7 * result["kd"] + 0.3 * result["seg"]
        assert torch.allclose(result["total"], expected)

    def test_component_names_are_prefixed(self, batch, features):
        student, teacher, labels = batch
        result = self.make()(
            student, teacher, labels,
            student_features=features[0], teacher_features=features[1],
        )
        assert {"kd", "seg", "kd_hint", "seg_dice", "seg_ce"} <= set(result)

    def test_adapters_of_the_terms_stay_visible_to_the_optimizer(self):
        """Параметры слагаемых обязаны быть параметрами композиции —
        иначе регрессор FitNets просто не будет обучаться."""
        criterion = self.make()
        assert len(list(criterion.parameters())) == len(
            list(criterion.losses["kd"].parameters())
        )

    def test_scratch_composition_needs_no_teacher(self):
        criterion = CompositeLoss(
            {"ce": CrossEntropy(ignore_index=IGNORE_INDEX), "dice": DiceLoss(ce_weight=0.0)}
        )
        assert criterion.requires_teacher is False

    def test_unknown_weight_is_rejected(self):
        with pytest.raises(ValueError, match="weights"):
            CompositeLoss({"ce": CrossEntropy()}, weights={"typo": 1.0})
