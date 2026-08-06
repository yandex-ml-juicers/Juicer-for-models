"""Дистилляция семантической сегментации: лоссы, модели и их стыковка.

Сетевых обращений тут нет: SegFormer собирается с pretrained=None, SegNeXt —
всегда локально. Тесты, которым нужны скачанные веса, помечены как integration.
"""

import pytest
import torch
import torch.nn.functional as F

from src.losses import ChannelWiseKD, DISTLoss, FitNetsKD, PixelWiseKD
from src.losses.segmentation_utils import subsample_spatially
from src.models import STAGE_TAPS, FeatureExtractor, SegFormer, SegNeXt

NUM_CLASSES = 5
IGNORE_INDEX = 255


@pytest.fixture()
def batch():
    """Логиты [B, C, H, W] и маска [B, H, W] с несколькими void-пикселями."""
    generator = torch.Generator().manual_seed(0)
    student_logits = torch.randn(2, NUM_CLASSES, 8, 12, generator=generator)
    teacher_logits = torch.randn(2, NUM_CLASSES, 8, 12, generator=generator)
    labels = torch.randint(0, NUM_CLASSES, (2, 8, 12), generator=generator)
    labels[0, 0, :4] = IGNORE_INDEX
    return student_logits, teacher_logits, labels


class TestPixelWiseKD:
    def test_alpha_zero_is_pure_ce(self, batch):
        student_logits, teacher_logits, labels = batch
        criterion = PixelWiseKD(alpha=0.0, ignore_index=IGNORE_INDEX)
        result = criterion(student_logits, teacher_logits, labels)
        expected = F.cross_entropy(student_logits, labels, ignore_index=IGNORE_INDEX)
        assert torch.allclose(result["total"], expected)

    def test_kd_is_zero_when_student_equals_teacher(self, batch):
        student_logits, _, labels = batch
        criterion = PixelWiseKD(alpha=1.0, ignore_index=IGNORE_INDEX)
        result = criterion(student_logits, student_logits.clone(), labels)
        assert result["kd"].abs().item() < 1e-6

    def test_ignore_index_excluded_from_ce(self, batch):
        """Void-пиксели не должны влиять на CE: подменяем их предсказания
        на мусор — значение CE обязано остаться прежним."""
        student_logits, teacher_logits, labels = batch
        criterion = PixelWiseKD(alpha=0.0, ignore_index=IGNORE_INDEX)
        before = criterion(student_logits, teacher_logits, labels)["ce"]

        corrupted = student_logits.clone()
        corrupted[0, :, 0, :4] = 50.0
        after = criterion(corrupted, teacher_logits, labels)["ce"]

        assert torch.allclose(before, after)

    def test_kd_does_not_scale_with_resolution(self, batch):
        """Главное отличие от HintonKD: KL усредняется по пикселям.

        Повторяем ту же карту 2x2 по пространству — распределение не
        изменилось, значит и лосс не должен. У reduction="batchmean"
        значение выросло бы вчетверо.
        """
        student_logits, teacher_logits, labels = batch
        criterion = PixelWiseKD(alpha=1.0, ignore_index=IGNORE_INDEX)

        small = criterion(student_logits, teacher_logits, labels)["kd"]
        big = criterion(
            student_logits.repeat(1, 1, 2, 2),
            teacher_logits.repeat(1, 1, 2, 2),
            labels.repeat(1, 2, 2),
        )["kd"]

        assert torch.allclose(small, big, atol=1e-5)

    def test_teacher_logits_are_interpolated_to_student_grid(self, batch):
        student_logits, _, labels = batch
        criterion = PixelWiseKD(ignore_index=IGNORE_INDEX)
        coarse_teacher = torch.randn(2, NUM_CLASSES, 4, 6)
        result = criterion(student_logits, coarse_teacher, labels)
        assert torch.isfinite(result["total"])

    def test_none_teacher_raises(self, batch):
        student_logits, _, labels = batch
        with pytest.raises(TypeError, match="requires_teacher"):
            PixelWiseKD()(student_logits, None, labels)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            PixelWiseKD(alpha=1.5)

    def test_class_count_mismatch_raises(self, batch):
        student_logits, _, labels = batch
        with pytest.raises(ValueError, match="классов"):
            PixelWiseKD()(student_logits, torch.randn(2, NUM_CLASSES + 1, 8, 12), labels)


class TestChannelWiseKD:
    def test_matches_paper_formula(self, batch):
        """Сверка с эталонной реализацией CWD: softmax по пространству,
        сумма KL по пикселям, нормировка на C и на размер батча."""
        student_logits, teacher_logits, labels = batch
        temperature = 4.0
        criterion = ChannelWiseKD(temperature=temperature, ignore_index=IGNORE_INDEX)
        result = criterion(student_logits, teacher_logits, labels)

        batch_size, channels = student_logits.shape[:2]
        teacher_flat = teacher_logits.reshape(-1, 8 * 12) / temperature
        student_flat = student_logits.reshape(-1, 8 * 12) / temperature
        teacher_probs = F.softmax(teacher_flat, dim=1)
        reference = torch.sum(
            teacher_probs * F.log_softmax(teacher_flat, dim=1)
            - teacher_probs * F.log_softmax(student_flat, dim=1)
        ) * temperature**2
        reference = reference / (channels * batch_size)

        assert torch.allclose(result["cwd"], reference, atol=1e-5)

    def test_zero_when_student_equals_teacher(self, batch):
        student_logits, _, labels = batch
        criterion = ChannelWiseKD(ignore_index=IGNORE_INDEX)
        result = criterion(student_logits, student_logits.clone(), labels)
        assert result["cwd"].abs().item() < 1e-5

    def test_total_is_weighted_sum(self, batch):
        student_logits, teacher_logits, labels = batch
        criterion = ChannelWiseKD(ce_weight=0.5, cwd_weight=3.0, ignore_index=IGNORE_INDEX)
        result = criterion(student_logits, teacher_logits, labels)
        assert torch.allclose(result["total"], 0.5 * result["ce"] + 3.0 * result["cwd"])

    def test_differs_from_pixel_wise_kd(self, batch):
        """Нормировка по разным осям — это разные методы, а не переобозначение."""
        student_logits, teacher_logits, labels = batch
        cwd = ChannelWiseKD(temperature=4.0, ignore_index=IGNORE_INDEX)(
            student_logits, teacher_logits, labels
        )["cwd"]
        kd = PixelWiseKD(temperature=4.0, ignore_index=IGNORE_INDEX)(
            student_logits, teacher_logits, labels
        )["kd"]
        assert not torch.allclose(cwd, kd)


class TestDIST:
    def test_relations_vanish_for_identical_logits(self, batch):
        student_logits, _, labels = batch
        criterion = DISTLoss(ignore_index=IGNORE_INDEX)
        result = criterion(student_logits, student_logits.clone(), labels)
        assert result["inter"].abs().item() < 1e-4
        assert result["intra"].abs().item() < 1e-4

    def test_total_is_weighted_sum(self, batch):
        student_logits, teacher_logits, labels = batch
        criterion = DISTLoss(ce_weight=0.5, beta=2.0, gamma=3.0, ignore_index=IGNORE_INDEX)
        result = criterion(student_logits, teacher_logits, labels)
        expected = 0.5 * result["ce"] + 2.0 * result["inter"] + 3.0 * result["intra"]
        assert torch.allclose(result["total"], expected)

    def test_matches_official_dist_per_image(self, batch):
        """Эталон — официальная реализация DIST, применённая к каждой
        картинке отдельно с пикселями в роли примеров батча."""
        student_logits, teacher_logits, labels = batch
        temperature = 4.0
        criterion = DISTLoss(temperature=temperature, ignore_index=IGNORE_INDEX)
        result = criterion(student_logits, teacher_logits, labels)

        def cosine(a, b, eps=1e-8):
            return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)

        def pearson(a, b):
            return cosine(a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1))

        def inter_relation(y_s, y_t):
            return 1 - pearson(y_s, y_t).mean()

        def intra_relation(y_s, y_t):
            return inter_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))

        inters, intras = [], []
        for index in range(student_logits.shape[0]):
            y_s = (student_logits[index].reshape(NUM_CLASSES, -1).T / temperature).softmax(dim=1)
            y_t = (teacher_logits[index].reshape(NUM_CLASSES, -1).T / temperature).softmax(dim=1)
            inters.append(inter_relation(y_s, y_t))
            intras.append(intra_relation(y_s, y_t))

        assert torch.allclose(
            result["inter"], temperature**2 * torch.stack(inters).mean(), atol=1e-5
        )
        assert torch.allclose(
            result["intra"], temperature**2 * torch.stack(intras).mean(), atol=1e-5
        )

    def test_spatial_stride_selects_every_nth_pixel(self):
        tensor = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        subsampled = subsample_spatially(tensor, 2)
        assert subsampled.shape == (1, 1, 2, 2)
        assert subsampled.flatten().tolist() == [0.0, 2.0, 8.0, 10.0]
        assert subsample_spatially(tensor, 1) is tensor

    def test_stride_is_exact_on_tiled_input(self, batch):
        """На карте, размноженной 2x2, прореживание с шагом 2 обязано вернуть
        ровно тот же лосс, что и исходная карта без прореживания."""
        student_logits, teacher_logits, labels = batch
        base = DISTLoss(spatial_stride=1, ignore_index=IGNORE_INDEX)(
            student_logits, teacher_logits, labels
        )
        strided = DISTLoss(spatial_stride=2, ignore_index=IGNORE_INDEX)(
            student_logits.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3),
            teacher_logits.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3),
            labels.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2),
        )
        assert torch.allclose(base["inter"], strided["inter"], atol=1e-5)
        assert torch.allclose(base["intra"], strided["intra"], atol=1e-5)

    def test_invalid_stride_raises(self):
        with pytest.raises(ValueError):
            DISTLoss(spatial_stride=0)


class TestFitNets:
    LAYERS = {"taps.stage3": {"student_channels": 4, "teacher_channels": 4, "weight": 1.0}}

    def _identity_criterion(self, **kwargs) -> FitNetsKD:
        criterion = FitNetsKD(layers=self.LAYERS, ignore_index=IGNORE_INDEX, **kwargs)
        with torch.no_grad():
            criterion.adapters.adapters["taps__stage3"].weight.copy_(
                torch.eye(4).reshape(4, 4, 1, 1)
            )
        return criterion

    def test_declares_required_features(self):
        criterion = FitNetsKD(layers=self.LAYERS)
        assert criterion.required_features == ("taps.stage3",)

    def test_dotted_tap_name_survives_module_dict(self):
        """nn.ModuleDict запрещает точку в ключе, а имена тапов — это пути
        модулей; лосс обязан сам разруливать это соответствие."""
        criterion = FitNetsKD(layers=self.LAYERS)
        assert "taps__stage3" in criterion.adapters.adapters
        assert criterion.adapter_keys["taps.stage3"] == "taps__stage3"

    def test_adapters_are_trainable(self):
        criterion = FitNetsKD(layers=self.LAYERS)
        params = list(criterion.parameters())
        assert params and all(p.requires_grad for p in params)

    def test_hint_is_zero_for_identical_features(self, batch):
        student_logits, teacher_logits, labels = batch
        criterion = self._identity_criterion()
        features = {"taps.stage3": torch.randn(2, 4, 8, 12)}
        result = criterion(
            student_logits,
            teacher_logits,
            labels,
            student_features=features,
            teacher_features={"taps.stage3": features["taps.stage3"].clone()},
        )
        assert result["hint"].abs().item() < 1e-6
        assert result["hint_taps.stage3"].abs().item() < 1e-6

    def test_spatial_mismatch_is_interpolated(self, batch):
        student_logits, teacher_logits, labels = batch
        criterion = self._identity_criterion()
        result = criterion(
            student_logits,
            teacher_logits,
            labels,
            student_features={"taps.stage3": torch.randn(2, 4, 4, 6)},
            teacher_features={"taps.stage3": torch.randn(2, 4, 8, 12)},
        )
        assert torch.isfinite(result["total"])

    def test_logits_term_is_off_by_default(self, batch):
        student_logits, teacher_logits, labels = batch
        criterion = self._identity_criterion()
        features = {"taps.stage3": torch.randn(2, 4, 8, 12)}
        result = criterion(
            student_logits,
            teacher_logits,
            labels,
            student_features=features,
            teacher_features={"taps.stage3": torch.randn(2, 4, 8, 12)},
        )
        assert "kd" not in result
        assert torch.allclose(result["total"], result["ce"] + result["hint"])

    def test_logits_term_enabled(self, batch):
        student_logits, teacher_logits, labels = batch
        criterion = self._identity_criterion(logits_weight=2.0)
        features = {"taps.stage3": torch.randn(2, 4, 8, 12)}
        result = criterion(
            student_logits,
            teacher_logits,
            labels,
            student_features=features,
            teacher_features={"taps.stage3": torch.randn(2, 4, 8, 12)},
        )
        assert torch.allclose(
            result["total"], result["ce"] + result["hint"] + 2.0 * result["kd"]
        )

    def test_missing_feature_raises(self, batch):
        student_logits, teacher_logits, labels = batch
        criterion = FitNetsKD(layers=self.LAYERS)
        with pytest.raises(KeyError, match="stage3"):
            criterion(
                student_logits,
                teacher_logits,
                labels,
                student_features={},
                teacher_features={},
            )

    def test_requires_features(self, batch):
        student_logits, teacher_logits, labels = batch
        with pytest.raises(TypeError, match="карты признаков"):
            FitNetsKD(layers=self.LAYERS)(student_logits, teacher_logits, labels)

    def test_empty_layers_raise(self):
        with pytest.raises(ValueError):
            FitNetsKD(layers={})


class TestSegNeXt:
    @pytest.mark.parametrize(
        "variant,expected_millions", [("tiny", 4.3), ("small", 13.9)]
    )
    def test_parameter_count_matches_paper(self, variant, expected_millions):
        model = SegNeXt(variant=variant, num_classes=19)
        millions = sum(p.numel() for p in model.parameters()) / 1e6
        assert abs(millions - expected_millions) < 0.15, millions

    def test_logits_have_input_resolution(self):
        model = SegNeXt(variant="tiny", num_classes=NUM_CLASSES)
        logits = model(torch.randn(2, 3, 64, 96))
        assert logits.shape == (2, NUM_CLASSES, 64, 96)

    def test_backward_reaches_every_parameter(self):
        model = SegNeXt(variant="tiny", num_classes=NUM_CLASSES)
        model(torch.randn(1, 3, 64, 64)).sum().backward()
        without_grad = [name for name, p in model.named_parameters() if p.grad is None]
        assert not without_grad, without_grad

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            SegNeXt(variant="huge")

    def test_nmf_is_finite_on_degenerate_input(self):
        """NMF делит на знаменатели, которые на нулевом входе обращаются
        в ноль: eps обязан удержать результат конечным."""
        from src.models.segnext import NMF2D

        module = NMF2D(num_bases=4)
        output = module(torch.zeros(1, 8, 4, 4))
        assert torch.isfinite(output).all()


class TestSegFormer:
    def test_logits_have_input_resolution(self):
        model = SegFormer(variant="b0", num_classes=NUM_CLASSES, pretrained=None)
        logits = model(torch.randn(2, 3, 64, 96))
        assert logits.shape == (2, NUM_CLASSES, 64, 96)

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            SegFormer(variant="b9", pretrained=None)

    def test_unknown_pretrained_raises(self):
        with pytest.raises(ValueError):
            SegFormer(variant="b0", pretrained="coco")


class TestFeatureTapsInterop:
    """Ради чего вообще заведены тапы: снять признаки с двух РАЗНЫХ
    архитектур одним и тем же списком имён слоёв."""

    def test_both_models_expose_the_same_tap_names(self):
        student = SegNeXt(variant="tiny", num_classes=NUM_CLASSES)
        teacher = SegFormer(variant="b0", num_classes=NUM_CLASSES, pretrained=None)

        student_taps = {n for n, _ in student.named_modules() if n.startswith("taps.")}
        teacher_taps = {n for n, _ in teacher.named_modules() if n.startswith("taps.")}
        expected = {f"taps.{name}" for name in STAGE_TAPS}

        assert student_taps == teacher_taps == expected

    def test_feature_extractor_captures_matching_stages(self):
        """SegNeXt-T и SegFormer-B0 имеют одинаковые ширины стадий
        [32, 64, 160, 256], поэтому карты обязаны совпасть и по форме."""
        layers = [f"taps.{name}" for name in STAGE_TAPS]
        student = SegNeXt(variant="tiny", num_classes=NUM_CLASSES)
        teacher = SegFormer(variant="b0", num_classes=NUM_CLASSES, pretrained=None)

        student_extractor = FeatureExtractor(student, layers)
        teacher_extractor = FeatureExtractor(teacher, layers)
        try:
            images = torch.randn(1, 3, 64, 96)
            student(images)
            teacher(images)

            assert set(student_extractor.features) == set(layers)
            for layer in layers:
                assert (
                    student_extractor.features[layer].shape
                    == teacher_extractor.features[layer].shape
                )
        finally:
            student_extractor.remove()
            teacher_extractor.remove()

    def test_taps_add_no_parameters_and_no_state(self):
        """Тапы не должны попадать ни в optimizer, ни в чекпоинт."""
        model = SegNeXt(variant="tiny", num_classes=NUM_CLASSES)
        assert list(model.taps.parameters()) == []
        assert model.taps.state_dict() == {}
        assert not any(key.startswith("taps.") for key in model.state_dict())


class TestAutocast:
    """Обучение идёт с amp=true, а Trainer зовёт и модель, и лосс ВНУТРИ
    torch.autocast. Проверяем на CPU/bfloat16 — механика autocast та же,
    что на CUDA/fp16.

    Тонкость, ради которой тесты и написаны: autocast перехватывает
    операции по СПИСКУ, а не по типу аргументов. Вызов .float() перед bmm
    или conv2d ничего не гарантирует — autocast приведёт аргументы обратно.
    """

    def test_nmf_stays_fp32_inside_autocast(self):
        """NMF делит на почти нулевые знаменатели; в bf16 это NaN."""
        from src.models.segnext import NMF2D

        captured = {}
        module = NMF2D(num_bases=4)
        original_step = module._update_coef

        def spy(x, bases, coef):
            result = original_step(x, bases, coef)
            captured["dtype"] = result.dtype
            return result

        module._update_coef = spy
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = module(torch.randn(1, 8, 4, 6).abs())

        assert captured["dtype"] == torch.float32, "итерации NMF ушли в половинную точность"
        assert torch.isfinite(output).all()

    def test_segnext_forward_is_finite_under_autocast(self):
        model = SegNeXt(variant="tiny", num_classes=NUM_CLASSES)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(torch.randn(1, 3, 64, 96))
        assert torch.isfinite(logits.float()).all()

    @pytest.mark.parametrize(
        "criterion_factory",
        [
            lambda: PixelWiseKD(ignore_index=IGNORE_INDEX),
            lambda: ChannelWiseKD(ignore_index=IGNORE_INDEX),
            lambda: DISTLoss(ignore_index=IGNORE_INDEX),
            lambda: FitNetsKD(
                layers={
                    "taps.stage3": {
                        "student_channels": 4,
                        "teacher_channels": 4,
                        "weight": 1.0,
                    }
                },
                ignore_index=IGNORE_INDEX,
            ),
        ],
    )
    def test_losses_are_finite_and_fp32_under_autocast(self, criterion_factory, batch):
        student_logits, teacher_logits, labels = batch
        criterion = criterion_factory()

        with torch.autocast("cpu", dtype=torch.bfloat16):
            losses = criterion(
                student_logits.to(torch.bfloat16),
                teacher_logits.to(torch.bfloat16),
                labels,
                student_features={"taps.stage3": torch.randn(2, 4, 8, 12, dtype=torch.bfloat16)},
                teacher_features={"taps.stage3": torch.randn(2, 4, 8, 12, dtype=torch.bfloat16)},
            )

        for name, value in losses.items():
            assert torch.isfinite(value), name
            # Скаляры лосса должны выходить в fp32: GradScaler масштабирует
            # именно total, и половинная точность здесь режет динамический диапазон.
            assert value.dtype == torch.float32, f"{name}: {value.dtype}"


class TestOptimizationStep:
    """Каждый лосс должен давать ненулевой градиент на ученике —
    иначе метод "работает", но ничему не учит."""

    @pytest.mark.parametrize(
        "criterion_factory",
        [
            lambda: PixelWiseKD(ignore_index=IGNORE_INDEX),
            lambda: ChannelWiseKD(ignore_index=IGNORE_INDEX),
            lambda: DISTLoss(ignore_index=IGNORE_INDEX),
            lambda: FitNetsKD(
                layers={
                    "taps.stage3": {
                        "student_channels": 160,
                        "teacher_channels": 160,
                        "weight": 1.0,
                    }
                },
                ignore_index=IGNORE_INDEX,
            ),
        ],
    )
    def test_single_step_updates_student(self, criterion_factory):
        torch.manual_seed(0)
        student = SegNeXt(variant="tiny", num_classes=NUM_CLASSES)
        teacher = SegFormer(variant="b0", num_classes=NUM_CLASSES, pretrained=None)
        teacher.eval()
        teacher.requires_grad_(False)

        criterion = criterion_factory()
        layers = list(criterion.required_features)
        student_extractor = FeatureExtractor(student, layers) if layers else None
        teacher_extractor = FeatureExtractor(teacher, layers) if layers else None

        try:
            images = torch.randn(1, 3, 64, 96)
            labels = torch.randint(0, NUM_CLASSES, (1, 64, 96))
            labels[0, :4] = IGNORE_INDEX

            with torch.no_grad():
                teacher_logits = teacher(images)

            losses = criterion(
                student(images),
                teacher_logits,
                labels,
                student_features=student_extractor.features if student_extractor else None,
                teacher_features=teacher_extractor.features if teacher_extractor else None,
            )
            losses["total"].backward()

            head_grad = student.decode_head.classifier.weight.grad
            assert head_grad is not None and head_grad.abs().sum() > 0
            assert torch.isfinite(losses["total"])
        finally:
            if student_extractor is not None:
                student_extractor.remove()
            if teacher_extractor is not None:
                teacher_extractor.remove()
