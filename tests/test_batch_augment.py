"""Mixup/CutMix: смешивание батча и его стыковка с лоссами проекта."""

import random

import pytest
import torch
import torch.nn.functional as F

from src.data.batch_augment import (
    MixedBatch,
    MixupCutmix,
    interpolate_losses,
    random_bbox,
    sample_lambda,
)
from src.losses import CrossEntropy, PixelWiseKD
from src.training.trainer import compute_losses, mix_batch

NUM_CLASSES = 5
IGNORE_INDEX = 255


@pytest.fixture()
def classification_batch():
    generator = torch.Generator().manual_seed(0)
    images = torch.randn(8, 3, 16, 16, generator=generator)
    labels = torch.randint(0, NUM_CLASSES, (8,), generator=generator)
    return images, labels


@pytest.fixture()
def segmentation_batch():
    generator = torch.Generator().manual_seed(0)
    images = torch.randn(4, 3, 16, 24, generator=generator)
    masks = torch.randint(0, NUM_CLASSES, (4, 16, 24), generator=generator)
    return images, masks


class TestSampling:
    def test_lambda_keeps_the_original_sample_dominant(self):
        """lam >= 0.5 — то, на чём держится осмысленность train-метрик:
        тренер считает их по targets_a."""
        random.seed(0)
        assert all(sample_lambda(1.0) >= 0.5 for _ in range(200))
        assert all(0.5 <= sample_lambda(0.2) <= 1.0 for _ in range(200))

    def test_bbox_area_matches_requested_fraction(self):
        """Площадь прямоугольника — (1 - lam) от кадра. Проверяется в среднем:
        центр берётся равномерно, поэтому отдельный прямоугольник может быть
        обрезан границей — ровно из-за этого lam и пересчитывается по факту."""
        random.seed(0)
        height, width = 200, 200
        areas = []
        for _ in range(500):
            top, left, bottom, right = random_bbox(height, width, 0.75)
            areas.append((bottom - top) * (right - left) / (height * width))

        assert 0.10 < sum(areas) / len(areas) <= 0.25

    def test_bbox_stays_inside_the_frame(self):
        random.seed(0)
        for _ in range(200):
            top, left, bottom, right = random_bbox(37, 53, random.random())
            assert 0 <= top <= bottom <= 37
            assert 0 <= left <= right <= 53


class TestConstruction:
    def test_both_alphas_zero_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            MixupCutmix(mixup_alpha=0.0, cutmix_alpha=0.0)

    @pytest.mark.parametrize("kwargs", [{"prob": 1.5}, {"switch_prob": -0.1}, {"mixup_alpha": -1}])
    def test_invalid_parameters_raise(self, kwargs):
        with pytest.raises(ValueError):
            MixupCutmix(**kwargs)

    def test_unsupported_target_shape_raises(self):
        augment = MixupCutmix(prob=1.0)
        with pytest.raises(ValueError, match=r"\[B\]"):
            augment(torch.randn(4, 3, 8, 8), torch.randn(4, 2, 8, 8))


class TestClassification:
    def test_prob_zero_leaves_the_batch_untouched(self, classification_batch):
        images, labels = classification_batch
        augment = MixupCutmix(prob=0.0)
        mixed = augment(images, labels)

        assert mixed.images is images
        assert mixed.targets_a is labels
        assert mixed.lam == 1.0

    def test_single_example_batch_is_not_mixed(self):
        """Перестановка батча из одного элемента тождественна — смешивать не с чем."""
        augment = MixupCutmix(prob=1.0)
        mixed = augment(torch.randn(1, 3, 8, 8), torch.tensor([2]))
        assert mixed.lam == 1.0

    def test_mixup_output_is_a_convex_combination(self, classification_batch):
        """Каждый пиксель смеси обязан лежать между исходными значениями
        двух картинок — это и есть определение mixup."""
        random.seed(0)
        torch.manual_seed(0)
        images, labels = classification_batch
        augment = MixupCutmix(mixup_alpha=1.0, cutmix_alpha=0.0, prob=1.0)
        mixed = augment(images, labels)

        assert mixed.images.min() >= images.min() - 1e-5
        assert mixed.images.max() <= images.max() + 1e-5
        assert 0.5 <= mixed.lam <= 1.0
        assert not torch.equal(mixed.images, images)

    def test_cutmix_pixels_come_from_one_of_the_two_images(self, classification_batch):
        """Отличие CutMix от mixup: каждый пиксель результата — настоящий
        пиксель одной из двух картинок, а не их среднее."""
        random.seed(0)
        torch.manual_seed(0)
        images, labels = classification_batch
        augment = MixupCutmix(mixup_alpha=0.0, cutmix_alpha=1.0, prob=1.0)
        mixed = augment(images, labels)

        originals = images.reshape(-1)
        assert torch.isin(mixed.images.reshape(-1), originals).all()

    def test_cutmix_lambda_matches_the_untouched_area(self, classification_batch):
        """lam обязан считаться по ФАКТИЧЕСКОЙ площади вставки: доля пикселей,
        совпавших с исходной картинкой, должна равняться lam."""
        random.seed(0)
        torch.manual_seed(0)
        images, labels = classification_batch
        augment = MixupCutmix(mixup_alpha=0.0, cutmix_alpha=1.0, prob=1.0)

        for _ in range(20):
            mixed = augment(images, labels)
            untouched = (mixed.images[0] == images[0]).all(dim=0).float().mean().item()
            # Совпадение может быть и случайным (перестановка иногда оставляет
            # элемент на месте), поэтому сравнение — снизу.
            assert untouched >= mixed.lam - 1e-5


class TestSegmentation:
    def test_cutmix_moves_the_mask_with_the_pixels(self, segmentation_batch):
        """Смысл CutMix для плотных задач: вместе с куском изображения
        переносится кусок маски, поэтому таргет остаётся ТОЧНЫМ и lam=1."""
        random.seed(0)
        torch.manual_seed(0)
        images, masks = segmentation_batch
        augment = MixupCutmix(mixup_alpha=0.0, cutmix_alpha=1.0, prob=1.0)
        mixed = augment(images, masks)

        assert mixed.lam == 1.0
        assert mixed.targets_a is mixed.targets_b
        # Там, где изображение осталось прежним, обязана остаться и маска.
        unchanged = (mixed.images == images).all(dim=1)
        assert torch.equal(mixed.targets_a[unchanged], masks[unchanged])

    def test_cutmix_mask_values_come_from_the_original_masks(self, segmentation_batch):
        random.seed(0)
        torch.manual_seed(0)
        images, masks = segmentation_batch
        augment = MixupCutmix(mixup_alpha=0.0, cutmix_alpha=1.0, prob=1.0)
        mixed = augment(images, masks)

        assert mixed.targets_a.dtype == masks.dtype
        assert torch.isin(mixed.targets_a, masks).all()

    def test_mixup_keeps_two_target_sets(self, segmentation_batch):
        """У mixup точного таргета нет — маски остаются разными,
        и лосс интерполируется."""
        random.seed(0)
        torch.manual_seed(0)
        images, masks = segmentation_batch
        augment = MixupCutmix(mixup_alpha=1.0, cutmix_alpha=0.0, prob=1.0)
        mixed = augment(images, masks)

        assert mixed.lam < 1.0
        assert mixed.targets_a.shape == mixed.targets_b.shape == masks.shape

    def test_mask_image_size_mismatch_raises(self):
        augment = MixupCutmix(mixup_alpha=0.0, cutmix_alpha=1.0, prob=1.0)
        with pytest.raises(ValueError, match="совпадали"):
            augment(torch.randn(4, 3, 16, 16), torch.zeros(4, 8, 8, dtype=torch.int64))


class TestLossInterpolation:
    def test_interpolation_of_mixed_ce_matches_soft_target_ce(self):
        """Главное обоснование всей схемы: интерполяция ДВУХ обычных CE по
        целым меткам тождественна одной CE по смешанному one-hot таргету.
        Если это равенство ломается, mixup обучает не тому, что заявлено.
        """
        torch.manual_seed(0)
        logits = torch.randn(6, NUM_CLASSES)
        labels_a = torch.randint(0, NUM_CLASSES, (6,))
        labels_b = torch.randint(0, NUM_CLASSES, (6,))
        lam = 0.7

        criterion = CrossEntropy()
        interpolated = interpolate_losses(
            criterion(logits, None, labels_a),
            criterion(logits, None, labels_b),
            lam,
        )["total"]

        soft_targets = (
            lam * F.one_hot(labels_a, NUM_CLASSES).float()
            + (1 - lam) * F.one_hot(labels_b, NUM_CLASSES).float()
        )
        reference = -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

        assert torch.allclose(interpolated, reference, atol=1e-6)

    def test_distillation_terms_survive_interpolation_unchanged(self):
        """KD-слагаемое от меток не зависит, поэтому в обеих ветках оно одно
        и то же и смешивание его не искажает — иначе mixup незаметно
        ослаблял бы саму дистилляцию."""
        torch.manual_seed(0)
        student_logits = torch.randn(2, NUM_CLASSES, 8, 8)
        teacher_logits = torch.randn(2, NUM_CLASSES, 8, 8)
        masks_a = torch.randint(0, NUM_CLASSES, (2, 8, 8))
        masks_b = torch.randint(0, NUM_CLASSES, (2, 8, 8))

        criterion = PixelWiseKD(ignore_index=IGNORE_INDEX)
        plain = criterion(student_logits, teacher_logits, masks_a)
        mixed = compute_losses(
            criterion,
            student_logits,
            teacher_logits,
            MixedBatch(student_logits, masks_a, masks_b, 0.6),
        )

        assert torch.allclose(mixed["kd"], plain["kd"], atol=1e-6)
        assert not torch.allclose(mixed["ce"], plain["ce"])

    def test_lambda_one_skips_the_second_criterion_call(self):
        """При lam=1 (например, CutMix по маскам) второй набор таргетов
        не должен влиять ни на что — иначе на нём считался бы лишний лосс."""
        torch.manual_seed(0)
        logits = torch.randn(4, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (4,))
        garbage = torch.zeros_like(labels)

        criterion = CrossEntropy()
        result = compute_losses(
            criterion, logits, None, MixedBatch(logits, labels, garbage, 1.0)
        )
        assert torch.allclose(result["total"], criterion(logits, None, labels)["total"])


class TestTrainerHook:
    def test_mix_batch_without_augment_is_a_pass_through(self, classification_batch):
        images, labels = classification_batch
        mixed = mix_batch(None, images, labels)

        assert mixed.images is images
        assert mixed.targets_a is labels
        assert mixed.targets_b is labels
        assert mixed.lam == 1.0

    def test_mix_batch_delegates_to_the_augment(self, segmentation_batch):
        random.seed(0)
        torch.manual_seed(0)
        images, masks = segmentation_batch
        mixed = mix_batch(MixupCutmix(mixup_alpha=0.0, cutmix_alpha=1.0, prob=1.0), images, masks)

        assert mixed.images.shape == images.shape
        assert not torch.equal(mixed.images, images)
