"""Второй вид кадра для учителя: ученик видит сильные аугментации,
учитель — слабые, а геометрия у них общая.
"""

import random

import pytest
import torch
from PIL import Image

from src.data.batch_augment import MixupCutmix
from src.data.transforms import build_segmentation_transform_train
from src.training.trainer import split_segmentation_batch
from src.utils.segmentation_transforms import (
    SegmentationCompose,
    SegmentationTeacherViewCompose,
    SegmentationToTensor,
    StudentOnly,
)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IGNORE_INDEX = 255


def sample(height: int = 64, width: int = 96):
    """Кадр с неоднородной картинкой (иначе размытие ничего не меняет) и маска."""
    generator = torch.Generator().manual_seed(0)
    pixels = torch.randint(0, 255, (height, width, 3), dtype=torch.uint8, generator=generator)
    image = Image.fromarray(pixels.numpy(), mode="RGB")
    mask = torch.randint(0, 4, (height, width), dtype=torch.uint8, generator=generator)
    return image, mask


def build(teacher_skips, **kwargs):
    return build_segmentation_transform_train(
        mean=MEAN,
        std=STD,
        crop_size=(32, 48),
        scale_range=(1.0, 1.0),
        ignore_index=IGNORE_INDEX,
        teacher_skips=teacher_skips,
        **kwargs,
    )


class TestBuilder:
    def test_no_skips_keeps_the_usual_pair(self):
        transform = build(teacher_skips=(), blur_p=0.5)
        assert isinstance(transform, SegmentationCompose)
        assert len(transform(*sample())) == 2

    def test_skips_switch_on_the_second_view(self):
        transform = build(teacher_skips=["blur"], blur_p=1.0)
        assert isinstance(transform, SegmentationTeacherViewCompose)
        assert len(transform(*sample())) == 3

    def test_unknown_skip_is_rejected(self):
        with pytest.raises(ValueError, match="teacher_skips"):
            build(teacher_skips=["mixup"])

    def test_disabled_augmentation_cannot_be_skipped_twice(self):
        """teacher_skips на выключенной аугментации ничего не меняет:
        расходиться видам не на чем."""
        transform = build(teacher_skips=["blur"], blur_p=0.0, color_jitter=0.0)
        assert isinstance(transform, SegmentationCompose)


class TestTwoViews:
    def test_geometry_is_shared_and_only_photometry_differs(self):
        """Главный инвариант: виды отличаются, но пиксель (y, x) на обоих
        описывает одно и то же место сцены — иначе таргеты учителя
        перестанут совпадать с маской."""
        transform = build(teacher_skips=["blur", "erasing"], blur_p=1.0, random_erasing_p=1.0)
        student_image, teacher_image, mask = transform(*sample())

        assert student_image.shape == teacher_image.shape == (3, 32, 48)
        assert mask.shape == (32, 48)
        assert not torch.allclose(student_image, teacher_image)

    def test_teacher_view_equals_the_pipeline_without_those_augmentations(self):
        """Вид учителя обязан совпасть с тем, что дал бы трансформ без
        пропущенных аугментаций вовсе."""
        image, mask = sample()

        # Геометрия трансформов случайна через модуль random, поэтому
        # сравнивать два прогона можно только с одинакового зерна.
        random.seed(0)
        transform = build(teacher_skips=["blur"], blur_p=1.0, color_jitter=0.0, hflip_p=0.0)
        _, teacher_image, _ = transform(image, mask)

        random.seed(0)
        plain = build(teacher_skips=(), blur_p=0.0, color_jitter=0.0, hflip_p=0.0)
        clean_image, _ = plain(image, mask)

        assert torch.allclose(teacher_image, clean_image)

    def test_random_shared_operation_after_the_branch_is_rejected(self):
        """Случайную общую операцию после ветвления два вида прошли бы
        разными бросками и разъехались."""
        with pytest.raises(ValueError, match="StudentOnly"):
            SegmentationTeacherViewCompose(
                [
                    StudentOnly(SegmentationToTensor()),
                    _RandomShift(),
                ]
            )

    def test_compose_without_any_branch_is_rejected(self):
        with pytest.raises(ValueError, match="StudentOnly"):
            SegmentationTeacherViewCompose([SegmentationToTensor()])


class _RandomShift:
    """Заглушка со случайностью — ни одна из детерминированных операций."""

    def __call__(self, image, mask):
        return image + torch.rand(1).item(), mask


class TestBatchLevel:
    def test_cutmix_moves_the_same_box_in_both_views(self):
        """Если прямоугольник переносится в видах по-разному, учитель и
        ученик увидят разные сцены, и дистилляция станет шумом."""
        images = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
        teacher_images = images.clone() + 1000.0
        masks = torch.randint(0, 4, (2, 8, 8))

        augment = MixupCutmix(mixup_alpha=0.0, cutmix_alpha=1.0, prob=1.0)
        torch.manual_seed(0)
        mixed = augment(images, masks, teacher_images)

        assert mixed.teacher_images is not None
        # Вид учителя отличается от ученического ровно на ту же константу,
        # что и до смешивания, — значит, куски взяты из тех же мест.
        assert torch.allclose(mixed.teacher_images - 1000.0, mixed.images)

    def test_batch_without_teacher_view_stays_a_pair(self):
        images = torch.zeros(2, 3, 4, 4)
        masks = torch.zeros(2, 4, 4, dtype=torch.long)

        augment = MixupCutmix(mixup_alpha=0.0, cutmix_alpha=1.0, prob=1.0)
        assert augment(images, masks).teacher_images is None

    def test_trainer_unpacks_both_batch_shapes(self):
        images = torch.zeros(2, 3, 4, 4)
        masks = torch.zeros(2, 4, 4, dtype=torch.long)
        teacher_images = torch.ones(2, 3, 4, 4)

        assert split_segmentation_batch((images, masks))[1] is None
        unpacked = split_segmentation_batch((images, teacher_images, masks))
        assert unpacked[1] is teacher_images and unpacked[2] is masks
