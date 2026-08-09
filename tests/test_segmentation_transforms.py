"""Совместные трансформы (изображение, маска): инварианты и новые аугментации.

Главный инвариант всего файла — маска обязана оставаться согласованной
с изображением и не приобретать значений, которых в ней не было.
"""

import random

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.transforms import build_segmentation_transform_train
from src.utils import segmentation_transforms as T

IGNORE_INDEX = 255
NUM_CLASSES = 4


def image_and_mask(height: int = 32, width: int = 48) -> tuple[Image.Image, torch.Tensor]:
    """Картинка со случайными цветами и маска с четырьмя классами и void'ом."""
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, : width // 4] = 1
    mask[:, width // 4 : width // 2] = 2
    mask[:, width // 2 : 3 * width // 4] = 3
    mask[:4, :4] = IGNORE_INDEX

    return image, torch.from_numpy(mask)


class TestRandomCropCatMaxRatio:
    def test_dominant_class_crops_are_rejected(self):
        """Кроп, где один класс занимает больше cat_max_ratio размеченных
        пикселей, должен перевыбираться. Маска здесь: левая половина — класс 0,
        правая — класс 1, поэтому годными являются только кропы, задевающие
        границу."""
        rng = np.random.default_rng(0)
        image = Image.fromarray(rng.integers(0, 256, (16, 64, 3), dtype=np.uint8))
        mask = torch.zeros(16, 64, dtype=torch.uint8)
        mask[:, 32:] = 1

        crop = T.SegmentationRandomCrop(
            crop_size=(16, 16),
            ignore_index=IGNORE_INDEX,
            cat_max_ratio=0.75,
            max_attempts=50,
        )

        random.seed(0)
        for _ in range(20):
            _, mask_crop = crop(image, mask)
            _, counts = torch.unique(mask_crop, return_counts=True)
            assert counts.numel() == 2, "кроп из одного класса должен был быть отброшен"
            assert counts.max().item() / counts.sum().item() < 0.75

    def test_ignore_pixels_do_not_count_as_a_class(self):
        """Доля считается от РАЗМЕЧЕННЫХ пикселей: void не должен создавать
        видимость разнообразия там, где размеченный класс всего один."""
        crop = T.SegmentationRandomCrop(
            crop_size=(8, 8), ignore_index=IGNORE_INDEX, cat_max_ratio=0.75
        )
        mask_crop = torch.full((8, 8), IGNORE_INDEX, dtype=torch.uint8)
        mask_crop[:4] = 1

        assert not crop._is_diverse_enough(mask_crop)

    def test_without_cat_max_ratio_the_first_crop_is_taken(self):
        """Дефолтное поведение обязано остаться прежним: один заход, без отбраковки."""
        image, mask = image_and_mask()
        crop = T.SegmentationRandomCrop(crop_size=(16, 16), ignore_index=IGNORE_INDEX)

        random.seed(0)
        cropped_image, cropped_mask = crop(image, mask)
        assert cropped_mask.shape == (16, 16)
        assert cropped_image.size == (16, 16)

    def test_padding_fills_mask_with_ignore_not_zero(self):
        """0 — валидный класс road; заполнять им паддинг нельзя."""
        image, mask = image_and_mask(height=8, width=8)
        crop = T.SegmentationRandomCrop(crop_size=(16, 16), ignore_index=IGNORE_INDEX)
        _, cropped_mask = crop(image, mask)

        assert cropped_mask.shape == (16, 16)
        assert (cropped_mask == IGNORE_INDEX).any()


class TestColorJitter:
    def test_probability_zero_is_identity(self):
        image, mask = image_and_mask()
        jitter = T.SegmentationColorJitter(brightness=0.9, p=0.0)

        random.seed(0)
        jittered, jittered_mask = jitter(image, mask)
        assert np.array_equal(np.array(jittered), np.array(image))
        assert torch.equal(jittered_mask, mask)

    def test_mask_is_never_touched(self):
        image, mask = image_and_mask()
        jitter = T.SegmentationColorJitter(brightness=0.9, hue=0.05, p=1.0)

        random.seed(0)
        jittered, jittered_mask = jitter(image, mask)
        assert not np.array_equal(np.array(jittered), np.array(image))
        assert torch.equal(jittered_mask, mask)

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError):
            T.SegmentationColorJitter(p=2.0)


class TestGaussianBlur:
    def test_blur_changes_the_image_and_not_the_mask(self):
        image, mask = image_and_mask()
        blur = T.SegmentationGaussianBlur(p=1.0, kernel_size=5, sigma=(2.0, 2.0))

        random.seed(0)
        blurred, blurred_mask = blur(image, mask)
        assert not np.array_equal(np.array(blurred), np.array(image))
        assert torch.equal(blurred_mask, mask)

    def test_even_kernel_size_raises(self):
        with pytest.raises(ValueError, match="нечётным"):
            T.SegmentationGaussianBlur(kernel_size=4)


class TestRandomErasing:
    def test_erases_a_rectangle_and_keeps_labels_by_default(self):
        """Метки под пятном сохраняются намеренно: модель обязана достроить
        класс по контексту, в этом вся аугментация."""
        random.seed(0)
        image = torch.zeros(3, 64, 64)
        mask = torch.ones(64, 64, dtype=torch.int64)

        erasing = T.SegmentationRandomErasing(p=1.0, scale=(0.1, 0.2), value=1.0)
        erased, erased_mask = erasing(image, mask)

        assert (erased == 1.0).any(), "стирание не сработало"
        assert torch.equal(erased_mask, mask)

    def test_erase_labels_marks_the_region_as_ignore(self):
        random.seed(0)
        image = torch.zeros(3, 64, 64)
        mask = torch.ones(64, 64, dtype=torch.int64)

        erasing = T.SegmentationRandomErasing(
            p=1.0, scale=(0.1, 0.2), value=1.0, erase_labels=True, ignore_index=IGNORE_INDEX
        )
        erased, erased_mask = erasing(image, mask)

        # Стёртой оказалась ровно та же область, что и на изображении.
        erased_pixels = (erased[0] == 1.0)
        assert erased_pixels.any()
        assert torch.equal(erased_mask == IGNORE_INDEX, erased_pixels)

    def test_probability_zero_is_identity(self):
        random.seed(0)
        image = torch.zeros(3, 32, 32)
        mask = torch.ones(32, 32, dtype=torch.int64)

        erased, erased_mask = T.SegmentationRandomErasing(p=0.0)(image, mask)
        assert erased is image
        assert erased_mask is mask

    def test_erased_area_respects_scale(self):
        random.seed(0)
        image = torch.zeros(3, 100, 100)
        mask = torch.zeros(100, 100, dtype=torch.int64)
        erasing = T.SegmentationRandomErasing(p=1.0, scale=(0.05, 0.10), value=1.0)

        for _ in range(30):
            erased, _ = erasing(image, mask)
            fraction = (erased[0] == 1.0).float().mean().item()
            assert 0.03 <= fraction <= 0.13, fraction

    def test_input_must_already_be_a_tensor(self):
        """Стирание задано в нормализованной шкале, поэтому обязано стоять
        после ToTensor/Normalize; PIL на входе — ошибка конфигурации."""
        image, mask = image_and_mask()
        with pytest.raises(TypeError, match="SegmentationToTensor"):
            T.SegmentationRandomErasing(p=1.0)(image, mask)

    @pytest.mark.parametrize(
        "kwargs",
        [{"p": 1.5}, {"scale": (0.5, 0.1)}, {"ratio": (0.0, 3.0)}, {"value": "mean"}],
    )
    def test_invalid_parameters_raise(self, kwargs):
        with pytest.raises(ValueError):
            T.SegmentationRandomErasing(**kwargs)


class TestTrainPipeline:
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def test_default_pipeline_is_unchanged(self):
        """Все новые аугментации выключены по умолчанию: базовый рецепт
        обязан остаться ровно тем, на котором посчитаны бейзлайны."""
        transform = build_segmentation_transform_train(
            mean=self.MEAN, std=self.STD, crop_size=(16, 24), color_jitter=0.4
        )
        assert [type(op).__name__ for op in transform.transforms] == [
            "SegmentationRandomScale",
            "SegmentationRandomCrop",
            "SegmentationRandomHorizontalFlip",
            "SegmentationColorJitter",
            "SegmentationToTensor",
            "SegmentationNormalize",
        ]

    def test_strong_pipeline_order_puts_erasing_after_normalize(self):
        """Стирание задано в нормализованной шкале — значит, оно обязано идти
        последним; размытие, наоборот, до ToTensor (дешевле на PIL)."""
        transform = build_segmentation_transform_train(
            mean=self.MEAN,
            std=self.STD,
            crop_size=(16, 24),
            color_jitter=0.4,
            hue=0.03,
            blur_p=0.3,
            random_erasing_p=0.3,
            cat_max_ratio=0.75,
        )
        assert [type(op).__name__ for op in transform.transforms] == [
            "SegmentationRandomScale",
            "SegmentationRandomCrop",
            "SegmentationRandomHorizontalFlip",
            "SegmentationColorJitter",
            "SegmentationGaussianBlur",
            "SegmentationToTensor",
            "SegmentationNormalize",
            "SegmentationRandomErasing",
        ]

    def test_strong_pipeline_output_shapes_and_dtypes(self):
        random.seed(0)
        image, mask = image_and_mask(height=64, width=96)
        transform = build_segmentation_transform_train(
            mean=self.MEAN,
            std=self.STD,
            crop_size=(32, 48),
            scale_range=(0.5, 2.0),
            ignore_index=IGNORE_INDEX,
            color_jitter=0.4,
            hue=0.03,
            blur_p=1.0,
            random_erasing_p=1.0,
            cat_max_ratio=0.75,
        )

        for _ in range(10):
            out_image, out_mask = transform(image, mask)
            assert out_image.shape == (3, 32, 48)
            assert out_mask.shape == (32, 48)
            assert out_mask.dtype == torch.int64
            # Маска не должна приобретать значений, которых в ней не было:
            # ни билинейная интерполяция, ни стирание не имеют права
            # изобретать новые номера классов.
            assert set(out_mask.unique().tolist()) <= {0, 1, 2, 3, IGNORE_INDEX}
