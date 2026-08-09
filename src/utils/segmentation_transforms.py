"""Совместные трансформы (изображение, маска) для семантической сегментации.

Полный аналог detection_transforms.py, но вторым аргументом идёт не dict с
боксами, а маска [H, W] — целочисленная карта классов.

Два инварианта, которые здесь соблюдаются везде:
- маска интерполируется ТОЛЬКО ближайшим соседом; любая билинейная
  интерполяция смешала бы номера классов и породила несуществующие метки;
- маска остаётся целочисленной (uint8) до самого конца пайплайна и переводится
  в int64 один раз, в SegmentationToTensor. Именно поэтому uint8, а не int64:
  torchvision-трансформы для тензоров рассчитаны на uint8/float, а trainId'ы
  Cityscapes (0..18 и 255 = ignore) в uint8 помещаются точно.
"""

import random
from collections.abc import Callable, Sequence

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.transforms import functional as F


def image_size(image: Image.Image | Tensor) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        width, height = image.size
        return height, width

    return int(image.shape[-2]), int(image.shape[-1])


def resize_mask(mask: Tensor, size: tuple[int, int]) -> Tensor:
    """Ресайз маски [H, W] ближайшим соседом.

    F.resize ждёт минимум [C, H, W], поэтому канал добавляется и снимается.
    """
    resized = F.resize(
        mask.unsqueeze(0),
        size=[size[0], size[1]],
        interpolation=InterpolationMode.NEAREST,
    )
    return resized.squeeze(0)


class SegmentationCompose:
    """Последовательно применяет трансформы к паре (изображение, маска)."""

    def __init__(
        self,
        transforms: Sequence[Callable],
    ) -> None:
        self.transforms = transforms

    def __call__(
        self,
        image: Image.Image | Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        for transform in self.transforms:
            image, mask = transform(image, mask)

        return image, mask


class SegmentationRandomScale:
    """Случайный масштаб из scale_range — базовая аугментация Cityscapes.

    Масштаб выбирается равномерно, картинка меняет размер целиком; кадрирование
    до фиксированного размера делает следующий за ним SegmentationRandomCrop.
    """

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.5, 2.0),
    ) -> None:
        minimum, maximum = float(scale_range[0]), float(scale_range[1])

        if minimum <= 0 or maximum < minimum:
            raise ValueError(
                f"scale_range должен быть (min, max) с 0 < min <= max, получено {scale_range}"
            )

        self.scale_range = (minimum, maximum)

    def __call__(
        self,
        image: Image.Image | Tensor,
        mask: Tensor,
    ) -> tuple[Image.Image | Tensor, Tensor]:
        scale = random.uniform(*self.scale_range)

        height, width = image_size(image)
        new_height = max(1, round(height * scale))
        new_width = max(1, round(width * scale))

        image = F.resize(
            image,
            size=[new_height, new_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = resize_mask(mask, (new_height, new_width))

        return image, mask


class SegmentationRandomCrop:
    """Случайный кроп до crop_size с предварительным паддингом.

    После RandomScale картинка может оказаться меньше crop_size, поэтому сначала
    добивается паддингом: изображение — image_fill (чёрный), маска — ignore_index.
    Заполнять маску нулём нельзя: 0 — это валидный класс road, и модель училась
    бы считать дорогой пустые поля.
    """

    def __init__(
        self,
        crop_size: tuple[int, int],
        ignore_index: int = 255,
        image_fill: int = 0,
    ) -> None:
        self.crop_height = int(crop_size[0])
        self.crop_width = int(crop_size[1])
        self.ignore_index = ignore_index
        self.image_fill = image_fill

    def __call__(
        self,
        image: Image.Image | Tensor,
        mask: Tensor,
    ) -> tuple[Image.Image | Tensor, Tensor]:
        height, width = image_size(image)

        pad_bottom = max(0, self.crop_height - height)
        pad_right = max(0, self.crop_width - width)

        if pad_bottom > 0 or pad_right > 0:
            # padding в torchvision: [left, top, right, bottom]
            padding = [0, 0, pad_right, pad_bottom]
            image = F.pad(image, padding, fill=self.image_fill)
            mask = F.pad(mask.unsqueeze(0), padding, fill=self.ignore_index).squeeze(0)
            height, width = image_size(image)

        top = random.randint(0, height - self.crop_height)
        left = random.randint(0, width - self.crop_width)

        image = F.crop(image, top, left, self.crop_height, self.crop_width)
        mask = F.crop(mask.unsqueeze(0), top, left, self.crop_height, self.crop_width).squeeze(0)

        return image, mask


class SegmentationRandomHorizontalFlip:
    """Отражение по горизонтали — картинка и маска обязаны отразиться вместе."""

    def __init__(
        self,
        p: float = 0.5,
    ) -> None:
        self.p = p

    def __call__(
        self,
        image: Image.Image | Tensor,
        mask: Tensor,
    ) -> tuple[Image.Image | Tensor, Tensor]:
        if random.random() < self.p:
            image = F.hflip(image)
            mask = F.hflip(mask.unsqueeze(0)).squeeze(0)

        return image, mask


class SegmentationColorJitter:
    """Фотометрическая аугментация. Маску не трогает по построению:
    геометрия не меняется, меняются только цвета пикселей.
    """

    def __init__(
        self,
        brightness: float = 0.5,
        contrast: float = 0.5,
        saturation: float = 0.5,
        hue: float = 0.0,
    ) -> None:
        self.jitter = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(
        self,
        image: Image.Image | Tensor,
        mask: Tensor,
    ) -> tuple[Image.Image | Tensor, Tensor]:
        return self.jitter(image), mask


class SegmentationResize:
    """Детерминированный ресайз к фиксированному (height, width) — для eval."""

    def __init__(
        self,
        size: tuple[int, int],
    ) -> None:
        self.size = (int(size[0]), int(size[1]))

    def __call__(
        self,
        image: Image.Image | Tensor,
        mask: Tensor,
    ) -> tuple[Image.Image | Tensor, Tensor]:
        if image_size(image) == self.size:
            return image, mask

        image = F.resize(
            image,
            size=[self.size[0], self.size[1]],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = resize_mask(mask, self.size)

        return image, mask


class SegmentationToTensor:
    """Изображение -> float [0, 1], маска -> int64.

    Маска НЕ делится на 255 и не превращается в float: это индексы классов,
    которых ждёт cross_entropy. Здесь же единственный переход uint8 -> int64.
    """

    def __call__(
        self,
        image: Image.Image | Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(image, Tensor):
            image = F.pil_to_tensor(image)

        if not image.is_floating_point():
            image = image.float() / 255.0
        elif image.max() > 1:
            image = image / 255.0

        return image, mask.to(torch.int64)


class SegmentationNormalize:
    """Нормализация изображения; маска проходит насквозь."""

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        self.mean = tuple(mean)
        self.std = tuple(std)

    def __call__(
        self,
        image: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return F.normalize(image, mean=self.mean, std=self.std), mask
