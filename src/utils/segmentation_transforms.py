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

import math
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

    cat_max_ratio — приём из mmsegmentation: кроп отбрасывается и берётся заново,
    если один класс занимает в нём больше указанной доли размеченных пикселей.
    На Cityscapes это спасает от батчей, наполовину состоящих из кропов «одна
    дорога» или «одно небо»: такие кропы почти не несут сигнала, но исправно
    съедают шаг оптимизатора. Попыток даётся max_attempts; если ни одна не
    прошла, берётся последний кроп — лучше слегка однородный кроп, чем
    зависший даталоадер.
    """

    def __init__(
        self,
        crop_size: tuple[int, int],
        ignore_index: int = 255,
        image_fill: int = 0,
        cat_max_ratio: float | None = None,
        max_attempts: int = 10,
    ) -> None:
        if cat_max_ratio is not None and not 0.0 < cat_max_ratio <= 1.0:
            raise ValueError(
                f"cat_max_ratio должен быть в (0, 1], получено {cat_max_ratio}"
            )

        self.crop_height = int(crop_size[0])
        self.crop_width = int(crop_size[1])
        self.ignore_index = ignore_index
        self.image_fill = image_fill
        self.cat_max_ratio = cat_max_ratio
        self.max_attempts = int(max_attempts)

    def _is_diverse_enough(self, mask_crop: Tensor) -> bool:
        """True, если ни один класс не доминирует в кропе сверх cat_max_ratio."""
        labels, counts = torch.unique(mask_crop, return_counts=True)
        counts = counts[labels != self.ignore_index]

        if counts.numel() <= 1:
            # Один класс на весь кроп (или сплошной ignore) — заведомо не годится.
            return False

        return bool((counts.max().float() / counts.sum().float()) < self.cat_max_ratio)

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

        attempts = self.max_attempts if self.cat_max_ratio is not None else 1
        for _ in range(attempts):
            top = random.randint(0, height - self.crop_height)
            left = random.randint(0, width - self.crop_width)
            mask_crop = mask[
                top : top + self.crop_height,
                left : left + self.crop_width,
            ]
            if self.cat_max_ratio is None or self._is_diverse_enough(mask_crop):
                break

        image = F.crop(image, top, left, self.crop_height, self.crop_width)

        return image, mask_crop


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

    p — вероятность применить джиттер к кадру. p < 1 оставляет часть батча
    в исходных цветах: полезно, потому что цветовая статистика Cityscapes
    (серый асфальт, зелень, небо) сама по себе информативна, и стирать её
    у каждого кадра не нужно.

    hue заведомо отделён от остальных трёх: сдвиг оттенка — самая агрессивная
    из фотометрий (красная машина становится синей), поэтому его дефолт 0,
    а разумный рабочий диапазон — до 0.05, а не до brightness/contrast.
    """

    def __init__(
        self,
        brightness: float = 0.5,
        contrast: float = 0.5,
        saturation: float = 0.5,
        hue: float = 0.0,
        p: float = 1.0,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p должна быть в [0, 1], получено {p}")

        self.p = float(p)
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
        if random.random() >= self.p:
            return image, mask

        return self.jitter(image), mask


class SegmentationGaussianBlur:
    """Гауссово размытие с вероятностью p. Маску не трогает.

    Смысл тот же, что у джиттера, только по другой оси: кадры Cityscapes сняты
    одной камерой и одинаково резкие, поэтому модель охотно цепляется за
    высокочастотные текстуры. Размытие части батча заставляет её опираться
    и на форму тоже.
    """

    def __init__(
        self,
        p: float = 0.5,
        kernel_size: int = 5,
        sigma: Sequence[float] = (0.1, 2.0),
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p должна быть в [0, 1], получено {p}")
        if kernel_size % 2 == 0 or kernel_size <= 0:
            raise ValueError(
                f"kernel_size должен быть положительным нечётным, получено {kernel_size}"
            )

        self.p = float(p)
        self.kernel_size = int(kernel_size)
        self.sigma = (float(sigma[0]), float(sigma[1]))

    def __call__(
        self,
        image: Image.Image | Tensor,
        mask: Tensor,
    ) -> tuple[Image.Image | Tensor, Tensor]:
        if random.random() >= self.p:
            return image, mask

        sigma = random.uniform(*self.sigma)
        return F.gaussian_blur(image, [self.kernel_size, self.kernel_size], [sigma, sigma]), mask


class SegmentationRandomErasing:
    """Random Erasing (Zhong et al., AAAI 2020, arXiv:1708.04896) для пары
    (изображение, маска).

    Прямоугольник случайной площади и пропорций заполняется константой или
    шумом. Идея — не дать модели опереться на одну характерную деталь: если
    её видно не всегда, приходится использовать контекст.

    МАСКА ПО УМОЛЧАНИЮ НЕ МЕНЯЕТСЯ, и это осознанно: класс под закрашенным
    пятном по-прежнему нужно предсказать — в этом вся аугментация. Если это
    кажется слишком жёстким (стёрлась целиком машина, а модель штрафуется за
    то, что её не угадала), erase_labels=True помечает стёртую область как
    ignore_index — тогда она просто выпадает и из лосса, и из метрики.

    Место в пайплайне — самый конец, ПОСЛЕ Normalize: значения заполнения
    заданы в нормализованной шкале, ровно как у torchvision.RandomErasing.
    Поэтому value=0.0 — это средний цвет датасета, а не чёрный.
    """

    def __init__(
        self,
        p: float = 0.25,
        scale: Sequence[float] = (0.02, 0.2),
        ratio: Sequence[float] = (0.3, 3.3),
        value: float | str = 0.0,
        erase_labels: bool = False,
        ignore_index: int = 255,
        max_attempts: int = 10,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p должна быть в [0, 1], получено {p}")
        if not 0 < scale[0] <= scale[1] <= 1:
            raise ValueError(f"scale должен быть (min, max) в (0, 1], получено {tuple(scale)}")
        if not 0 < ratio[0] <= ratio[1]:
            raise ValueError(f"ratio должен быть (min, max) с 0 < min <= max, получено {tuple(ratio)}")
        if isinstance(value, str) and value != "random":
            raise ValueError(f"value должно быть числом или 'random', получено {value!r}")

        self.p = float(p)
        self.scale = (float(scale[0]), float(scale[1]))
        self.ratio = (float(ratio[0]), float(ratio[1]))
        self.value = value
        self.erase_labels = bool(erase_labels)
        self.ignore_index = int(ignore_index)
        self.max_attempts = int(max_attempts)

    def _sample_box(self, height: int, width: int) -> tuple[int, int, int, int] | None:
        """Прямоугольник (top, left, box_height, box_width) или None.

        Алгоритм — из статьи: площадь и пропорции берутся независимо, и если
        полученный прямоугольник не влезает в кадр, попытка повторяется.
        None означает, что за max_attempts попыток не влез ни один — стирание
        просто пропускается (так же поступает torchvision).
        """
        area = height * width

        for _ in range(self.max_attempts):
            erase_area = area * random.uniform(*self.scale)
            aspect = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))

            box_height = int(round(math.sqrt(erase_area * aspect)))
            box_width = int(round(math.sqrt(erase_area / aspect)))

            if box_height >= height or box_width >= width or box_height == 0 or box_width == 0:
                continue

            top = random.randint(0, height - box_height)
            left = random.randint(0, width - box_width)
            return top, left, box_height, box_width

        return None

    def __call__(
        self,
        image: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(image, Tensor):
            raise TypeError(
                "SegmentationRandomErasing работает с тензором и должен стоять после "
                "SegmentationToTensor/SegmentationNormalize, а не до них"
            )
        if random.random() >= self.p:
            return image, mask

        height, width = image_size(image)
        box = self._sample_box(height, width)
        if box is None:
            return image, mask

        top, left, box_height, box_width = box
        region = (..., slice(top, top + box_height), slice(left, left + box_width))

        image = image.clone()
        if self.value == "random":
            image[region] = torch.randn(
                image.shape[0], box_height, box_width, dtype=image.dtype
            )
        else:
            image[region] = float(self.value)

        if self.erase_labels:
            mask = mask.clone()
            mask[top : top + box_height, left : left + box_width] = self.ignore_index

        return image, mask


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
