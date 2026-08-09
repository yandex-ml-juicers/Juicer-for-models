"""Растровая визуализация предсказаний сегментации.

Собирает из тензоров одну картинку-полосу

    [ кадр | разметка | предсказание | карта ошибок ]

которую тренер отправляет в Debug Samples трекера. Числовые метрики говорят,
насколько модель ошибается, но не говорят ГДЕ: пропали тонкие столбы, слиплись
машины, поехала граница тротуара — это видно только глазами.

Логика раскраски повторяет notebooks/segmentation_predictions_visualization.ipynb,
но без matplotlib: тут нужен готовый uint8-массив, а не фигура.
"""

import colorsys
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

IGNORE_COLOR = (0, 0, 0)
ERROR_COLOR = (230, 57, 53)
SEPARATOR_COLOR = (255, 255, 255)


def default_palette(num_classes: int) -> tuple[tuple[int, int, int], ...]:
    """Различимые цвета для датасета без своей палитры — равномерно по кругу оттенков."""
    return tuple(
        tuple(round(255 * channel) for channel in colorsys.hsv_to_rgb(i / max(num_classes, 1), 0.65, 0.95))
        for i in range(num_classes)
    )


def build_color_lut(palette: Sequence[Sequence[int]]) -> np.ndarray:
    """Таблица [256, 3]: индекс класса -> цвет. Всё вне палитры (в т.ч. ignore) — чёрное."""
    lut = np.full((256, 3), IGNORE_COLOR, dtype=np.uint8)
    for train_id, color in enumerate(palette):
        if train_id < 256:
            lut[train_id] = color
    return lut


def denormalize(image: torch.Tensor, mean: Sequence[float] | None, std: Sequence[float] | None) -> np.ndarray:
    """Тензор [3, H, W] после Normalize -> uint8 HWC.

    Без mean/std нормировка неизвестна, и картинку приходится тянуть по
    min-max: цвета будут приблизительными, но структура кадра различима.
    """
    image = image.detach().float().cpu()

    if mean is not None and std is not None:
        mean_tensor = torch.tensor(list(mean)).view(-1, 1, 1)
        std_tensor = torch.tensor(list(std)).view(-1, 1, 1)
        image = image * std_tensor + mean_tensor
    else:
        low, high = image.amin(), image.amax()
        image = (image - low) / (high - low).clamp_min(1e-6)

    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return (image * 255).round().astype(np.uint8)


def _blend(image: np.ndarray, colors: np.ndarray, alpha: float, painted: np.ndarray) -> np.ndarray:
    """Полупрозрачная маска поверх кадра; painted=False оставляет пиксель как есть."""
    mixed = (1 - alpha) * image.astype(np.float32) + alpha * colors.astype(np.float32)
    mixed = mixed.round().clip(0, 255).astype(np.uint8)
    return np.where(painted[..., None], mixed, image)


def _error_map(image: np.ndarray, prediction: np.ndarray, mask: np.ndarray, ignore_index: int) -> np.ndarray:
    """Красным — где модель ошиблась. Void не ошибка: там нет разметки."""
    labeled = mask != ignore_index
    wrong = (prediction != mask) & labeled

    gray = image.mean(axis=2, keepdims=True).repeat(3, axis=2)
    red = np.array(ERROR_COLOR, dtype=np.float32)
    painted = (0.8 * red + 0.2 * gray).round().clip(0, 255).astype(np.uint8)

    return np.where(wrong[..., None], painted, gray.round().astype(np.uint8))


def _resize_mask(values: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Ближайший сосед: любая другая интерполяция придумала бы несуществующие классы."""
    return F.interpolate(values[None, None].float(), size=size, mode="nearest")[0, 0].long()


def _resize(
    image: torch.Tensor,
    mask: torch.Tensor,
    prediction: torch.Tensor,
    max_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Уменьшение до max_width по ширине: кадр билинейно, маски — ближайшим соседом.

    Уменьшать надо ДО раскраски: интерполяция цветов размывала бы границы
    классов и рисовала бы несуществующие оттенки между ними.
    """
    height, width = image.shape[-2:]
    if max_width <= 0 or width <= max_width:
        return image, mask, prediction

    size = (max(1, round(height * max_width / width)), max_width)
    image = F.interpolate(image[None].float(), size=size, mode="bilinear", align_corners=False)[0]
    return image, _resize_mask(mask, size), _resize_mask(prediction, size)


def prediction_panel(
    image: torch.Tensor,
    mask: torch.Tensor,
    prediction: torch.Tensor,
    *,
    palette: Sequence[Sequence[int]],
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    ignore_index: int = 255,
    alpha: float = 0.55,
    max_width: int = 512,
    gap: int = 4,
) -> np.ndarray:
    """[кадр | разметка | предсказание | ошибки] одной картинкой uint8 [H, W, 3].

    image — [3, H, W] после Normalize, mask и prediction — [H, W] с индексами
    классов. Пиксели ignore_index маской не закрашиваются: под ними видно
    исходный кадр, и сразу понятно, что там разметки нет.
    """
    image, mask, prediction = _resize(image, mask.detach(), prediction.detach(), max_width)

    frame = denormalize(image, mean, std)
    mask_np = mask.cpu().numpy()
    prediction_np = prediction.cpu().numpy()

    lut = build_color_lut(palette)
    labeled = mask_np != ignore_index

    panels = [
        frame,
        _blend(frame, lut[np.clip(mask_np, 0, 255)], alpha, labeled),
        _blend(frame, lut[np.clip(prediction_np, 0, 255)], alpha, np.ones_like(labeled)),
        _error_map(frame, prediction_np, mask_np, ignore_index),
    ]

    if gap > 0:
        separator = np.full((frame.shape[0], gap, 3), SEPARATOR_COLOR, dtype=np.uint8)
        spaced = [panels[0]]
        for panel in panels[1:]:
            spaced += [separator, panel]
        panels = spaced

    return np.concatenate(panels, axis=1)
