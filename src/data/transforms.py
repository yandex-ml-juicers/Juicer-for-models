"""Сборка torchvision-трансформов из конфига."""

from collections.abc import Sequence

from torchvision import transforms


def build_transforms(
    mean: Sequence[float],
    std: Sequence[float],
    image_size: int | None = None,
) -> transforms.Compose:
    """ToTensor + Normalize, опционально с Resize (для ImageNet-учителей, 224).

    Аугментаций нет намеренно: бейзлайны в ноутбуках обучались без них,
    а воспроизводить нужно ровно их. Точка расширения — сюда.
    """
    ops: list = []
    if image_size is not None:
        ops.append(transforms.Resize(image_size))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(tuple(mean), tuple(std)))
    return transforms.Compose(ops)
