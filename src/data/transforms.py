"""Сборка torchvision-трансформов из конфига."""

from collections.abc import Sequence

from torchvision import transforms


def base_transform(
    mean: Sequence[float],
    std: Sequence[float],
    image_size: int | None = None,
) -> transforms.Compose:
    
    ops: list = []
    if image_size is not None:
        ops.append(transforms.Resize(image_size))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(tuple(mean), tuple(std)))
    return transforms.Compose(ops)
