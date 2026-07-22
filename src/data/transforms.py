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
        # ops.append(transforms.Resize(image_size))
        ops.append(transforms.Resize((image_size, image_size)))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(tuple(mean), tuple(std)))
    return transforms.Compose(ops)

def build_transform_train(
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
        ops.append(transforms.RandomResizedCrop(image_size))
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(tuple(mean), tuple(std)))
    return transforms.Compose(ops)

def build_transform_eval(
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
        ops.append(transforms.Resize(256))
        ops.append(transforms.CenterCrop(image_size))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(tuple(mean), tuple(std)))
    return transforms.Compose(ops)

def build_transform_tinyvit_train(
    mean: Sequence[float],
    std: Sequence[float],
    image_size: int | None = None,
) -> transforms.Compose:
    ops: list = []
    if image_size is not None:
        ops.append(transforms.RandomResizedCrop(size=image_size, scale=(0.7, 1.0), interpolation=transforms.InterpolationMode.BICUBIC))
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(tuple(mean), tuple(std)))
    ops.append(transforms.RandomErasing(p=0.25, value="random"))
    return transforms.Compose(ops)

def build_transform_tinyvit_eval(
    mean: Sequence[float],
    std: Sequence[float],
    image_size: int | None = None,
) -> transforms.Compose:
    ops: list = []
    if image_size is not None:
        ops.append(transforms.Resize(size=image_size, interpolation=transforms.InterpolationMode.BICUBIC))
    ops.append(transforms.CenterCrop(image_size))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(tuple(mean), tuple(std)))
    return transforms.Compose(ops)