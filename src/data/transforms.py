"""Сборка torchvision-трансформов из конфига."""

from collections.abc import Sequence

from torchvision import transforms

from src.utils import detection_transforms


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
        ops.append(transforms.RandomResizedCrop(
            size=image_size,
            scale=(0.7, 1.0),
            ratio=(0.75, 4 / 3),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ))
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
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
        ops.append(transforms.Resize(
            size=256,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        ))
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
        ops.append(transforms.RandomResizedCrop(
            size=image_size, 
            scale=(0.6, 1.0), 
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        ))
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.RandAugment(
        num_ops=2, 
        magnitude=9,
        interpolation=transforms.InterpolationMode.BICUBIC
    ))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(tuple(mean), tuple(std)))
    ops.append(transforms.RandomErasing(
        p=0.25,
        scale=(0.02, 0.20),
        ratio=(0.3, 3.3), 
        value="random"
    ))
    return transforms.Compose(ops)


def build_transform_tinyvit_eval(
    mean: Sequence[float],
    std: Sequence[float],
    image_size: int | None = None,
) -> transforms.Compose:
    ops: list = []
    if image_size is not None:
        ops.append(transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BICUBIC))
    ops.append(transforms.CenterCrop(image_size))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(tuple(mean), tuple(std)))
    return transforms.Compose(ops)

def build_base_transform_for_cityscapes(
    mean: Sequence[float],
    std: Sequence[float],
    image_size: tuple[int, int] | None = None,
) -> transforms.Compose:
    ops: list = []
    if image_size is not None:
        ops.append(detection_transforms.DetectionResize(image_size))
    ops.append(detection_transforms.DetectionToTensor())
    ops.append(detection_transforms.DetectionNormalize(mean, std))
    return detection_transforms.DetectionCompose(ops)

# def build_eval_transform_for_cityscapes(
#     mean: Sequence[float],
#     std: Sequence[float],
#     image_size: tuple[int, int] | None = None,
# ) -> transforms.Compose: