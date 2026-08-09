"""Сборка torchvision-трансформов из конфига."""

from collections.abc import Sequence
from torchvision import transforms
from src.utils import detection_transforms, segmentation_transforms


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
        value="random" # type: ignore
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


def build_segmentation_transform_train(
    mean: Sequence[float],
    std: Sequence[float],
    crop_size: Sequence[int],
    scale_range: Sequence[float] = (0.5, 2.0),
    ignore_index: int = 255,
    color_jitter: float = 0.5,
    hflip_p: float = 0.5,
) -> segmentation_transforms.SegmentationCompose:
    """Стандартный train-рецепт семантической сегментации Cityscapes.

    scale -> crop -> flip -> jitter -> tensor -> normalize.

    Порядок не произвольный: геометрия применяется до фотометрии (иначе
    jitter считался бы по уже обрезанной статистике), а ToTensor/Normalize
    идут последними, потому что до них дешевле работать с uint8/PIL.

    Обучение идёт на кропах, а не на полном кадре 1024x2048: полный кадр
    не влезает в память батчем осмысленного размера, а случайный масштаб
    с кропом заодно даёт модели объекты разных размеров.
    """
    ops: list = [
        segmentation_transforms.SegmentationRandomScale(
            scale_range=(float(scale_range[0]), float(scale_range[1])),
        ),
        segmentation_transforms.SegmentationRandomCrop(
            crop_size=(int(crop_size[0]), int(crop_size[1])),
            ignore_index=ignore_index,
        ),
        segmentation_transforms.SegmentationRandomHorizontalFlip(p=hflip_p),
    ]

    if color_jitter > 0:
        ops.append(
            segmentation_transforms.SegmentationColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
            )
        )

    ops.append(segmentation_transforms.SegmentationToTensor())
    ops.append(segmentation_transforms.SegmentationNormalize(mean, std))

    return segmentation_transforms.SegmentationCompose(ops)


def build_segmentation_transform_eval(
    mean: Sequence[float],
    std: Sequence[float],
    image_size: Sequence[int] | None = None,
) -> segmentation_transforms.SegmentationCompose:
    """Eval без аугментаций. image_size=null — считать mIoU в родном
    разрешении 1024x2048 (так метрика сравнима с публичными числами).
    """
    ops: list = []

    if image_size is not None:
        ops.append(
            segmentation_transforms.SegmentationResize(
                size=(int(image_size[0]), int(image_size[1])),
            )
        )

    ops.append(segmentation_transforms.SegmentationToTensor())
    ops.append(segmentation_transforms.SegmentationNormalize(mean, std))

    return segmentation_transforms.SegmentationCompose(ops)