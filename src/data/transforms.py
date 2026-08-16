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
    hflip_p: float = 0.5,
    rand_augment_num_ops: int = 0,
    rand_augment_magnitude: int = 9,
    random_erasing_p: float = 0.0,
    random_erasing_scale: Sequence[float] = (0.02, 0.2),
    random_erasing_ratio: Sequence[float] = (0.3, 3.3),
    random_erasing_value: float | str = "random",
) -> transforms.Compose:
    """RandomResizedCrop + flip (+ опционально RandAugment и RandomErasing).

    Дефолты повторяют прежнее поведение один в один: без RandAugment и без
    стирания. Всё новое включается только явно из конфига, чтобы уже
    посчитанные бейзлайны остались сравнимыми.

    Args:
        rand_augment_num_ops: сколько операций RandAugment применять к кадру;
            0 — не применять. Стандартный ImageNet-рецепт — 2 при magnitude 9.
        random_erasing_p: вероятность стереть прямоугольник. Стирание идёт
            ПОСЛЕ Normalize, поэтому "random" — это шум из N(0, 1) уже
            в нормализованной шкале (так же устроен torchvision).
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
    if hflip_p > 0:
        ops.append(transforms.RandomHorizontalFlip(p=hflip_p))
    if rand_augment_num_ops > 0:
        # До ToTensor: RandAugment работает с PIL/uint8.
        ops.append(transforms.RandAugment(
            num_ops=rand_augment_num_ops,
            magnitude=rand_augment_magnitude,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(tuple(mean), tuple(std)))
    if random_erasing_p > 0:
        ops.append(transforms.RandomErasing(
            p=random_erasing_p,
            scale=tuple(random_erasing_scale),
            ratio=tuple(random_erasing_ratio),
            value=random_erasing_value,
        ))
    return transforms.Compose(ops)


def build_transform_eval(
    mean: Sequence[float],
    std: Sequence[float],
    image_size: int | None = None,
    resize_size: int | None = None,
) -> transforms.Compose:
    """Resize по короткой стороне + CenterCrop, затем ToTensor + Normalize.

    resize_size=None означает канонические 87.5% ImageNet-рецепта:
    resize до image_size / 0.875 и центральный кроп до image_size
    (224 -> 256, как было захардкожено раньше). Явное значение нужно, только
    если воспроизводится чужой рецепт с другим соотношением.
    """
    ops: list = []
    if image_size is not None:
        if resize_size is None:
            resize_size = int(round(image_size / 0.875))
        ops.append(transforms.Resize(
            size=resize_size,
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
        # 256/224 — то же соотношение 0.875, что и в build_transform_eval.
        ops.append(transforms.Resize(
            size=int(round(image_size / 0.875)),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ))
        # CenterCrop обязан быть внутри этой ветки: с image_size=None
        # он падает с TypeError вместо того, чтобы просто ничего не делать.
        ops.append(transforms.CenterCrop(image_size))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(tuple(mean), tuple(std)))
    return transforms.Compose(ops)

def build_base_transform_for_cityscapes(
    mean: Sequence[float],
    std: Sequence[float],
    train: bool = True,
    image_size: tuple[int, int] | None = None,
    horizontal_flip: float = 0.0,
) -> detection_transforms.DetectionCompose:
    ops: list = []
    if train:
        # Флип идёт первым: он не меняет геометрию кадра, а кроп ниже
        # рассчитывает свои параметры уже по итоговому изображению.
        if horizontal_flip > 0.0:
            ops.append(detection_transforms.DetectionRandomHorizontalFlip(p=horizontal_flip))

        ops.append(
                detection_transforms.DetectionRandomResizedCrop(
                    size=image_size,
                    scale=(0.6, 1.0),
                    ratio=(1.7, 2.3),
                )
            )
        ops.append(detection_transforms.DetectionColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ))
        ops.append(
            detection_transforms.DetectionGaussianBlur(
                kernel_size=5,
                sigma=(0.1, 2.0),
                p=0.2,
            )
        )
    else:
        if image_size is not None:
            ops.append(detection_transforms.DetectionResize(image_size))

    ops.append(detection_transforms.DetectionToTensor())
    ops.append(detection_transforms.DetectionNormalize(mean, std))
    return detection_transforms.DetectionCompose(ops)

def build_transforms_for_yolo(
    mean: Sequence[float] | None,
    std: Sequence[float] | None,
    train: bool = True,
    image_size: tuple[int, int] | None = None,
) -> detection_transforms.DetectionCompose:
    ops: list = []

    if train:
        # Флип первым: он не меняет геометрию кадра, а кроп ниже считает свои
        # параметры уже по итоговому изображению.
        ops.append(detection_transforms.DetectionRandomHorizontalFlip(p=0.5))

        # Единственная геометрия: кроп задаёт и сдвиг, и масштаб сразу.
        # ratio держится около аспекта Cityscapes (2048/1024 = 2.0). При
        # прежних (0.8, 1.25) кроп был почти квадратным и растягивался до
        # 512x1024, тогда как eval делал честный resize, — train и eval
        # видели кадры разной геометрии.
        ops.append(
            detection_transforms.DetectionRandomResizedCrop(
                size=image_size,
                scale=(0.5, 1.0),
                ratio=(1.8, 2.2),
            )
        )

        # Одна фотометрия вместо трёх, параметры из рецепта ultralytics.
        ops.append(
            detection_transforms.DetectionRandomHSV(
                hgain=0.015,
                sgain=0.7,
                vgain=0.4,
                p=1.0,
            )
        )
        ops.append(
            detection_transforms.DetectionGaussianBlur(
                kernel_size=5,
                sigma=(0.1, 2.0),
                p=0.1,
            )
        )

        # После кропа у объектов на границе кадра остаются вырожденные рамки.
        ops.append(detection_transforms.DetectionFilterBoxes(min_size=2.0))

    else:
        if image_size is not None:
            ops.append(detection_transforms.DetectionResize(image_size))

    ops.append(detection_transforms.DetectionToTensor())

    # mean/std = null — вход остаётся в 0..1, как обучает ultralytics и как
    # приходят COCO-веса. ImageNet-нормализация сдвигала бы распределение
    # входа относительно предобученной части.
    if mean is not None and std is not None:
        ops.append(detection_transforms.DetectionNormalize(mean, std))

    return detection_transforms.DetectionCompose(ops)
# def build_eval_transform_for_cityscapes(
#     mean: Sequence[float],
#     std: Sequence[float],
#     image_size: tuple[int, int] | None = None,
# ) -> transforms.Compose:

def build_train_transform_for_cityscapes(
    mean: Sequence[float],
    std: Sequence[float],
    image_size: tuple[int, int] | None = None,
    flip_prob: float = 0.5,
) -> detection_transforms.DetectionCompose:
    ops: list = []
    
    # Сначала ресайз (чтобы все картинки были одного размера)
    if image_size is not None:
        ops.append(detection_transforms.DetectionResize(image_size))
        
    # Затем случайное отражение
    ops.append(detection_transforms.DetectionHorizontalFlip(p=flip_prob))
    
    # В конце перевод в тензор и нормализация
    ops.append(detection_transforms.DetectionToTensor())
    ops.append(detection_transforms.DetectionNormalize(mean, std))
    
    return detection_transforms.DetectionCompose(ops)

def build_segmentation_transform_train(
    mean: Sequence[float],
    std: Sequence[float],
    crop_size: Sequence[int],
    scale_range: Sequence[float] = (0.5, 2.0),
    ignore_index: int = 255,
    color_jitter: float = 0.5,
    hflip_p: float = 0.5,
    hue: float = 0.0,
    color_jitter_p: float = 1.0,
    cat_max_ratio: float | None = None,
    blur_p: float = 0.0,
    blur_kernel_size: int = 5,
    blur_sigma: Sequence[float] = (0.1, 2.0),
    random_erasing_p: float = 0.0,
    random_erasing_scale: Sequence[float] = (0.02, 0.2),
    random_erasing_ratio: Sequence[float] = (0.3, 3.3),
    random_erasing_value: float | str = 0.0,
    random_erasing_erases_labels: bool = False,
) -> segmentation_transforms.SegmentationCompose:
    """Стандартный train-рецепт семантической сегментации Cityscapes.

    scale -> crop -> flip -> jitter -> blur -> tensor -> normalize -> erasing.

    Порядок не произвольный: геометрия применяется до фотометрии (иначе
    jitter считался бы по уже обрезанной статистике), ToTensor/Normalize
    идут ближе к концу, потому что до них дешевле работать с uint8/PIL,
    а стирание — самым последним, потому что оно задаётся в нормализованной
    шкале (см. SegmentationRandomErasing).

    Обучение идёт на кропах, а не на полном кадре 1024x2048: полный кадр
    не влезает в память батчем осмысленного размера, а случайный масштаб
    с кропом заодно даёт модели объекты разных размеров.

    Всё, что добавлено сверх базового рецепта (hue, cat_max_ratio, blur,
    erasing), по умолчанию ВЫКЛЮЧЕНО: базовый конфиг обязан оставаться тем же,
    иначе уже посчитанные бейзлайны команды перестанут быть сравнимыми.
    Усиленный набор лежит отдельным конфигом — configs/data/transform/train/
    cityscapes_seg_strong_transform.yaml.

    Args:
        color_jitter: сила brightness/contrast/saturation; 0 — выключить.
        color_jitter_p: вероятность применить джиттер к кадру.
        hue: сила сдвига оттенка (0.05 уже заметно; 0 — выключен).
        cat_max_ratio: доля, выше которой доминирование одного класса делает
            кроп непригодным; None — брать первый попавшийся кроп.
        blur_p: вероятность гауссова размытия.
        random_erasing_p: вероятность стереть прямоугольник.
        random_erasing_erases_labels: помечать ли стёртое как ignore_index.
    """
    ops: list = [
        segmentation_transforms.SegmentationRandomScale(
            scale_range=(float(scale_range[0]), float(scale_range[1])),
        ),
        segmentation_transforms.SegmentationRandomCrop(
            crop_size=(int(crop_size[0]), int(crop_size[1])),
            ignore_index=ignore_index,
            cat_max_ratio=cat_max_ratio,
        ),
        segmentation_transforms.SegmentationRandomHorizontalFlip(p=hflip_p),
    ]

    if color_jitter > 0 or hue > 0:
        ops.append(
            segmentation_transforms.SegmentationColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=hue,
                p=color_jitter_p,
            )
        )

    if blur_p > 0:
        ops.append(
            segmentation_transforms.SegmentationGaussianBlur(
                p=blur_p,
                kernel_size=blur_kernel_size,
                sigma=blur_sigma,
            )
        )

    ops.append(segmentation_transforms.SegmentationToTensor())
    ops.append(segmentation_transforms.SegmentationNormalize(mean, std))

    if random_erasing_p > 0:
        ops.append(
            segmentation_transforms.SegmentationRandomErasing(
                p=random_erasing_p,
                scale=random_erasing_scale,
                ratio=random_erasing_ratio,
                value=random_erasing_value,
                erase_labels=random_erasing_erases_labels,
                ignore_index=ignore_index,
            )
        )

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
