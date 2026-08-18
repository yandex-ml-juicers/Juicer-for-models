"""Фабрики моделей. Каждая функция — точка входа для _target_ в конфигах
configs/model/teacher/*.yaml и configs/model/student/*.yaml.

Здесь только сборка: сами архитектуры лежат в отдельных модулях
(src/models/segformer.py, src/models/unet.py), а работа с весами —
в src/utils/checkpoints.py, потому что она общая для всех задач.

Заморозка учителя здесь НЕ делается: это политика обучения, а не свойство
модели, и ею владеет Trainer.
"""

import timm
from pathlib import Path
import warnings

import torch
import torchvision
from torch import nn
from torchvision import models as tv_models
from torchvision.models._api import WeightsEnum
from torchvision.models import get_model
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet18, ResNet18_Weights
from transformers import LwDetrConfig, LwDetrForObjectDetection, RTDetrConfig, RTDetrForObjectDetection

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.head import Detect

from src.models.lwdetr_clockdistill import LwDetrCLoCKDistillMemory
from src.models.lwdetr_kd_detr import LwDetrKDDETRProbes
from src.models.segformer import SegFormer
from src.models.stochastic_depth import apply_stochastic_depth
from src.models.timm_unet import TIMM_UNET_VARIANTS, TimmUNet
from src.models.unet import UNET_VARIANTS, UNet
from src.utils.checkpoints import load_checkpoint_into, resolve_weights_dir

from hydra.utils import to_absolute_path


def from_torch_hub(repo: str, name: str, pretrained: bool = False) -> nn.Module:
    """Модель из torch.hub (например, chenyaofo/pytorch-cifar-models).

    trust_repo=True отключает интерактивный вопрос про доверие к репозиторию —
    скрипт обязан работать без stdin (запуски в tmux/CI).
    """
    return torch.hub.load(repo, name, pretrained=pretrained, trust_repo=True) # type: ignore


def from_detectors(name: str, pretrained: bool = True) -> nn.Module:
    """Предобученные CIFAR-модели из пакета detectors (например, resnet50_cifar10)."""
    import detectors  # noqa: F401  (импорт регистрирует модели в timm)
    import timm

    return timm.create_model(name, pretrained=pretrained)


def cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """ResNet-18 из torchvision, адаптированный под входы 32x32.

    Стандартный стем ResNet (conv 7x7 stride 2 + maxpool) за два слоя
    уменьшает 32x32 до 8x8 и убивает пространственную информацию.
    Замена: conv 3x3 stride 1 и тождественный maxpool — канонический
    CIFAR-вариант архитектуры.
    """
    model = tv_models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity() # type: ignore
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def torchvision_model_for_classification(
    model_name: str,
    num_classes: int = 1000,
    weights: str | WeightsEnum | None = None,
    checkpoint_path: str | None = None,
    drop_path_rate: float = 0.0,
    drop_path_mode: str = "linear",
) -> nn.Module:
    """
    Function for load models from torchvision
    Editing the last layer for current num of classes

    drop_path_rate > 0 включает stochastic depth в residual-блоках
    (только ResNet-семейство; подробности и ограничения — в
    src/models/stochastic_depth.py). У ShuffleNet residual-сложения нет,
    и включение drop_path_rate для него осознанно падает с ошибкой,
    а не молча ничего не делает.

    Ключи state_dict от этого не меняются, поэтому чекпоинт модели,
    обученной с drop_path_rate > 0, грузится и в модель без него.
    """

    if isinstance(weights, str):
        weights = tv_models.get_weight(weights)

    model = get_model(
        model_name,
        weights=weights,
        num_classes=num_classes
    )

    if drop_path_rate > 0:
        apply_stochastic_depth(model, drop_path_rate, mode=drop_path_mode)

    if checkpoint_path is None:
        return model

    return load_checkpoint_into(model, checkpoint_path, model_name)


def timm_model_for_classification(
    model_name: str,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
    num_classes: int = 100,
    drop_path_rate: float | None = None,
    **kwargs: any,
) -> nn.Module:
    """drop_path_rate — stochastic depth средствами самого timm.

    None означает "не передавать аргумент вовсе": у timm для каждой
    архитектуры свой дефолт, и затирать его нулём без причины не нужно.
    Модели без поддержки drop_path timm отвергает сам, с внятной ошибкой.
    """

    if drop_path_rate is not None:
        kwargs["drop_path_rate"] = drop_path_rate

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    )

    if checkpoint_path is None:
        return model

    return load_checkpoint_into(model, checkpoint_path, model_name)


def unet_for_segmentation(
    num_classes: int = 19,
    in_channels: int = 3,
    variant: str = "full",
    base_channels: int | None = None,
    depth: int | None = None,
    checkpoint_path: str | None = None,
    dropout: float = 0.0,
) -> nn.Module:
    """U-Net для семантической сегментации.

    Args:
        variant: именованный размер из UNET_VARIANTS
            (tiny ~1.9M | small ~4.4M | base ~7.8M | large ~17.4M | full ~31.0M).
        base_channels, depth: точечное переопределение варианта. None —
            взять из спецификации. Оба рычага дорогие: число параметров
            примерно квадратично по base_channels, а +1 к depth добавляет
            самый широкий уровень и умножает размер примерно вчетверо
            (base_channels=32: depth=4 -> 7.8M, depth=5 -> 31.1M). Для
            подбора размера крутите base_channels; depth поднимайте только
            если нужен тап на страйде 32 — и тогда вход обязан быть кратен 32.
        checkpoint_path: чекпоинт нашего тренера (ключ student_state) — так
            обученный на первом этапе U-Net становится учителем.
        dropout: Dropout2d на боттлнеке (регуляризация вместо stochastic
            depth, которую в U-Net применять не к чему — residual-блоков там
            нет). Параметров не добавляет, поэтому старые чекпоинты грузятся
            без изменений.
    """
    if variant not in UNET_VARIANTS:
        raise ValueError(
            f"Неизвестный вариант U-Net: {variant!r}. Доступны: {sorted(UNET_VARIANTS)}"
        )

    spec = UNET_VARIANTS[variant]
    model = UNet(
        num_classes=num_classes,
        in_channels=in_channels,
        base_channels=spec["base_channels"] if base_channels is None else base_channels,
        depth=spec["depth"] if depth is None else depth,
        dropout=dropout,
    )

    if checkpoint_path is None:
        return model

    return load_checkpoint_into(model, checkpoint_path, f"U-Net-{variant}")


def timm_unet_for_segmentation(
    variant: str = "resnet18",
    num_classes: int = 19,
    in_channels: int = 3,
    pretrained: bool = True,
    encoder_name: str | None = None,
    dropout: float = 0.0,
    align_corners: bool = False,
    checkpoint_path: str | None = None,
) -> nn.Module:
    """U-Net с предобученным энкодером из timm.

    Args:
        variant: именованный энкодер из TIMM_UNET_VARIANTS (размеры — в
            комментарии к таблице, от 2.3M до 24.5M).
        pretrained: ГЛАВНЫЙ ПЕРЕКЛЮЧАТЕЛЬ. True — веса энкодера с ImageNet,
            False — та же архитектура со случайной инициализацией. Пара
            прогонов true/false и есть честный ответ на вопрос, сколько
            дало именно предобучение.
        encoder_name: имя модели timm в обход таблицы (тег весов — после
            точки, например "convnext_nano.in12k").
        checkpoint_path: чекпоинт нашего тренера. Если задан, pretrained
            игнорируется: веса всё равно будут перезаписаны, качать их незачем.
    """
    if encoder_name is None:
        if variant not in TIMM_UNET_VARIANTS:
            raise ValueError(
                f"Неизвестный вариант timm-U-Net: {variant!r}. "
                f"Доступны: {sorted(TIMM_UNET_VARIANTS)}. "
                f"Либо задайте encoder_name напрямую."
            )
        encoder_name = TIMM_UNET_VARIANTS[variant]["encoder_name"]

    model = TimmUNet(
        encoder_name=encoder_name,
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained and checkpoint_path is None,
        dropout=dropout,
        align_corners=align_corners,
    )

    if checkpoint_path is None:
        return model

    return load_checkpoint_into(model, checkpoint_path, f"TimmUNet-{variant}")


def segformer_for_segmentation(
    variant: str = "b2",
    num_classes: int = 19,
    pretrained: str | None = "imagenet",
    checkpoint_path: str | None = None,
    weights_dir: str | None = None,
    align_corners: bool = False,
    drop_path_rate: float | None = None,
    classifier_dropout_prob: float | None = None,
) -> nn.Module:
    """SegFormer-B{0..5} для семантической сегментации.

    Args:
        pretrained: None | "imagenet" (энкодер MiT с ImageNet) |
            "cityscapes" (готовая дообученная модель nvidia/segformer-*).
            Игнорируется, если задан checkpoint_path.
        checkpoint_path: чекпоинт нашего тренера — так модель, обученная
            на этапе scratch, подставляется учителем в дистилляцию.
        weights_dir: куда качать веса; по умолчанию data/weights.
        drop_path_rate: stochastic depth в блоках трансформера. None —
            дефолт transformers (0.1). Для B4/B5 и длинных расписаний
            имеет смысл поднимать до 0.2-0.3.
        classifier_dropout_prob: dropout перед головой декодера. None —
            дефолт transformers (0.1).

    Веса Hugging Face кладутся в <weights_dir>/huggingface: hub сам
    проверяет, что уже скачано, поэтому повторный запуск ничего не тянет.
    """
    cache_dir = str(resolve_weights_dir(weights_dir) / "huggingface")

    # Если веса всё равно будут перезаписаны чекпоинтом, качать их незачем.
    model = SegFormer(
        variant=variant,
        num_classes=num_classes,
        pretrained=None if checkpoint_path is not None else pretrained,
        cache_dir=cache_dir,
        align_corners=align_corners,
        drop_path_rate=drop_path_rate,
        classifier_dropout_prob=classifier_dropout_prob,
    )

    if checkpoint_path is None:
        return model

    return load_checkpoint_into(model, checkpoint_path, f"SegFormer-{variant.upper()}")


def lwdetr_small_for_detection(
    num_classes: int = 8,
    disable_custom_kernels: bool = True,
    dropout: float = 0.1,
    attention_dropout: float = 0.1,
    activation_dropout: float = 0.1,
    backbone_dropout: float = 0.0,
    checkpoint_path: str | Path | None = None,
    num_probe_points: int | None = None,
    clockdistill: bool = False,
) -> nn.Module:
    """
    Args:
        num_probe_points: None (по умолчанию) — обычный LwDetrForObjectDetection,
            как раньше. Число > 0 — модель оборачивается в
            _LwDetrWithKDDETRProbes: на каждый forward decoder дополнительно
            прогоняется по num_probe_points probe-точкам с anchor-сетки
            encoder'а ("general distillation points" из KD-DETR, см. docstring
            _LwDetrWithKDDETRProbes), а результат кладётся в outputs.probe_logits
            / outputs.probe_boxes. Нужен только для src/losses/kd_detr_loss.py;
            для DCKD и остальных лоссов должен оставаться None.
        clockdistill: True — модель оборачивается в LwDetrCLoCKDistillMemory
            (src/models/lwdetr_clockdistill.py): наружу дополнительно
            отдаются decoder и геометрия его входа для второго,
            target-aware decoder-прохода. Нужен только для
            src/losses/clockdistill_loss.py; несовместим с
            num_probe_points (обе — разные обёртки одной и той же модели).
    """
    lwdetr_small_checkpoint = "AnnaZhang/lwdetr_small_60e_coco"
    cityscapes_classes = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]

    id2label = {index: class_name for index, class_name in enumerate(cityscapes_classes)}
    label2id = {class_name: index for index, class_name in id2label.items()}

    config = LwDetrConfig.from_pretrained(lwdetr_small_checkpoint)

    config.num_labels = num_classes
    config.id2label = id2label
    config.label2id = label2id
    config.disable_custom_kernels = disable_custom_kernels

    config.dropout = dropout
    config.attention_dropout = attention_dropout
    config.activation_dropout = activation_dropout
    config.backbone_config.dropout_prob = backbone_dropout

    model = LwDetrForObjectDetection.from_pretrained(lwdetr_small_checkpoint, config=config, ignore_mismatched_sizes=True)

    if checkpoint_path is not None:
        weights_path = Path(to_absolute_path(checkpoint_path))
        if not weights_path.is_file():
            raise FileNotFoundError(f"Checkpoint file was not found: {weights_path}")

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

        if isinstance(checkpoint, dict) and "student_state" in checkpoint:
            state_dict = checkpoint["student_state"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

        model.load_state_dict(state_dict, strict=True)

    if num_probe_points is not None and clockdistill:
        raise ValueError("num_probe_points и clockdistill — взаимоисключающие обёртки")

    if num_probe_points is None and not clockdistill:
        return model

    if num_probe_points is not None:
        return LwDetrKDDETRProbes(model, num_probe_points=num_probe_points)

    return LwDetrCLoCKDistillMemory(model)


def yolo(
    model_name: str = "yolov8n",
    num_classes: int = 8,
    weights: str | Path | None = None,
    weights_dir: str | None = "data/weights",
    checkpoint_path: str | Path | None = None,
    backbone_dropout: float = 0.05,
    neck_dropout: float = 0.10,
    bbox_dropout: float = 0.05,
    cls_dropout: float = 0.15,
):
    """Универсальная фабрика для любой YOLO-архитектуры из ultralytics —
    yolov8{n,s,m,l,x}, yolov9{t,s,m,c,e}, yolov10{n,s,m,b,l,x}, yolo11{n,...},
    yolo12{n,...} и т.д. Подходит любое имя, для которого в пакете ultralytics
    есть cfg cfg/models/**/<model_name>.yaml (это все стоковые архитектуры;
    ultralytics сама ищет файл по имени независимо от подпапки).

    Args:
        model_name: имя архитектуры/масштаба ultralytics, например "yolov8n",
            "yolov9t", "yolov10n", "yolo11n". Определяет и cfg (структуру
            сети), и имя файла канонического претрейна.
        weights: путь к локальному чекпоинту; None — канонический
            претрейн (ultralytics сама докачает его в
            <weights_dir>/<model_name>.pt, если файла там ещё нет);
            "random" — без претрейна, чистая инициализация. Загружается
            через ultralytics' YOLO(path).model — ждёт именно её нативный
            формат чекпоинта, не подходит для чекпоинтов этого проекта.
        weights_dir: куда докачивать канонический чекпоинт при
            weights=None; по умолчанию data/weights.
        checkpoint_path: путь к чекпоинту, сохранённому этим проектом
            (best.pt/last.pt с ключом "student_state" — например, обученный
            здесь же учитель-YOLO для дистилляции). Формат другой, чем у
            weights (там ultralytics-нативный), поэтому отдельный параметр
            и отдельная ветка загрузки — та же логика, что уже используется
            в lwdetr_small_for_detection для той же задачи. Если задан,
            полностью заменяет обычную загрузку через weights: канонический
            претрейн не тянется вовсе, он всё равно был бы перезаписан
            ниже strict-загрузкой.
        backbone_dropout, neck_dropout, bbox_dropout, cls_dropout: точечный
            dropout в блоках C2f/SPPF (backbone/neck) и в ветках cv2/cv3
            Detect-головы (bbox/cls). Это инъекция под конкретные типы
            модулей v8-семейства: она работает для архитектур, что
            унаследовали C2f/SPPF/Detect (v8, v10, v11, v12), но НЕ для
            архитектур без них (например чистый YOLOv9 использует
            RepNCSPELAN4/ELAN1/SPPELAN вместо C2f/SPPF). Если запрошен
            ненулевой dropout, а подходящих слоёв в модели не нашлось —
            функция не глотает это молча, а кидает предупреждение.
            Для v10Detect (наследник Detect с доп. one2one-веткой)
            bbox/cls dropout встаёт только в общую cv2/cv3-ветку
            (one2many), one2one-ветка его не получает.
    """
    cityscapes_names = {
        0: "person",
        1: "rider",
        2: "car",
        3: "truck",
        4: "bus",
        5: "train",
        6: "motorcycle",
        7: "bicycle",
    }

    model = DetectionModel(cfg=f"{model_name}.yaml", ch=3, nc=num_classes, verbose=False)
    if num_classes == 8:
        model.names = cityscapes_names

    if checkpoint_path is not None:
        weights_path = Path(to_absolute_path(str(checkpoint_path)))
        if not weights_path.is_file():
            raise FileNotFoundError(f"Checkpoint file was not found: {weights_path}")

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

        if isinstance(checkpoint, dict) and "student_state" in checkpoint:
            state_dict = checkpoint["student_state"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

        model.load_state_dict(state_dict, strict=True)
    elif weights != "random":
        if weights is None:
            weights_path = resolve_weights_dir(weights_dir) / f"{model_name}.pt"
        else:
            weights_path = Path(to_absolute_path(str(weights)))
            if not weights_path.is_file():
                raise FileNotFoundError(
                    f"Файл весов не найден: {weights_path}. "
                    f"Ожидается локальная копия претрейна ({model_name}.pt)."
                )

        pretrained_model = YOLO(str(weights_path)).model
        model.load(pretrained_model, verbose=True)

    dropout_applied = {"backbone": False, "neck": False, "bbox": False, "cls": False}

    for layer in model.model:
        if isinstance(layer, C2f):
            is_backbone = layer.i < 10
            dropout = backbone_dropout if is_backbone else neck_dropout
            if dropout > 0:
                layer.cv2 = nn.Sequential(layer.cv2, nn.Dropout2d(p=dropout))
                dropout_applied["backbone" if is_backbone else "neck"] = True

        elif isinstance(layer, SPPF):
            if backbone_dropout > 0:
                layer.cv2 = nn.Sequential(layer.cv2, nn.Dropout2d(p=backbone_dropout))
                dropout_applied["backbone"] = True

        elif isinstance(layer, Detect):
            if bbox_dropout > 0:
                for branch in layer.cv2:
                    branch.insert(len(branch) - 1, nn.Dropout2d(p=bbox_dropout))
                dropout_applied["bbox"] = True

            if cls_dropout > 0:
                for branch in layer.cv3:
                    branch.insert(len(branch) - 1, nn.Dropout2d(p=cls_dropout))
                dropout_applied["cls"] = True

    requested = {
        "backbone": backbone_dropout > 0,
        "neck": neck_dropout > 0,
        "bbox": bbox_dropout > 0,
        "cls": cls_dropout > 0,
    }
    missed = [name for name, was_requested in requested.items() if was_requested and not dropout_applied[name]]
    if missed:
        warnings.warn(
            f"yolo(model_name={model_name!r}): запрошен dropout для {missed}, "
            "но в этой архитектуре нет подходящих слоёв (C2f/SPPF/Detect) — "
            "dropout не применён.",
            stacklevel=2,
        )

    return model


def rtdetr_for_detection(
    num_classes: int = 8,
    checkpoint_id: str = "PekingU/rtdetr_r50vd",
    disable_custom_kernels: bool = True,
    dropout: float = 0.1,
    checkpoint_path: str | Path | None = None,
) -> nn.Module:
    """RT-DETR (CNN-бэкбон ResNet-50-vd + лёгкий transformer encoder-decoder)
    как учитель/студент для детекции. Задуман как архитектурный «мост» между
    чисто-CNN YOLO и чисто-ViT LW-DETR в прогрессивной KD-цепочке (см.
    разбор статьи MTPD) — feature-only дистилляция для YOLOKDLoss берёт фичу
    с model.encoder_input_proj.2 (последний уровень backbone'а перед
    hybrid encoder'ом, голый тензор [B,256,H,W], тот же stride32, что у
    YOLO-студента — проверено прогоном модели).

    Args:
        checkpoint_id: канонический COCO-претрейн с HuggingFace Hub.
        checkpoint_path: путь к чекпоинту, сохранённому этим проектом
            (best.pt с ключом "student_state") — та же логика, что у
            lwdetr_small_for_detection и yolo(). Если задан, применяется
            ПОСЛЕ канонического COCO-претрейна (strict=True, полностью
            перезаписывает веса).
    """
    cityscapes_classes = [
        "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    ]
    id2label = {index: name for index, name in enumerate(cityscapes_classes)}
    label2id = {name: index for index, name in id2label.items()}

    config = RTDetrConfig.from_pretrained(checkpoint_id)
    config.num_labels = num_classes
    config.id2label = id2label
    config.label2id = label2id
    config.dropout = dropout
    config.disable_custom_kernels = disable_custom_kernels

    model = RTDetrForObjectDetection.from_pretrained(
        checkpoint_id, config=config, ignore_mismatched_sizes=True
    )

    if checkpoint_path is not None:
        weights_path = Path(to_absolute_path(str(checkpoint_path)))
        if not weights_path.is_file():
            raise FileNotFoundError(f"Checkpoint file was not found: {weights_path}")

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

        if isinstance(checkpoint, dict) and "student_state" in checkpoint:
            state_dict = checkpoint["student_state"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

    return model


def faster_rcnn_resnet18_for_detection(
    num_classes: int = 8,
) -> nn.Module:

    backbone_model = resnet18(weights=None)
    
    
    modules = list(backbone_model.children())[:-2]
    backbone = nn.Sequential(*modules)
    
    
    backbone.out_channels = 512
    
    
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    
    model = FasterRCNN(
        backbone,
        num_classes=num_classes + 1,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    
    return model