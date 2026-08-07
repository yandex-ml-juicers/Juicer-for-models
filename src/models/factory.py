"""Фабрики моделей. Каждая функция — точка входа для _target_ в конфигах
configs/model/teacher/*.yaml и configs/model/student/*.yaml.

Здесь только сборка: сами архитектуры лежат в отдельных модулях
(src/models/segformer.py, src/models/unet.py), а работа с весами —
в src/utils/checkpoints.py, потому что она общая для всех задач.

Заморозка учителя здесь НЕ делается: это политика обучения, а не свойство
модели, и ею владеет Trainer.
"""

import timm
import torch
from torch import nn
from torchvision import models as tv_models
from torchvision.models._api import WeightsEnum
from torchvision.models import get_model

from transformers import LwDetrConfig, LwDetrForObjectDetection

from src.models.segformer import SegFormer
from src.models.stochastic_depth import apply_stochastic_depth
from src.models.unet import UNET_VARIANTS, UNet
from src.utils.checkpoints import load_checkpoint_into, resolve_weights_dir


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
) -> nn.Module:
    lwdetr_small_checkpoint = "AnnaZhang/lwdetr_small_60e_coco"
    cityscapes_classes = [
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]

    id2label = {
        index: class_name
        for index, class_name in enumerate(cityscapes_classes)
    }

    label2id = {
        class_name: index
        for index, class_name in id2label.items()
    }

    config = LwDetrConfig.from_pretrained(
        lwdetr_small_checkpoint,
        id2label=id2label,
        label2id=label2id,
        disable_custom_kernels=disable_custom_kernels,
    )

    return LwDetrForObjectDetection(config)
