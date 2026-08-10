"""Фабрики моделей. Каждая функция — точка входа для _target_ в конфигах
configs/model/teacher/*.yaml и configs/model/student/*.yaml.

Заморозка учителя здесь НЕ делается: это политика обучения, а не свойство
модели, и ею владеет Trainer.
"""

import timm
from pathlib import Path

import torch
from torch import nn
from torchvision import models as tv_models
from torchvision.models._api import WeightsEnum
from torchvision.models import get_model

from transformers import LwDetrConfig, LwDetrForObjectDetection

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.head import Detect

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
    checkpoint_path: str | None = None
) -> nn.Module:
    """
    Function for load models from torchvision
    Editing the last layer for current num of classes
    """

    if isinstance(weights, str):
        weights = tv_models.get_weight(weights)

    model = get_model(
        model_name,
        weights=weights,
        num_classes=num_classes
    )

    if checkpoint_path is None:
        return model
    
    weights_path = Path(to_absolute_path(checkpoint_path))
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file was not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"A checkpoint with a dict-style weight was expected, but it was received {type(checkpoint)}")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "student_state" in checkpoint:
        state_dict = checkpoint["student_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=True)

    print(f"Weights for {model_name} has been loaded: {weights_path}")

    return model

def timm_model_for_classification(
    model_name: str, 
    pretrained: bool = False,
    checkpoint_path: str | None = None, 
    num_classes: int = 100,
    **kwargs: any,
) -> nn.Module:

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    )

    if checkpoint_path is None:
        return model

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

    print(f"Weights for {model_name} has been loaded: {weights_path}")

    return model
    

class UNetDoubleConv(nn.Sequential):
    """Conv-BN-ReLU x2 — базовый блок U-Net.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Классический U-Net: симметричный энкодер-декодер со skip-connections.

    Свёртки с padding=1 сохраняют размер, поэтому выход совпадает по разрешению
    со входом и логиты [B, num_classes, H, W] можно скармливать cross_entropy
    напрямую, без интерполяции. Требование: H и W кратны 2**depth
    (512x1024 и 1024x2048 при depth=4 подходят).
    """

    def __init__(
        self,
        num_classes: int = 19,
        in_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()

        channels = [base_channels * 2 ** level for level in range(depth + 1)]

        self.encoders = nn.ModuleList()
        previous_channels = in_channels
        for level_channels in channels[:-1]:
            self.encoders.append(UNetDoubleConv(previous_channels, level_channels))
            previous_channels = level_channels

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = UNetDoubleConv(channels[-2], channels[-1])

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level_channels in reversed(channels[:-1]):
            self.upsamples.append(
                nn.ConvTranspose2d(level_channels * 2, level_channels, kernel_size=2, stride=2)
            )
            # На вход декодера идёт конкатенация апсемпла и skip-связи.
            self.decoders.append(UNetDoubleConv(level_channels * 2, level_channels))

        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        skips = []

        features = images
        for encoder in self.encoders:
            features = encoder(features)
            skips.append(features)
            features = self.pool(features)

        features = self.bottleneck(features)

        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            features = upsample(features)
            features = torch.cat([skip, features], dim=1)
            features = decoder(features)

        return self.head(features)


def unet_for_segmentation(
    num_classes: int = 19,
    in_channels: int = 3,
    base_channels: int = 64,
    depth: int = 4,
    checkpoint_path: str | None = None,
) -> nn.Module:
    """U-Net для семантической сегментации.

    checkpoint_path нужен на втором этапе: обученный учитель подгружается
    из чекпоинта тренера (ключ student_state) и идёт в дистилляцию.
    """
    model = UNet(
        num_classes=num_classes,
        in_channels=in_channels,
        base_channels=base_channels,
        depth=depth,
    )

    if checkpoint_path is None:
        return model

    weights_path = Path(to_absolute_path(checkpoint_path))
    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint file was not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict) and "student_state" in checkpoint:
        state_dict = checkpoint["student_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=True)

    print(f"Weights for U-Net has been loaded: {weights_path}")

    return model


def lwdetr_small_for_detection(
    num_classes: int = 8, 
    disable_custom_kernels: bool = True, 
    dropout: float = 0.1, 
    attention_dropout: float = 0.1, 
    activation_dropout: float = 0.1, 
    backbone_dropout: float = 0.0, 
    checkpoint_path: str | Path | None = None
) -> nn.Module:
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

    if checkpoint_path is None:
        return model

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

    return model


def yolov8n(
    num_classes: int = 8,
    weights: str | Path | None = "yolov8n.pt",
    backbone_dropout: float = 0.05,
    neck_dropout: float = 0.10,
    bbox_dropout: float = 0.05,
    cls_dropout: float = 0.15,
):
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

    model = DetectionModel(cfg="yolov8n.yaml", ch=3, nc=num_classes, verbose=False)
    if num_classes == 8:
        model.names = cityscapes_names

    if weights is not None:
        pretrained_model = YOLO(str(weights)).model
        model.load(pretrained_model, verbose=True)

    for layer in model.model:
        if isinstance(layer, C2f):
            dropout = (backbone_dropout if layer.i < 10 else neck_dropout)
            layer.cv2 = nn.Sequential(layer.cv2, nn.Dropout2d(p=dropout))

        elif isinstance(layer, SPPF):
            layer.cv2 = nn.Sequential(layer.cv2, nn.Dropout2d(p=backbone_dropout))

        elif isinstance(layer, Detect):
            for branch in layer.cv2:
                branch.insert(len(branch) - 1, nn.Dropout2d(p=bbox_dropout))

            for branch in layer.cv3:
                branch.insert(len(branch) - 1, nn.Dropout2d(p=cls_dropout))

    return model