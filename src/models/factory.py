"""Фабрики моделей. Каждая функция — точка входа для _target_ в конфигах
configs/model/teacher/*.yaml и configs/model/student/*.yaml.

Заморозка учителя здесь НЕ делается: это политика обучения, а не свойство
модели, и ею владеет Trainer.
"""

import timm
from pathlib import Path
import torchvision
import torch
from torch import nn
from torchvision import models as tv_models
from torchvision.models._api import WeightsEnum
from torchvision.models import get_model
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet18, ResNet18_Weights
from transformers import LwDetrConfig, LwDetrForObjectDetection

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


def lwdetr_small_for_detection(
    num_classes: int = 8,
    disable_custom_kernels: bool = True,
) -> nn.Module:
    from transformers import LwDetrConfig, LwDetrForObjectDetection
    
    # Берем только чертеж архитектуры Small
    model_name = "AnnaZhang/lwdetr_small_60e_coco"
    
    # Конфигурация классов Cityscapes
    cityscapes_classes = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
    id2label = {i: name for i, name in enumerate(cityscapes_classes)}
    label2id = {name: i for i, name in id2label.items()}

    # 1. Загружаем РОДНОЙ конфиг (в нем уже прописано num_feature_levels=2)
    config = LwDetrConfig.from_pretrained(model_name)
    
    # 2. Обновляем только классы и включаем Group DETR (это ускорит Scratch)
    config.update({
        "num_labels": num_classes,
        "id2label": id2label,
        "label2id": label2id,
        "disable_custom_kernels": disable_custom_kernels,
        "use_group_detr": True,
        "num_groups": 11
    })

    # 3. Создаем "пустую" модель (ВЕСА НЕ ГРУЗЯТСЯ, ЭТО SCRATCH)
    model = LwDetrForObjectDetection(config)
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
