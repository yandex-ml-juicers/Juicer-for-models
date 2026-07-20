"""Фабрики моделей. Каждая функция — точка входа для _target_ в конфигах
configs/model/teacher/*.yaml и configs/model/student/*.yaml.

Заморозка учителя здесь НЕ делается: это политика обучения, а не свойство
модели, и ею владеет Trainer.
"""

import torch
from torch import nn
from torchvision import models as tv_models

from hydra.utils import to_absolute_path
from pathlib import Path


def from_torch_hub(repo: str, name: str, pretrained: bool = False) -> nn.Module:
    """Модель из torch.hub (например, chenyaofo/pytorch-cifar-models).

    trust_repo=True отключает интерактивный вопрос про доверие к репозиторию —
    скрипт обязан работать без stdin (запуски в tmux/CI).
    """
    return torch.hub.load(repo, name, pretrained=pretrained, trust_repo=True)


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
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def imagenet_resnet18(num_classes: int = 1000) -> nn.Module:
    """ResNet-18 classifier for ImageNet-1K."""
    model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def imagenet_resnet50_pretrained(num_classes: int = 1000, checkpoint_path: str | None = None) -> nn.Module:
    if checkpoint_path is None:
        return tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    
    model = tv_models.resnet50(weights=None,num_classes=num_classes,)
    path = Path(to_absolute_path(checkpoint_path))

    if not path.exists():
        raise FileNotFoundError(f"Файл с весами не найден: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Ожидался checkpoint в виде dict, получен {type(checkpoint)}")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "student_state" in checkpoint:
        state_dict = checkpoint["student_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    else:
        # Файл может содержать непосредственно state_dict.
        state_dict = checkpoint

    # Убираем префиксы, возникающие после DataParallel/DDP/torch.compile.
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key.removeprefix("module.")

        if key.startswith("_orig_mod."):
            key = key.removeprefix("_orig_mod.")

        cleaned_state_dict[key] = value

    incompatible = model.load_state_dict(cleaned_state_dict, strict=True)

    print(f"Веса ResNet-50 загружены из: {path}")

    return model


def imagenet_resnet152_pretrained(num_classes: int = 1000) -> nn.Module:
    return tv_models.resnet152(weights=tv_models.ResNet152_Weights.IMAGENET1K_V2)

def imagenet_resnet50(num_classes: int = 1000) -> nn.Module:
    """ResNet-50 classifier for ImageNet-1K."""
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def imagenet_resnet152(num_classes: int = 1000) -> nn.Module:
    """ResNet-152 classifier for ImageNet-1K."""
    model = tv_models.resnet152(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model