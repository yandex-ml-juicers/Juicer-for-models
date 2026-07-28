"""Фабрики моделей. Каждая функция — точка входа для _target_ в конфигах
configs/model/teacher/*.yaml и configs/model/student/*.yaml.

Заморозка учителя здесь НЕ делается: это политика обучения, а не свойство
модели, и ею владеет Trainer.
"""

import torch
from torch import nn
from torchvision import models as tv_models
from torchvision.models._api import WeightsEnum

from hydra.utils import to_absolute_path
from pathlib import Path


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

def torchvision_model(
    model_name: str, 
    num_classes: int = 1000,
    weights: WeightsEnum | None = None, 
    checkpoint_path: str | None = None
) -> nn.Module:
    """
    Function for load models from torchvision
    Edit last layer for current classes
    """

    # Создание модели с выходным слоем на определенное количество классов и с весами из torchvision
    model = get_model(
        model_name,
        weights=weights,
        num_classes=num_classes
    )

    if checkpoint_path is None:
        return model
    
    weights_path = Path(to_absolute_path(checkpoint_path))
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file was not found: {path}")

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"A checkpoint with a dict-style weight was expected, but it was received {type(checkpoint)}")

    # Trainer сохраянет "student_state", перебор на случай изменений
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

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key.removeprefix("module.")

        if key.startswith("_orig_mod."):
            key = key.removeprefix("_orig_mod.")

        cleaned_state_dict[key] = value

    incompatible = model.load_state_dict(cleaned_state_dict, strict=True)

    print(f"Weights for {model_name} has been uploaded: {weights_path}")

    return model