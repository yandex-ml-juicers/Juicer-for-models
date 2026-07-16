"""Фабрики моделей. Каждая функция — точка входа для _target_ в конфигах
configs/model/teacher/*.yaml и configs/model/student/*.yaml.

Заморозка учителя здесь НЕ делается: это политика обучения, а не свойство
модели, и ею владеет Trainer.
"""

import torch
from torch import nn
from torchvision import models as tv_models


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
