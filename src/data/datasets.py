"""Фабрики датасетов. Вызываются через hydra.utils.instantiate из cfg.data.dataset
с дозаполнением аргументов train= и transform= в рантайме (см. loaders.py)."""

from typing import Callable

import torchvision
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset


def cifar10(
    root: str,
    train: bool,
    transform: Callable | None = None,
    download: bool = True,
    mirror_url: str | None = None,
) -> Dataset:
    """CIFAR-10 из torchvision.

    Args:
        root: путь к данным относительно корня репозитория (Hydra меняет cwd
            на run-dir, поэтому путь разворачивается через to_absolute_path).
        mirror_url: альтернативный URL архива, если оригинальный недоступен
            (архив тот же, md5 совпадает — иначе torchvision его отвергнет).
    """
    if mirror_url is not None:
        torchvision.datasets.CIFAR10.url = mirror_url
    return torchvision.datasets.CIFAR10(
        root=to_absolute_path(root),
        train=train,
        download=download,
        transform=transform,
    )


def fake_cifar_like(
    train: bool,
    transform: Callable | None = None,
    train_size: int = 512,
    eval_size: int = 256,
    num_classes: int = 10,
) -> Dataset:
    """Синтетический датасет формы CIFAR-10 (3x32x32) для смоук-тестов и CI:
    проверяет сборку всего пайплайна без сети и скачивания данных."""
    return torchvision.datasets.FakeData(
        size=train_size if train else eval_size,
        image_size=(3, 32, 32),
        num_classes=num_classes,
        transform=transform,
    )
