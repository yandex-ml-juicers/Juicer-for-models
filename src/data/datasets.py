"""Фабрики датасетов. Вызываются через hydra.utils.instantiate из cfg.data.dataset
с дозаполнением аргументов train= и transform= в рантайме (см. loaders.py)."""

import random
from collections import defaultdict
from typing import Callable

import torchvision
from datasets import DownloadConfig, concatenate_datasets, load_dataset
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset


class HFImageNet(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable = None,
        download: bool = True,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError(
                "split должен быть 'train' или 'validation'"
            )

        self.dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split=split,
            cache_dir=to_absolute_path(root),
            download_config=DownloadConfig(
                local_files_only=not download,
            ),
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]

        image = item["image"].convert("RGB")
        label = int(item["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


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

def imagenet_1k(
    root: str,
    train: bool,
    transform: Callable | None = None,
    download: bool = True,
) -> Dataset:
    """ImageNet-1K dataset.

    The official ImageNet-1K dataset is used for image classification tasks.
    """
    
    return HFImageNet(
        root=root,
        split="train" if train else "validation",
        transform=transform,
        download=download,
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
