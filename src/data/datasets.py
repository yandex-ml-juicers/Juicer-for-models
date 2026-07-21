"""Фабрики датасетов. Вызываются через hydra.utils.instantiate из cfg.data.dataset
с дозаполнением аргументов train= и transform= в рантайме (см. loaders.py)."""

import zipfile
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torchvision
from datasets import DownloadConfig, concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset


class HFImageNet(Dataset):
    IMAGENET100_REPO = "asafaa/imagent100"

    def __init__(
        self,
        root: str,
        repo_path: str,
        split: str,
        transform: Callable | None = None,
        download: bool = True,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split должен быть 'train' или 'validation'")

        root_path = Path(to_absolute_path(root))

        self.dataset = load_dataset(
            repo_path,
            split=split,
            cache_dir=str(root_path),
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
    if mirror_url is not None:
        torchvision.datasets.CIFAR10.url = mirror_url
    return torchvision.datasets.CIFAR10(
        root=to_absolute_path(root),
        train=train,
        download=download,
        transform=transform,
    )


def imagenet(
    root: str,
    repo_path: str,
    train: bool,
    transform: Callable | None = None,
    download: bool = True,
) -> Dataset:
    """Function for loading imagenet datasets."""

    return HFImageNet(
        root=root,
        repo_path=repo_path,
        split="train" if train else "validation",
        transform=transform,
        download=download,
    )

def fake_cifar10(
    train: bool,
    transform: Callable | None = None,
    train_size: int = 512,
    eval_size: int = 256,
    num_classes: int = 10,
) -> Dataset:
    return torchvision.datasets.FakeData(
        size=train_size if train else eval_size,
        image_size=(3, 32, 32),
        num_classes=num_classes,
        transform=transform,
    )
