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
    if mirror_url is not None:
        torchvision.datasets.CIFAR10.url = mirror_url
    return torchvision.datasets.CIFAR10(
        root=to_absolute_path(root),
        train=train,
        download=download,
        transform=transform,
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
