"""Фабрики датасетов. Вызываются через hydra.utils.instantiate из cfg.data.dataset
с дозаполнением аргументов train= и transform= в рантайме (см. loaders.py)."""

import zipfile
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

from datasets import DownloadConfig, concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download
from hydra.utils import to_absolute_path

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from torchvision.transforms.functional import pil_to_tensor


class HFImageNet(Dataset):
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

class CityscapesDetection(CocoDetection):
    def __init__(
        self,
        image_dir: str | Path,
        annotation_file: str | Path,
        transform: Callable | None = None,
        label_offset: int = 1,
    ) -> None:
        super().__init__(
            root=str(image_dir),
            annFile=str(annotation_file),
        )

        self.transform = transform
        self.label_offset = label_offset

        category_ids = sorted(self.coco.getCatIds())
        self.category_id_to_label = {
            category_id: index + label_offset
            for index, category_id in enumerate(category_ids)
        }

        self.label_to_name = {
            index + label_offset: self.coco.cats[category_id]["name"]
            for index, category_id in enumerate(category_ids)
        }

        self.num_classes = len(category_ids)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image, annotations = super().__getitem__(index)

        image_id = self.ids[index]
        image_width, image_height = image.size

        boxes = []
        labels = []
        areas = []
        crowds = []

        for annotation in annotations:
            x, y, width, height = annotation["bbox"]

            x = float(x)
            y = float(y)
            width = float(width)
            height = float(height)

            if width <= 0 or height <= 0:
                continue

            boxes.append([x, y, x + width, y + height])
            labels.append(self.category_id_to_label[annotation["category_id"]])
            areas.append(float(annotation.get("area", width * height)))
            crowds.append(int(annotation.get("iscrowd", 0)))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(crowds, dtype=torch.int64),
            "orig_size": torch.tensor([image_height, image_width], dtype=torch.int64),
            "size": torch.tensor([image_height, image_width], dtype=torch.int64),
        }

        if self.transform is not None:
            image, target = self.transform(image, target)
        else:
            image = pil_to_tensor(image).float() / 255.0

        return image, target

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


def cityscapes(
    root: str | Path,
    annotation_dir: str | Path,
    train: bool,
    transform: Callable | None = None,
    label_offset: int = 1
) -> Dataset:
    """Create a Cityscapes detection dataset."""

    root = Path(root)
    annotation_dir = Path(annotation_dir)

    if train:
        image_dir = root / "train"
        annotation_file = annotation_dir / "train.json"
    else:
        image_dir = root / "val"
        annotation_file = annotation_dir / "val.json"

    if not image_dir.exists():
        raise FileNotFoundError(f"Cannot find image directory: {image_dir}")

    if not annotation_file.is_file():
        raise FileNotFoundError(
            f"Cannot find annotation file: {annotation_file}"
        )

    return CityscapesDetection(
        image_dir=image_dir,
        annotation_file=annotation_file,
        transform=transform,
        label_offset=label_offset,
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
