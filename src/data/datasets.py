"""Фабрики датасетов. Вызываются через hydra.utils.instantiate из cfg.data.dataset
с дозаполнением аргументов train= и transform= в рантайме (см. loaders.py)."""

import zipfile
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from datasets import DownloadConfig, concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download
from hydra.utils import to_absolute_path
from PIL import Image

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

CITYSCAPES_SEGMENTATION_CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)
# Официальные цвета Cityscapes в порядке trainId — те же, что в публикациях
# и в notebooks/segmentation_predictions_visualization.ipynb. Держим рядом с
# именами классов, чтобы предсказания везде раскрашивались одинаково.
CITYSCAPES_SEGMENTATION_PALETTE = (
    (128, 64, 128),   # road
    (244, 35, 232),   # sidewalk
    (70, 70, 70),     # building
    (102, 102, 156),  # wall
    (190, 153, 153),  # fence
    (153, 153, 153),  # pole
    (250, 170, 30),   # traffic light
    (220, 220, 0),    # traffic sign
    (107, 142, 35),   # vegetation
    (152, 251, 152),  # terrain
    (70, 130, 180),   # sky
    (220, 20, 60),    # person
    (255, 0, 0),      # rider
    (0, 0, 142),      # car
    (0, 0, 70),       # truck
    (0, 60, 100),     # bus
    (0, 80, 100),     # train
    (0, 0, 230),      # motorcycle
    (119, 11, 32),    # bicycle
)

# Канонический маппинг Cityscapes labelId -> trainId (19 оценочных классов).

CITYSCAPES_LABEL_ID_TO_TRAIN_ID = {
    7: 0,    # road
    8: 1,    # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}


class CityscapesSegmentation(Dataset):
    """
    Ожидаемая структура:

        root/leftImg8bit/<split>/<city>/<name>_leftImg8bit.png
        root/gtFine/<split>/<city>/<name>_gtFine_labelIds.png

    Возвращает (image, mask); mask — uint8 [H, W] с trainIds 0..18
    и ignore_index там, где класс не входит в оценочные. В int64 маску
    переводит SegmentationToTensor в конце трансформа.
    """

    IMAGE_SUFFIX = "_leftImg8bit.png"
    MASK_SUFFIX = "_gtFine_labelIds.png"

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Callable | None = None,
        ignore_index: int = 255,
    ) -> None:
        if split not in {"train", "val"}:
            # test-разметка в Cityscapes — заглушка
            raise ValueError(f"split должен быть 'train' или 'val', получено {split!r}")

        root = Path(root)
        image_dir = root / "leftImg8bit" / split
        mask_dir = root / "gtFine" / split

        if not image_dir.is_dir():
            raise FileNotFoundError(f"Cannot find image directory: {image_dir}")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Cannot find mask directory: {mask_dir}")

        self.image_paths = sorted(image_dir.glob(f"*/*{self.IMAGE_SUFFIX}"))
        if not self.image_paths:
            raise FileNotFoundError(f"No images were found in {image_dir}")

        self.mask_paths = []
        for image_path in self.image_paths:
            base_name = image_path.name.removesuffix(self.IMAGE_SUFFIX)
            mask_path = mask_dir / image_path.parent.name / f"{base_name}{self.MASK_SUFFIX}"

            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Mask for image {image_path} was not found: {mask_path}"
                )

            self.mask_paths.append(mask_path)

        # Таблица подстановки на все 256 возможных значений uint8: всё, чего нет
        # в маппинге (включая мусорные значения вроде license plate), падает в ignore_index
        self.label_id_to_train_id = np.full(256, ignore_index, dtype=np.uint8)
        for label_id, train_id in CITYSCAPES_LABEL_ID_TO_TRAIN_ID.items():
            self.label_id_to_train_id[label_id] = train_id

        self.transform = transform
        self.ignore_index = ignore_index
        self.classes = CITYSCAPES_SEGMENTATION_CLASSES
        self.palette = CITYSCAPES_SEGMENTATION_PALETTE
        self.num_classes = len(CITYSCAPES_SEGMENTATION_CLASSES)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[index]).convert("RGB")

        raw_mask = np.array(Image.open(self.mask_paths[index]))
        if raw_mask.dtype != np.uint8:
            raise TypeError(
                f"Ожидалась 8-битная маска labelIds, получено {raw_mask.dtype} "
                f"в {self.mask_paths[index]}"
            )

        mask = torch.from_numpy(self.label_id_to_train_id[raw_mask])

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image = pil_to_tensor(image).float() / 255.0
            mask = mask.to(torch.int64)

        return image, mask


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

    root = Path(to_absolute_path(root))
    annotation_dir = Path(to_absolute_path(annotation_dir))

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


def cityscapes_segmentation(
    root: str | Path,
    train: bool,
    transform: Callable | None = None,
    ignore_index: int = 255,
) -> Dataset:
    """Create a Cityscapes semantic segmentation dataset.

    root указывает на каталог, где рядом лежат leftImg8bit/ и gtFine/
    (в отличие от детекционной cityscapes(), которой передаётся сам
    leftImg8bit/ плюс отдельный каталог с COCO-аннотациями).
    """
    root = Path(to_absolute_path(str(root)))

    return CityscapesSegmentation(
        root=root,
        split="train" if train else "val",
        transform=transform,
        ignore_index=ignore_index,
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
