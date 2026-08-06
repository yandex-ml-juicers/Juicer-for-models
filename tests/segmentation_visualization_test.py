"""
Запуск из корня репозитория:
    python tests/segmentation_visualization_test.py
    python tests/segmentation_visualization_test.py --count 5 --split val
    python tests/segmentation_visualization_test.py --seed 0 --save outputs/preview.png

override:
    python tests/segmentation_visualization_test.py data.dataset.crop_size=[768,768]
"""

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from src.data.datasets import CITYSCAPES_SEGMENTATION_CLASSES, CityscapesSegmentation

CITYSCAPES_PALETTE = (
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

DEFAULT_OVERRIDES = [
    "data/dataset=cityscapes_seg",
    "data/transform/train=cityscapes_seg_transform",
    "data/transform/eval=cityscapes_seg_transform",
    "task_type=segmentation",
]


def build_color_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for train_id, color in enumerate(CITYSCAPES_PALETTE):
        lut[train_id] = color
    return lut


def colorize_mask(mask: torch.Tensor, lut: np.ndarray) -> np.ndarray:
    return lut[mask.numpy().astype(np.uint8)]


def to_uint8_image(image: torch.Tensor, mean=None, std=None) -> np.ndarray:
    image = image.detach().float().cpu()

    if mean is not None and std is not None:
        mean_tensor = torch.tensor(mean).view(-1, 1, 1)
        std_tensor = torch.tensor(std).view(-1, 1, 1)
        image = image * std_tensor + mean_tensor

    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return (image * 255).round().astype(np.uint8)


def blend(image: np.ndarray, color_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    mixed = (1 - alpha) * image.astype(np.float32) + alpha * color_mask.astype(np.float32)
    return mixed.round().clip(0, 255).astype(np.uint8)


def describe_mask(mask: torch.Tensor, ignore_index: int) -> str:
    total = mask.numel()
    values, counts = torch.unique(mask, return_counts=True)

    parts = []
    for value, count in sorted(zip(values.tolist(), counts.tolist()), key=lambda pair: -pair[1]):
        share = 100 * count / total
        name = "IGNORE" if value == ignore_index else CITYSCAPES_SEGMENTATION_CLASSES[value]
        parts.append(f"{name} {share:.1f}%")

    return " | ".join(parts)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Показать случайные картинки датасета до и после трансформа.",
    )
    parser.add_argument("--count", type=int, default=3, help="сколько картинок показать")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument(
        "--pipeline",
        choices=["train", "eval"],
        default=None,
        help="какой трансформ применять; по умолчанию совпадает со split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="фиксирует и выбор картинок, и аугментации; без него каждый запуск новый",
    )
    parser.add_argument("--root", type=Path, default=None, help="каталог с leftImg8bit/ и gtFine/")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="сохранить в файл вместо показа окна",
    )
    return parser.parse_known_args()


def main() -> int:
    args, overrides = parse_args()
    pipeline = args.pipeline or ("train" if args.split == "train" else "eval")

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="config", overrides=DEFAULT_OVERRIDES + overrides)

    mean = list(cfg.data.dataset.normalize.mean)
    std = list(cfg.data.dataset.normalize.std)
    ignore_index = cfg.data.dataset.ignore_index

    root = args.root if args.root is not None else Path(cfg.data.dataset.build.root)
    if not root.is_absolute():
        root = REPO_ROOT / root

    raw_dataset = CityscapesSegmentation(root=root, split=args.split, ignore_index=ignore_index)
    transformed_dataset = CityscapesSegmentation(
        root=root,
        split=args.split,
        transform=instantiate(cfg.data.transform[pipeline]),
        ignore_index=ignore_index,
    )

    if args.seed is not None:
        random.seed(args.seed)

    indices = random.sample(range(len(raw_dataset)), min(args.count, len(raw_dataset)))

    print(f"Датасет: {root}")
    print(f"split={args.split} ({len(raw_dataset)} изображений), трансформ={pipeline}")
    print(f"normalize: mean={mean} std={std}\n")

    lut = build_color_lut()
    figure, axes = plt.subplots(len(indices), 4, figsize=(18, 2.6 * len(indices) + 1), squeeze=False)

    for row, index in enumerate(indices):
        raw_image, raw_mask = raw_dataset[index]
        model_image, model_mask = transformed_dataset[index]

        raw_height, raw_width = raw_image.shape[-2:]
        model_height, model_width = model_image.shape[-2:]
        name = f"{raw_dataset.image_paths[index].parent.name}/{raw_dataset.image_paths[index].name}"

        print(f"[{row + 1}/{len(indices)}] {name}  (индекс {index})")
        print(f"  оригинал:    {raw_width}x{raw_height} (WxH)  image {raw_image.dtype}, mask {raw_mask.dtype}")
        print(
            f"  вход модели: {model_width}x{model_height} (WxH)  image {model_image.dtype} "
            f"[{model_image.min():.2f}, {model_image.max():.2f}], mask {model_mask.dtype}"
        )
        print(f"  классы на входе модели: {describe_mask(model_mask, ignore_index)}\n")

        raw_display = to_uint8_image(raw_image)
        raw_color_mask = colorize_mask(raw_mask, lut)
        model_display = to_uint8_image(model_image, mean, std)
        model_color_mask = colorize_mask(model_mask, lut)

        panels = (
            (raw_display, f"оригинал\n{raw_width}x{raw_height}"),
            (blend(raw_display, raw_color_mask), f"оригинал + маска\n{raw_width}x{raw_height}"),
            (model_display, f"вход модели ({pipeline})\n{model_width}x{model_height}"),
            (
                blend(model_display, model_color_mask),
                f"вход модели + маска\n{model_width}x{model_height}",
            ),
        )

        for column, (picture, title) in enumerate(panels):
            axis = axes[row][column]
            axis.imshow(picture)
            axis.set_title(title, fontsize=9)
            axis.axis("off")

    figure.suptitle(
        f"Cityscapes {args.split}: слева — как лежит на диске, справа — что уходит в модель "
        f"(денормализовано для показа)",
        fontsize=11,
    )
    figure.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=110, bbox_inches="tight")
        print(f"Сохранено: {args.save}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
