from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes

# Исправь эти импорты под структуру своего проекта
from src.data.datasets import cityscapes
from src.data.transforms import build_base_transform_for_cityscapes


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

DATA_ROOT = Path("data/raw/cityscapes/leftImg8bit")
ANNOTATION_ROOT = Path("data/raw/cityscapes/annotations")

BATCH_SIZE = 8

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

CITYSCAPES_CLASSES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


# ---------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------

def detection_collate_fn(
    batch: list[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    """
    Не объединяет изображения через torch.stack.

    Это важно, если DetectionRandomResize создает изображения
    разных размеров внутри одного batch.
    """
    images, targets = zip(*batch)

    return list(images), list(targets)


# ---------------------------------------------------------------------
# Denormalization
# ---------------------------------------------------------------------

def denormalize(
    image: Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> Tensor:
    """
    Обратная операция к DetectionNormalize.
    """

    mean_tensor = torch.tensor(
        mean,
        dtype=image.dtype,
        device=image.device,
    ).view(-1, 1, 1)

    std_tensor = torch.tensor(
        std,
        dtype=image.dtype,
        device=image.device,
    ).view(-1, 1, 1)

    image = image * std_tensor + mean_tensor

    return image.clamp(0, 1)


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def show_batch(
    images: list[Tensor],
    targets: list[dict[str, Tensor]],
    class_names: list[str],
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> None:

    batch_size = len(images)

    fig, axes = plt.subplots(
        batch_size,
        1,
        figsize=(16, 7 * batch_size),
    )

    if batch_size == 1:
        axes = [axes]

    for ax, image, target in zip(axes, images, targets):

        # -------------------------------------------------------------
        # Undo normalization
        # -------------------------------------------------------------

        image = denormalize(
            image=image,
            mean=mean,
            std=std,
        )

        # draw_bounding_boxes ожидает uint8 [0, 255]
        image = (image * 255).to(torch.uint8)

        boxes = target["boxes"]

        # -------------------------------------------------------------
        # Convert class ids -> class names
        # -------------------------------------------------------------

        labels = target["labels"].tolist()

        label_names = [
            class_names[label]
            if 0 <= label < len(class_names)
            else str(label)
            for label in labels
        ]

        # -------------------------------------------------------------
        # Draw boxes
        # -------------------------------------------------------------

        if boxes.numel() > 0:
            image = draw_bounding_boxes(
                image=image,
                boxes=boxes,
                labels=label_names,
                width=3,
                font_size=20,
            )

        # CHW -> HWC
        image = image.permute(1, 2, 0).cpu().numpy()

        ax.imshow(image)
        ax.axis("off")

        height, width = image.shape[:2]

        ax.set_title(
            f"Image size: {width} x {height} | "
            f"Objects: {len(boxes)}"
        )

    plt.tight_layout()
    plt.savefig("batch_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved batch to batch_visualization.png")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    transform = build_base_transform_for_cityscapes(
        mean=MEAN,
        std=STD,
        train=True,
        image_size=[1024, 2048]
    )

    dataset = cityscapes(
        root=DATA_ROOT,
        annotation_dir=ANNOTATION_ROOT,
        train=True,
        transform=transform,
        label_offset=0,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=detection_collate_fn,
    )

    # -------------------------------------------------------------
    # Get one batch
    # -------------------------------------------------------------

    images, targets = next(iter(dataloader))

    print(f"Batch size: {len(images)}")

    for i, (image, target) in enumerate(zip(images, targets)):
        print()
        print(f"Image {i}")
        print(f"  shape:  {tuple(image.shape)}")
        print(f"  boxes:  {target['boxes'].shape}")
        print(f"  labels: {target['labels'].tolist()}")

        if "size" in target:
            print(f"  size:   {target['size'].tolist()}")

    # -------------------------------------------------------------
    # Show
    # -------------------------------------------------------------

    show_batch(
        images=images,
        targets=targets,
        class_names=CITYSCAPES_CLASSES,
        mean=MEAN,
        std=STD,
    )


if __name__ == "__main__":
    main()