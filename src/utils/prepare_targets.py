from collections.abc import Sequence

import torch
from torchvision.ops import box_convert

def prepare_targets(
    targets: Sequence[dict],
    device: torch.device | str,
    mode: str,
) -> list[dict]:
    if mode == "lw-detr-small":
        prepared_targets = []

        for target in targets:
            class_labels = target["labels"].to(device=device, dtype=torch.long, non_blocking=True)
            boxes = target["boxes"].to(device=device, dtype=torch.float32, non_blocking=True)

            size = target["size"].to(device=device, dtype=torch.float32, non_blocking=True)
            height, width = size.unbind()

            boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
            scale = torch.stack([width, height, width, height])
            boxes = (boxes / scale).clamp(0.0, 1.0)

            prepared_targets.append(
                {
                    "class_labels": class_labels,
                    "boxes": boxes,
                }
            )

        return prepared_targets

    if mode == "yolov8n":
        batch_idx = []
        classes = []
        bboxes = []

        for image_idx, target in enumerate(targets):
            labels = target["labels"].to(device=device, dtype=torch.float32, non_blocking=True)
            boxes = target["boxes"].to(device=device, dtype=torch.float32, non_blocking=True)
            size = target["size"].to(device=device, dtype=torch.float32, non_blocking=True)

            # Если labels в датасете идут от 1 до 8
            labels = labels - 1

            height, width = size.unbind()

            boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
            scale = torch.stack([width, height, width, height])
            boxes = (boxes / scale).clamp(0.0, 1.0)

            indices = torch.full((len(labels),), image_idx, dtype=torch.long, device=device)

            batch_idx.append(indices)
            classes.append(labels)
            bboxes.append(boxes)

        return {
            "batch_idx": torch.cat(batch_idx),
            "cls": torch.cat(classes),
            "bboxes": torch.cat(bboxes),
        }

    return [
        {
            key: (
                value.to(
                    device,
                    non_blocking=True,
                )
                if isinstance(value, Tensor)
                else value
            )
            for key, value in target.items()
        }
        for target in targets
    ]