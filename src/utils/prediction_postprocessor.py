import torch
from torch import Tensor
from torchvision.ops import box_convert


@torch.no_grad()
def prediction_postprocessor(
    outputs: dict[str, Tensor],
    images: list[Tensor] | Tensor,
    score_threshold: float = 0.05,
    label_offset: int = 1,
) -> list[dict[str, Tensor]]:
    if isinstance(images, Tensor):
        images = list(images)

    probabilities = outputs["pred_logits"].softmax(dim=-1)[..., :-1]
    scores, labels = probabilities.max(dim=-1)

    boxes = box_convert(
        outputs["pred_boxes"],
        in_fmt="cxcywh",
        out_fmt="xyxy",
    )

    predictions = []

    for image, image_boxes, image_scores, image_labels in zip(
        images,
        boxes,
        scores,
        labels,
    ):
        height, width = image.shape[-2:]

        image_boxes = image_boxes * image_boxes.new_tensor(
            [width, height, width, height]
        )

        keep = image_scores >= score_threshold

        predictions.append({
            "boxes": image_boxes[keep],
            "scores": image_scores[keep],
            "labels": image_labels[keep].long() + label_offset,
        })

    return predictions