import torch
from torchvision.utils import draw_bounding_boxes, make_grid


def visualize_detection(
    image: torch.Tensor,
    target: dict,
    prediction: dict,
    label_to_name: dict[int, str],
    score_threshold: float = 0.3,
) -> torch.Tensor:

    image = image.detach().cpu()

    # если image после Normalize — сначала денормализовать
    image = (image * 255).to(torch.uint8)

    gt_labels = [label_to_name[int(label)] for label in target["labels"]]
    gt_image = draw_bounding_boxes(image=image, boxes=target["boxes"].cpu(), labels=gt_labels, width=3)

    scores = prediction["scores"].cpu()
    keep = scores >= score_threshold

    boxes = prediction["boxes"].cpu()[keep]
    labels = prediction["labels"].cpu()[keep]
    scores = scores[keep]

    pred_labels = [f"{label_to_name[int(label)]} {score:.2f}" for label, score in zip(labels, scores)]
    pred_image = draw_bounding_boxes(
        image=image,
        boxes=boxes,
        labels=pred_labels,
        width=3,
    )

    # GT | Prediction
    return make_grid([gt_image, pred_image], nrow=2, padding=10)