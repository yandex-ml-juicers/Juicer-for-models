from collections.abc import Sequence

import torch
from torchvision.utils import draw_bounding_boxes, make_grid


def visualize_detection(
    image: torch.Tensor,
    target: dict,
    prediction: dict,
    label_to_name: dict[int, str],
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    score_threshold: float = 0.3,
) -> torch.Tensor:

    image = image.detach().cpu().float()

    # Денормализация ровно тогда, когда была нормализация: у YOLO вход
    # остаётся в 0..1, и прежние ImageNet-константы обесцвечивали кадр.
    if mean is not None and std is not None:
        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        image = image * std_tensor + mean_tensor

    image = image.clamp(0, 1)
    image = (image * 255).to(torch.uint8)

    gt_labels = [label_to_name.get(int(label), str(int(label))) for label in target["labels"]]
    gt_image = draw_bounding_boxes(image=image, boxes=target["boxes"].cpu(), labels=gt_labels, width=3)

    scores = prediction["scores"].cpu()
    keep = scores >= score_threshold

    boxes = prediction["boxes"].cpu()[keep]
    labels = prediction["labels"].cpu()[keep]
    scores = scores[keep]

    pred_labels = [
        f"{label_to_name.get(int(label), str(int(label)))} {score:.2f}"
        for label, score in zip(labels, scores)
    ]
    pred_image = draw_bounding_boxes(
        image=image,
        boxes=boxes,
        labels=pred_labels,
        width=3,
    )

    # GT | Prediction
    return make_grid([gt_image, pred_image], nrow=2, padding=10)