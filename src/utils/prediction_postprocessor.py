import torch
from torch import Tensor
from torchvision.ops import box_convert

from ultralytics.utils import nms
from ultralytics.utils.tal import dist2bbox, make_anchors

@torch.no_grad()
def lwdetr_prediction_postprocessor(
    outputs,
    images: list[Tensor] | Tensor,
    score_threshold: float = 0.001,
    label_offset: int = 0,
    max_detections: int = 100,
) -> list[dict[str, Tensor]]:
    if isinstance(images, Tensor):
        images = list(images)

    logits = outputs.logits
    pred_boxes = outputs.pred_boxes

    num_classes = logits.shape[-1]
    probabilities = logits.sigmoid()

    # DETR-семейство обучается сигмоидой на класс, и один запрос может отвечать
    # сразу за несколько классов. probabilities.max(dim=-1) оставлял от запроса
    # ровно один вариант и терял recall, поэтому top-k берётся по расплющенной
    # паре (запрос x класс) — так же, как в референсной реализации DETR.
    flat_scores = probabilities.flatten(1)
    top_scores, top_indices = flat_scores.topk(min(max_detections, flat_scores.shape[1]), dim=1)

    query_indices = top_indices // num_classes
    class_indices = top_indices % num_classes

    boxes = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    predictions = []

    for image, image_boxes, image_queries, image_labels, image_scores in zip(
        images,
        boxes,
        query_indices,
        class_indices,
        top_scores,
    ):
        height, width = image.shape[-2:]

        image_boxes = image_boxes[image_queries]
        scale = image_boxes.new_tensor([width, height, width, height])
        image_boxes = image_boxes * scale
        keep = image_scores >= score_threshold

        predictions.append(
            {
                "boxes": image_boxes[keep],
                "scores": image_scores[keep],
                "labels": (image_labels[keep].long() + label_offset),
            }
        )

    return predictions

@torch.no_grad()
def yolov8_prediction_postprocessor(
    outputs,
    images: list[Tensor] | Tensor,
    score_threshold: float = 0.001,
    iou_threshold: float = 0.7,
    label_offset: int = 0,
    strides: tuple[int, ...] = (8, 16, 32),
    reg_max: int = 16,
    max_detections: int = 300,
) -> list[dict[str, Tensor]]:
    if isinstance(outputs, tuple):
        decoded_predictions = outputs[0]

    elif isinstance(outputs, dict):
        pred_distri = outputs["boxes"]
        pred_scores = outputs["scores"]
        feats = outputs["feats"]

        device = pred_distri.device
        dtype = pred_distri.dtype
        stride = torch.tensor(strides, device=device, dtype=dtype)

        anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)

        batch_size = pred_distri.shape[0]
        num_anchors = pred_distri.shape[-1]

        pred_distri = pred_distri.view(batch_size, 4, reg_max, num_anchors)
        pred_distri = pred_distri.softmax(dim=2)

        projection = torch.arange(reg_max, device=device, dtype=dtype)
        pred_distri = (pred_distri * projection.view(1, 1, reg_max, 1)).sum(dim=2)

        boxes = dist2bbox(pred_distri, anchor_points.T.unsqueeze(0), xywh=True, dim=1)
        boxes = boxes * stride_tensor.T.unsqueeze(0)

        scores = pred_scores.sigmoid()
        decoded_predictions = torch.cat((boxes, scores), dim=1)

    else:
        raise TypeError(f"Unsupported YOLO output type: {type(outputs)}")

    detections = nms.non_max_suppression(
        decoded_predictions,
        conf_thres=score_threshold,
        iou_thres=iou_threshold,
        multi_label=True,
        max_det=max_detections,
    )

    predictions = []

    for detection in detections:
        predictions.append(
            {
                "boxes": detection[:, :4],
                "scores": detection[:, 4],
                "labels": (detection[:, 5].long() + label_offset),
            }
        )

    return predictions