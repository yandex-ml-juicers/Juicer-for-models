import torch
from torch import Tensor
from torchvision.ops import box_convert



@torch.no_grad()
def prediction_postprocessor(
    outputs,
    images: list[Tensor] | Tensor,
    score_threshold: float = 0.05,
    label_offset: int = 0,
) -> list[dict[str, Tensor]]:
    
    if isinstance(outputs, (list, tuple)):
        return outputs 

    
    if hasattr(outputs, "logits"):
        
        if isinstance(images, Tensor): images = list(images)
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        probabilities = logits.sigmoid()
        scores, predicted_labels = probabilities.max(dim=-1)
        boxes = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
        predictions = []
        for image, image_boxes, image_scores, image_labels in zip(images, boxes, scores, predicted_labels):
            height, width = image.shape[-2:]
            scale = image_boxes.new_tensor([width, height, width, height])
            image_boxes = image_boxes * scale
            keep = image_scores >= score_threshold
            predictions.append({
                "boxes": image_boxes[keep],
                "scores": image_scores[keep],
                "labels": image_labels[keep].long() + label_offset
            })
        return predictions
    
    return outputs
