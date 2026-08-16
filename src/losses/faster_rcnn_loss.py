import torch
from src.losses.base import DistillationLoss

class FasterRCNNLoss(DistillationLoss):
    requires_teacher = False
    required_features = ()

    def forward(self, student_outputs, teacher_outputs, labels, **kwargs):
        if not isinstance(student_outputs, dict):
            return {"total": torch.tensor(0.0, device=labels[0]['boxes'].device)}

        total_loss = sum(loss for loss in student_outputs.values())
        
        
        res = {"total": total_loss}
        res.update(student_outputs)
        return res
