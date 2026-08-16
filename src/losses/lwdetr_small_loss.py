import torch
from torch import Tensor
from torchvision.ops import box_convert

from src.losses.base import DistillationLoss


class LWDETRLoss(DistillationLoss):
    requires_teacher = False
    required_features: tuple[str, ...] = ()

    def __init__(
        self,
        return_all_components: bool = True,
    ) -> None:
        super().__init__()

        self.return_all_components = return_all_components

    def forward(
        self,
        student_outputs,
        teacher_outputs = None,
        labels: list[dict[str, torch.Tensor]] | None = None,
        student_features: dict | None = None,
        teacher_features: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        total = getattr(student_outputs, "loss", None)
        raw_losses = getattr(student_outputs, "loss_dict", None)

        if raw_losses is None:
            raw_losses = {}

        losses = {
            "total": total,
            "ce": raw_losses.get("loss_ce", 0.0),
            "bbox": raw_losses.get("loss_bbox", 0.0),
            "giou": raw_losses.get("loss_giou", 0.0),
        }

        if self.return_all_components:
            for name, value in raw_losses.items():
                if not torch.is_tensor(value):
                    continue

                if name not in losses:
                    losses[name] = value

        return losses