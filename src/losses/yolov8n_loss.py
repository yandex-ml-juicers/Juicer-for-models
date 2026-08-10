from types import SimpleNamespace

import torch
from torch import nn

from ultralytics.utils.loss import v8DetectionLoss

from src.losses.base import DistillationLoss


class _YOLOModelProxy(nn.Module):
    def __init__(
        self,
        device: torch.device,
        num_classes: int,
        strides: tuple[int, ...],
        reg_max: int,
        box_gain: float,
        cls_gain: float,
        dfl_gain: float,
    ) -> None:
        super().__init__()

        self.dummy = nn.Parameter(torch.empty(0, device=device), requires_grad=False)
        self.args = SimpleNamespace(box=box_gain, cls=cls_gain, dfl=dfl_gain)

        detect_head = nn.Module()
        detect_head.nc = num_classes
        detect_head.reg_max = reg_max
        detect_head.stride = torch.tensor(
            strides,
            dtype=torch.float32,
            device=device,
        )

        self.model = nn.ModuleList([detect_head])


class YOLOv8Loss(DistillationLoss):
    requires_teacher = False
    required_features: tuple[str, ...] = ()

    def __init__(
        self,
        num_classes: int = 8,
        strides: tuple[int, ...] = (8, 16, 32),
        reg_max: int = 16,
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        dfl_gain: float = 1.5,
        return_all_components: bool = True,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.strides = strides
        self.reg_max = reg_max

        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain

        self.return_all_components = return_all_components

        self.loss_fn = None

    def forward(
        self,
        student_outputs,
        teacher_outputs=None,
        labels: dict[str, torch.Tensor] | None = None,
        student_features: dict | None = None,
        teacher_features: dict | None = None,
    ) -> dict[str, torch.Tensor]:

        if labels is None:
            raise ValueError("YOLOv8Loss requires labels.")

        if self.loss_fn is None:
            if isinstance(student_outputs, dict):
                first_tensor = next(
                    value
                    for value in student_outputs.values()
                    if torch.is_tensor(value)
                )

                device = first_tensor.device

            elif isinstance(student_outputs, (list, tuple)):
                device = student_outputs[0].device
            else:
                device = student_outputs.device

            proxy_model = _YOLOModelProxy(
                device=device,
                num_classes=self.num_classes,
                strides=self.strides,
                reg_max=self.reg_max,
                box_gain=self.box_gain,
                cls_gain=self.cls_gain,
                dfl_gain=self.dfl_gain,
            )

            self.loss_fn = v8DetectionLoss(proxy_model)

        total, raw_losses = self.loss_fn(student_outputs, labels)

        losses = {
            "total": total.sum(),
            "bbox": raw_losses.get("box_loss", 0.0),
            "cls": raw_losses.get("cls_loss", 0.0),
            "dfl": raw_losses.get("dfl_loss", 0.0),
        }

        if self.return_all_components:
            for name, value in raw_losses.items():
                if not torch.is_tensor(value):
                    continue

                if name not in losses:
                    losses[name] = value

        return losses