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


class YOLO(DistillationLoss):
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
            raise ValueError("YOLO requires labels.")

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

        # self.loss_fn(student_outputs, labels) (= v8DetectionLoss.__call__)
        # отдаёт total суммой по box/cls/dfl и ОТДЕЛЬНО detach()-нутый dict
        # исключительно для логов — внутри get_assigned_targets_and_loss
        # ultralytics прямо возвращает "loss, dict(zip(names, loss.detach()))",
        # т.е. разбивку по компонентам даёт только оторванной от графа.
        # Для градиентных зондов (GradientContributionTracker) это не годится:
        # torch.autograd.grad на detach()-нутом тензоре падает с "does not
        # require grad". Поэтому box/cls/dfl берём сами через тот же
        # internal-метод, которым пользуется сама библиотека —
        # get_assigned_targets_and_loss отдаёт [box, cls, dfl] ДО суммирования
        # и БЕЗ detach, то есть то же вычисление, но с живым графом.
        parsed = self.loss_fn.parse_output(student_outputs)
        batch_size = parsed["boxes"].shape[0]
        _, loss_vec, _ = self.loss_fn.get_assigned_targets_and_loss(parsed, labels)
        scaled = loss_vec * batch_size
        by_name = dict(zip(self.loss_fn.loss_names, scaled))

        losses = {
            "total": scaled.sum(),
            "bbox": by_name.get("box_loss", scaled.new_zeros(())),
            "cls": by_name.get("cls_loss", scaled.new_zeros(())),
            "dfl": by_name.get("dfl_loss", scaled.new_zeros(())),
        }

        if self.return_all_components:
            for name, value in by_name.items():
                if name not in losses:
                    losses[name] = value

        return losses