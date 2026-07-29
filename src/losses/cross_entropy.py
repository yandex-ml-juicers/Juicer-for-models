"""Обычная кросс-энтропия: обучение ученика с нуля, без учителя.

Нужна, чтобы бейзлайн "ученик сам по себе" запускался тем же тренером
и отличался от дистилляции только конфигом (loss=ce, model/teacher=null).
"""

import torch
import torch.nn.functional as F

from src.losses.base import DistillationLoss


class CrossEntropy(DistillationLoss):
    requires_teacher = False

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        labels: torch.Tensor,
        student_features: dict | None = None,
        teacher_features: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        ce = F.cross_entropy(student_logits, labels, label_smoothing=self.label_smoothing)
        return {"total": ce, "ce": ce}
