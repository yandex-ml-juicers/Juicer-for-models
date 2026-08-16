"""Обычная кросс-энтропия: обучение ученика с нуля, без учителя.

Нужна, чтобы бейзлайн "ученик сам по себе" запускался тем же тренером
и отличался от дистилляции только конфигом (loss=ce, model/teacher=null).
"""

import torch
import torch.nn.functional as F

from src.losses.base import DistillationLoss


class CrossEntropy(DistillationLoss):
    requires_teacher = False

    def __init__(self, label_smoothing: float = 0.0, ignore_index: int = -100) -> None:
        """ignore_index: метка, исключаемая из лосса. -100 (дефолт torch) — для
        классификации; для сегментации на Cityscapes сюда идёт 255, иначе
        void-пиксели превратились бы в 20-й «класс» и портили обучение.
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        labels: torch.Tensor,
        student_features: dict | None = None,
        teacher_features: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        # Работает и для [B, C] с метками [B], и для [B, C, H, W] с [B, H, W]:
        # cross_entropy сам сворачивает пространственные оси.
        ce = F.cross_entropy(
            student_logits,
            labels,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )
        return {"total": ce, "ce": ce}

