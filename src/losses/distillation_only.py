"""Чистая дистилляция без разметки — прогрев ученика распределением учителя
до основного обучения (см. configs/experiment/segmentation/pretrain/).

Отдельный класс, а не PixelWiseKD/HintonKD с alpha=1: оба всегда считают
CE(student, labels), даже когда её вес 0 — а вес не спасает, если ВСЕ
пиксели батча ignore_index (ровно так устроена official-заглушка разметки
Cityscapes test, см. docs/CityScapes.md): F.cross_entropy на пустом
множестве валидных пикселей возвращает NaN, а 0*NaN=NaN отравляет total.
Здесь labels не используется вовсе.
"""

import torch

from src.losses.base import DistillationLoss
from src.losses.segmentation_utils import align_logits


class DistillationOnlyLoss(DistillationLoss):
    """total = T^2 * mean_pixels KL(p_teacher || p_student). labels игнорируются."""

    def __init__(self, temperature: float = 4.0) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature должна быть > 0, получено {temperature}")
        self.temperature = float(temperature)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        labels: torch.Tensor,
        student_features: dict | None = None,
        teacher_features: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        student, teacher = align_logits(student_logits, teacher_logits, "DistillationOnlyLoss")

        temperature = self.temperature
        log_student = torch.log_softmax(student / temperature, dim=1)
        log_teacher = torch.log_softmax(teacher / temperature, dim=1)

        per_pixel_kl = (log_teacher.exp() * (log_teacher - log_student)).sum(dim=1)
        total = per_pixel_kl.mean() * temperature**2
        return {"total": total}
