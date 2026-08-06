"""CWD — Channel-wise Knowledge Distillation (Shu et al., ICCV 2021,
arXiv:2011.13256)."""

import torch
import torch.nn.functional as F

from src.losses.base import DistillationLoss
from src.losses.segmentation_utils import align_logits


class ChannelWiseKD(DistillationLoss):
    """total = ce_weight * CE + cwd_weight * T^2/C * sum_c KL(phi(t_c) || phi(s_c)).

    Разница с попиксельной KD — в том, вдоль какой оси берётся softmax.
    Попиксельная KD нормирует по КЛАССАМ внутри пикселя и требует, чтобы
    ученик угадал распределение в каждой точке. CWD нормирует по
    ПРОСТРАНСТВУ внутри канала: phi(y_c) — это распределение активации
    c-го класса по всем пикселям, то есть карта "где именно учитель видит
    этот класс".

    Практический смысл: карты активаций сегментатора сильно разрежены —
    несколько классов доминируют по площади, остальные занимают проценты
    пикселей. Попиксельная KD тонет в фоне, а канальная нормировка ставит
    редкие классы в равные условия с дорогой, потому что каждый канал
    нормируется сам по себе. Отсюда и выигрыш метода на плотных задачах.

    Нормировка на число каналов C оставлена как в статье, чтобы величина
    лосса не зависела от количества классов.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        ce_weight: float = 1.0,
        cwd_weight: float = 3.0,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature должна быть > 0, получено {temperature}")

        self.temperature = float(temperature)
        self.ce_weight = float(ce_weight)
        self.cwd_weight = float(cwd_weight)
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        labels: torch.Tensor,
        student_features: dict | None = None,
        teacher_features: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        student, teacher = align_logits(student_logits, teacher_logits, "ChannelWiseKD")

        ce = F.cross_entropy(
            student,
            labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        batch_size, num_channels = student.shape[:2]
        temperature = self.temperature

        # Softmax по последней оси = по всем пикселям канала.
        student_flat = (student.reshape(batch_size, num_channels, -1) / temperature)
        teacher_flat = (teacher.reshape(batch_size, num_channels, -1) / temperature)

        log_student = F.log_softmax(student_flat, dim=2)
        log_teacher = F.log_softmax(teacher_flat, dim=2)

        # Сумма по пикселям внутри канала, среднее по каналам и по батчу.
        per_channel_kl = (log_teacher.exp() * (log_teacher - log_student)).sum(dim=2)
        cwd = per_channel_kl.mean() * temperature**2

        total = self.ce_weight * ce + self.cwd_weight * cwd
        return {"total": total, "ce": ce, "cwd": cwd}
