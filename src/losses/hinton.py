"""Классическая дистилляция Хинтона (Hinton et al., 2015): KL по смягчённым
логитам + кросс-энтропия по жёстким меткам."""

import torch
import torch.nn.functional as F

from src.losses.base import DistillationLoss


class HintonKD(DistillationLoss):
    """total = (1 - alpha) * CE(student, labels) + alpha * T^2 * KL(soft_teacher || soft_student).

    Множитель T^2 компенсирует уменьшение градиентов от смягчения:
    производная KL по логитам масштабируется как 1/T^2 (см. статью, раздел 2).
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.9) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha должна быть в [0, 1], получено {alpha}")
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        labels: torch.Tensor,
        student_features: dict | None = None,
        teacher_features: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        temperature = self.temperature
        ce = F.cross_entropy(student_logits, labels)

        soft_student = F.log_softmax(student_logits / temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits.detach() / temperature, dim=1)
        kd = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * temperature**2

        total = (1.0 - self.alpha) * ce + self.alpha * kd
        return {"total": total, "ce": ce, "kd": kd}
