import torch
import torch.nn.functional as F
from torch import Tensor

from src.losses.base import DistillationLoss


class AIDTeacherAdaptationLoss(DistillationLoss):
    """
    В текущем Trainer:
      student_logits = logits обучаемого большого учителя;
      teacher_logits = logits замороженного малого ученика.
    """

    requires_teacher = True
    required_features: tuple[str, ...] = ()

    def __init__(
        self,
        temperature: float = 4.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature должна быть > 0")

        if beta < 0:
            raise ValueError("beta должна быть >= 0")

        self.temperature = temperature
        self.beta = beta

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Tensor,
        *,
        student_features=None,
        teacher_features=None,
    ) -> dict[str, Tensor]:
        # В этом режиме:
        # student_logits = обучаемый TinyViT
        # teacher_logits = замороженный ShuffleNet

        adapted_teacher_logits = student_logits.float()
        frozen_student_logits = teacher_logits.detach().float()

        ce = F.cross_entropy(adapted_teacher_logits,labels)

        temperature = self.temperature

        log_p_adapted_teacher = F.log_softmax(adapted_teacher_logits / temperature, dim=1)
        p_adapted_teacher = log_p_adapted_teacher.exp()

        log_p_frozen_student = F.log_softmax(frozen_student_logits / temperature, dim=1)

        # KL(
        #   adapted TinyViT
        #   ||
        #   frozen ShuffleNet
        # )
        kd = (p_adapted_teacher * (log_p_adapted_teacher - log_p_frozen_student)).sum(dim=1).mean()

        total = ce + self.beta * kd

        return {"total": total, "ce": ce, "kd": kd}