"""Ванильная дистилляция Хинтона (arXiv:1503.02531), применённая попиксельно.

Базовый метод сравнения для всех остальных: каждый пиксель карты логитов
трактуется как самостоятельный пример классификации, и по нему считается
та же KL по смягчённым распределениям, что и в исходной статье.
"""

import torch
import torch.nn.functional as F

from src.losses.base import DistillationLoss
from src.losses.segmentation_utils import align_logits


class PixelWiseKD(DistillationLoss):
    """total = (1 - alpha) * CE + alpha * T^2 * mean_pixels KL(p_teacher || p_student).

    Почему отдельный класс, а не HintonKD на картах:
    - HintonKD зовёт kl_div с reduction="batchmean", то есть делит сумму
      на размер батча. Для [B, C, H, W] это оставляет в лоссе множитель
      H*W (полмиллиона на кропе 512x1024) — градиент улетает на порядки;
    - его CE не знает про ignore_index, и void-пиксели Cityscapes попали бы
      в обучение как настоящий класс.

    Здесь KL усредняется по пикселям, а CE выбрасывает ignore_index.
    Множитель T^2 — как в статье: он компенсирует то, что градиент
    смягчённого softmax масштабируется как 1/T^2.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha должна быть в [0, 1], получено {alpha}")
        if temperature <= 0.0:
            raise ValueError(f"temperature должна быть > 0, получено {temperature}")

        self.temperature = float(temperature)
        self.alpha = float(alpha)
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
        student, teacher = align_logits(student_logits, teacher_logits, "PixelWiseKD")

        ce = F.cross_entropy(
            student,
            labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        temperature = self.temperature
        log_student = F.log_softmax(student / temperature, dim=1)
        log_teacher = F.log_softmax(teacher / temperature, dim=1)

        # KL(p_t || p_s) = sum_c p_t * (log p_t - log p_s): сумма по классам,
        # среднее по всем пикселям батча.
        per_pixel_kl = (log_teacher.exp() * (log_teacher - log_student)).sum(dim=1)
        kd = per_pixel_kl.mean() * temperature**2

        total = (1.0 - self.alpha) * ce + self.alpha * kd
        return {"total": total, "ce": ce, "kd": kd}
