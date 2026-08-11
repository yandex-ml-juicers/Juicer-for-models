"""DIST — Knowledge Distillation from A Stronger Teacher
(Huang et al., NeurIPS 2022, arXiv:2205.10536)."""

import torch
import torch.nn.functional as F

from src.losses.base import DistillationLoss
from src.losses.segmentation_utils import align_logits, pearson_correlation, subsample_spatially


class DISTLoss(DistillationLoss):
    """total = ce_weight * CE + T^2 * (beta * inter + gamma * intra).

    Отправная точка статьи: KL требует ТОЧНОГО совпадения вероятностей, и
    чем сильнее учитель, тем это требование вреднее — сильная модель
    уверена и остра, слабый ученик физически не может её повторить, а
    попытка это сделать ломает то, что он умеет. DIST требует совпадения
    не значений, а ПОРЯДКА: сохраняются отношения, а не числа.

    Формализация — корреляция Пирсона (она инвариантна к сдвигу и
    масштабу, то есть штрафует только за перестановку предпочтений):

    - inter (межклассовая): для каждого пикселя корреляция векторов
      вероятностей ученика и учителя по классам. "Ранжируй классы в этой
      точке так же, как учитель".
    - intra (внутриклассовая): для каждого класса корреляция его карт
      вероятностей у ученика и учителя по пикселям. "Расставляй пиксели по
      уверенности в этом классе так же, как учитель".

    Адаптация к сегментации: в статье примером батча выступает картинка,
    здесь — пиксель. inter считается по всем пикселям, intra — внутри
    каждой картинки отдельно (карта вероятностей класса привязана к сцене;
    корреляция поверх нескольких разных сцен смысла не имеет).
    """

    requires_teacher = True

    def __init__(
        self,
        temperature: float = 4.0,
        ce_weight: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        spatial_stride: int = 1,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
    ) -> None:
        """
        Args:
            beta, gamma: веса межклассового и внутриклассового членов.
            spatial_stride: прореживание пикселей перед подсчётом DIST.
                Лосс материализует несколько тензоров размера логитов; на
                кропе 512x1024 это сотни мегабайт на копию. stride=2
                снимает четверть пикселей и режет память вчетверо, статистики
                при этом не меняются.
        """
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature должна быть > 0, получено {temperature}")
        if spatial_stride < 1:
            raise ValueError(f"spatial_stride должен быть >= 1, получено {spatial_stride}")

        self.temperature = float(temperature)
        self.ce_weight = float(ce_weight)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.spatial_stride = int(spatial_stride)
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
        student, teacher = align_logits(student_logits, teacher_logits, "DISTLoss")

        ce = F.cross_entropy(
            student,
            labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        student_sub = subsample_spatially(student, self.spatial_stride)
        teacher_sub = subsample_spatially(teacher, self.spatial_stride)

        temperature = self.temperature
        batch_size, num_classes = student_sub.shape[:2]

        # [B, C, N]: ось 1 — классы, ось 2 — пиксели.
        probs_student = (student_sub / temperature).softmax(dim=1).reshape(batch_size, num_classes, -1)
        probs_teacher = (teacher_sub / temperature).softmax(dim=1).reshape(batch_size, num_classes, -1)

        inter = 1.0 - pearson_correlation(probs_student, probs_teacher, dim=1).mean()
        intra = 1.0 - pearson_correlation(probs_student, probs_teacher, dim=2).mean()

        # T^2 — та же компенсация масштаба градиента, что у Хинтона.
        inter = inter * temperature**2
        intra = intra * temperature**2

        total = self.ce_weight * ce + self.beta * inter + self.gamma * intra
        return {"total": total, "ce": ce, "inter": inter, "intra": intra}
