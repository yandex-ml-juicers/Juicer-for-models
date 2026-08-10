"""Lovász-Softmax (Berman et al., CVPR 2018, arXiv:1705.08790) — выпуклая
поверхность потерь, напрямую приближающая Jaccard index (то есть IoU).
"""

import torch
import torch.nn.functional as F

from src.losses.base import DistillationLoss
from src.losses.segmentation_utils import subsample_spatially


def lovasz_grad(sorted_labels: torch.Tensor) -> torch.Tensor:
    """Градиент расширения Ловаса для отсортированных ошибок одного класса.

    sorted_labels — индикатор принадлежности классу [N], переставленный в
    порядке убывания ошибки. Возвращает веса, скалярное произведение которых
    с отсортированными ошибками и есть выпуклое расширение (1 - Jaccard).
    """
    pixels = sorted_labels.numel()
    positives = sorted_labels.sum()

    intersection = positives - sorted_labels.cumsum(0)
    union = positives + (1.0 - sorted_labels).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1e-6)

    if pixels > 1:
        # Разности, а не сами значения: расширение Ловаса — это сумма
        # приращений, взвешенных отсортированными ошибками.
        jaccard[1:] = jaccard[1:] - jaccard[:-1]

    return jaccard


class LovaszSoftmax(DistillationLoss):
    """total = ce_weight * CE + lovasz_weight * Lovasz-Softmax.

    Чем это лучше Dice и CE. CE оптимизирует правильность каждого пикселя
    по отдельности, Dice — гладкую аппроксимацию F1, а Lovász — точное
    выпуклое расширение (1 - IoU) на непрерывные предсказания. То есть это
    единственный из трёх лоссов, который минимизирует ровно ту величину,
    которой потом меряется качество (mIoU), а не её суррогат.

    Классы считаются в режиме "present": в лосс входят только классы, реально
    присутствующие в батче. Отсутствующий класс дал бы Jaccard = 0/0, а
    попытка «предсказывать его пореже» — это уже не про IoU.

    Чистый Lovász с нуля обучается плохо (у него нет градиента там, где
    предсказание совсем случайно), поэтому по умолчанию он идёт слагаемым
    к CE — так его и применяют в статье.

    pixel_stride: прореживание пикселей перед подсчётом. Lovász требует
    СОРТИРОВКИ всех пикселей по каждому классу: на батче 16x512x1024 это
    19 сортировок по 8.4M элементов каждый шаг. Прореживание вдвое по каждой
    оси ускоряет это вчетверо, а оценка Jaccard по 2M пикселей от этого не
    меняется.
    """

    requires_teacher = False

    def __init__(
        self,
        ce_weight: float = 1.0,
        lovasz_weight: float = 1.0,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
        pixel_stride: int = 2,
    ) -> None:
        super().__init__()
        if pixel_stride < 1:
            raise ValueError(f"pixel_stride должен быть >= 1, получено {pixel_stride}")

        self.ce_weight = float(ce_weight)
        self.lovasz_weight = float(lovasz_weight)
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)
        self.pixel_stride = int(pixel_stride)

    def _lovasz(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = subsample_spatially(logits, self.pixel_stride).softmax(dim=1)
        labels = subsample_spatially(labels, self.pixel_stride)

        num_classes = probs.shape[1]
        valid = labels != self.ignore_index
        # [B, C, H, W] -> [N, C], [B, H, W] -> [N]; батч сворачивается целиком:
        # IoU в метрике тоже считается по всей выборке, а не покадрово.
        probs = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)[valid.reshape(-1)]
        labels = labels.reshape(-1)[valid.reshape(-1)]

        if labels.numel() == 0:
            return probs.sum() * 0.0

        losses = []
        for class_index in labels.unique().tolist():
            target = (labels == class_index).to(probs.dtype)
            errors = (target - probs[:, class_index]).abs()
            errors_sorted, permutation = torch.sort(errors, descending=True)
            losses.append(torch.dot(errors_sorted, lovasz_grad(target[permutation])))

        return torch.stack(losses).mean()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        labels: torch.Tensor,
        student_features: dict | None = None,
        teacher_features: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        ce = F.cross_entropy(
            student_logits,
            labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        # В fp32 независимо от AMP: сортировка и кумулятивные суммы по
        # миллионам элементов в fp16 теряют хвост распределения ошибок.
        lovasz = self._lovasz(student_logits.float(), labels)

        total = self.ce_weight * ce + self.lovasz_weight * lovasz
        return {"total": total, "ce": ce, "lovasz": lovasz}
