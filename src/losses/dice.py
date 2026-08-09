"""Dice Loss (Milletari et al., 3DV 2016, arXiv:1606.04797) для семантической
сегментации.

Зачем он рядом с кросс-энтропией: CE считает пиксели независимо и потому
пропорциональна площади класса. На Cityscapes road занимает ~35% пикселей,
а traffic light — доли процента, и с точки зрения CE выгоднее вообще не
предсказывать редкие классы. Dice же нормируется на площадь самого класса:
вклад traffic light в лосс не зависит от того, сколько его на кадре. Это
прямо оптимизирует пересечение с разметкой, то есть то же, что меряет mIoU.

Чистый Dice в начале обучения нестабилен (у почти случайной модели знаменатель
близок к нулю), поэтому по умолчанию он идёт слагаемым к CE — стандартная
связка CE + Dice.
"""

import torch
import torch.nn.functional as F

from src.losses.base import DistillationLoss


class DiceLoss(DistillationLoss):
    """total = ce_weight * CE + dice_weight * (1 - mean_c Dice_c).

    Dice считается по классам, с агрегацией по всему батчу сразу (а не
    покадрово и потом усреднением): при покадровом варианте кадр без
    traffic light давал бы для этого класса Dice = 1 из-за smooth, и
    редкие классы получали бы фиктивно хорошие значения.

    ignore_index исключается везде: void-пиксели Cityscapes не входят ни
    в пересечение, ни в объединение.

    Замечание про Mixup: интерполяция лосса в тренере точна для CE, но Dice
    нелинеен по таргету, поэтому для него это приближение. С CutMix (таргет
    точный, lam=1) вопрос не возникает.
    """

    requires_teacher = False

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if smooth <= 0:
            raise ValueError(f"smooth должен быть > 0, получено {smooth}")

        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
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
        ce = F.cross_entropy(
            student_logits,
            labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        # В fp32 независимо от AMP: знаменатель Dice — сумма по сотням тысяч
        # пикселей, в fp16 она переполняется.
        logits = student_logits.float()
        num_classes = logits.shape[1]

        valid = labels != self.ignore_index
        # one_hot не принимает 255 при num_classes=19, поэтому void-пиксели
        # временно подменяются нулём и тут же вычищаются маской.
        targets = F.one_hot(labels.masked_fill(~valid, 0), num_classes)
        targets = targets.permute(0, 3, 1, 2).to(logits.dtype)

        mask = valid.unsqueeze(1)
        probs = logits.softmax(dim=1) * mask
        targets = targets * mask

        # Сумма по батчу и пространству -> по одному числу на класс.
        dims = (0, 2, 3)
        intersection = (probs * targets).sum(dims)
        cardinality = probs.sum(dims) + targets.sum(dims)

        # Класс, которого нет в батче и который модель не предсказала, даёт
        # smooth/smooth = 1 и в лосс не вносит ничего.
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice = 1.0 - dice_per_class.mean()

        total = self.ce_weight * ce + self.dice_weight * dice
        return {"total": total, "ce": ce, "dice": dice}
