"""OHEM Cross-Entropy — online hard example mining на уровне пикселей
(Shrivastava et al., CVPR 2016, arXiv:1604.03540; в сегментации — стандартный
рецепт PSPNet/OCRNet и mmsegmentation).
"""

import math

import torch
import torch.nn.functional as F

from src.losses.base import DistillationLoss


class OhemCrossEntropy(DistillationLoss):
    """CE, усреднённая только по «трудным» пикселям кадра.

    Зачем. Обычная CE усредняется по всем пикселям, а на Cityscapes подавляющее
    их большинство — уверенно предсказанная середина дороги, неба и зданий.
    Их вклад в градиент почти нулевой по величине, но они разбавляют среднее,
    и шаг оптимизатора определяется в основном лёгкими пикселями. OHEM просто
    выбрасывает их из среднего.

    Отбор — по величине per-pixel лосса, на КАЖДОМ кадре отдельно (иначе один
    трудный кадр батча забрал бы всю квоту, а на остальных не осталось бы
    ни одного пикселя):

        порог = min(k-й по величине лосс, -log(thresh)),   k = keep_ratio * H*W

    То есть берутся все пиксели, где вероятность верного класса упала ниже
    thresh, но не меньше keep_ratio доли самых трудных. Второе слагаемое —
    защита от вырождения: на хорошо обученной модели пикселей ниже порога
    может не остаться вовсе, и лосс стал бы нулевым.

    ignore_index исключается: cross_entropy обнуляет такие пиксели, а порог
    строго положителен (thresh < 1), поэтому нули в отбор не попадают.
    """

    requires_teacher = False

    def __init__(
        self,
        thresh: float | None = 0.7,
        keep_ratio: float = 0.1,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
    ) -> None:
        """
        Args:
            thresh: вероятность верного класса, ниже которой пиксель считается
                трудным. None — отбирать только по keep_ratio.
            keep_ratio: минимальная доля пикселей кадра, которая остаётся
                в лоссе в любом случае. 0.1 — рабочий дефолт mmsegmentation
                (там это min_kept=100000 на кроп 512x1024, то есть те же ~19%).
        """
        super().__init__()
        if thresh is not None and not 0.0 < thresh < 1.0:
            raise ValueError(f"thresh должен быть в (0, 1), получено {thresh}")
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio должен быть в (0, 1], получено {keep_ratio}")

        self.thresh = float(thresh) if thresh is not None else None
        self.keep_ratio = float(keep_ratio)
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
        per_pixel = F.cross_entropy(
            student_logits.float(),
            labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        valid = labels != self.ignore_index
        batch_size = per_pixel.shape[0]
        flat = per_pixel.reshape(batch_size, -1)

        keep = max(1, int(self.keep_ratio * flat.shape[1]))
        # kthvalue по убыванию = topk, но нужен только сам порог, а не индексы.
        threshold = flat.topk(keep, dim=1).values[:, -1]
        if self.thresh is not None:
            threshold = threshold.clamp(max=-math.log(self.thresh))

        selected = (flat >= threshold[:, None]) & valid.reshape(batch_size, -1)
        kept = selected.sum().clamp(min=1)
        ohem = (flat * selected).sum() / kept

        # Обычная CE рядом — чтобы прогон оставался сравним с остальными
        # по одной и той же величине; в backward она не идёт.
        ce = per_pixel.sum() / valid.sum().clamp(min=1)

        return {"total": ohem, "ohem": ohem, "ce": ce.detach()}
