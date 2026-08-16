"""Focal Loss (Lin et al., ICCV 2017, arXiv:1708.02002) для семантической
сегментации."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from src.losses.base import DistillationLoss


class FocalLoss(DistillationLoss):
    """total = mean_pixels alpha_c * (1 - p_true)^gamma * (-log p_true).

    Та же цель, что у OHEM — сместить обучение на трудные пиксели, — но
    непрерывным весом вместо жёсткого отбора: уверенно предсказанный пиксель
    (p_true близка к 1) глушится множителем (1 - p)^gamma, а ошибочный
    остаётся почти с полным весом. Ничего не выбрасывается, поэтому лосс
    гладкий и не зависит от размера кадра, в отличие от top-K.

    gamma=2 — значение из статьи и рабочий дефолт. gamma=0 превращает
    focal обратно в обычную CE.

    alpha: вес класса. В статье это баланс «объект/фон» для детекции; в
    сегментации 19 классов его либо не задают вовсе (None), либо передают
    список длины num_classes. Скаляр смысла не имеет — он просто умножает
    весь лосс, — поэтому не поддержан.
    """

    requires_teacher = False

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Sequence[float] | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma должна быть >= 0, получено {gamma}")

        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        # Буфер, а не тензор-атрибут: так веса классов уезжают на нужное
        # устройство вместе с .to(device) и попадают в state_dict лосса.
        self.register_buffer(
            "alpha",
            torch.tensor([float(value) for value in alpha]) if alpha is not None else None,
            persistent=alpha is not None,
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        labels: torch.Tensor,
        student_features: dict | None = None,
        teacher_features: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        logits = student_logits.float()
        valid = labels != self.ignore_index
        # one_hot/gather не принимают 255, поэтому void-пиксели временно
        # становятся нулевым классом и вычищаются маской ниже.
        safe_labels = labels.masked_fill(~valid, 0)

        log_probs = F.log_softmax(logits, dim=1)
        log_true = log_probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
        probs_true = log_true.exp()

        focal = -((1.0 - probs_true) ** self.gamma) * log_true
        if self.alpha is not None:
            focal = focal * self.alpha.to(focal.dtype)[safe_labels]

        count = valid.sum().clamp(min=1)
        loss = (focal * valid).sum() / count
        ce = (-log_true * valid).sum() / count

        return {"total": loss, "focal": loss, "ce": ce.detach()}
