"""Сумма нескольких лоссов под одним интерфейсом DistillationLoss.

Тренер знает ровно один criterion, поэтому «CE + FitNets + Dice» собирается
не в тренере, а здесь: CompositeLoss держит слагаемые в ModuleDict, зовёт
каждое на одних и тех же логитах и признаках и складывает их total'ы
с весами из конфига.

Три следствия, ради которых это работает без правок остального кода:
- обучаемые параметры слагаемых (регрессор FitNets, проекторы HeteroAKD)
  видны через criterion.parameters() и попадают в optimizer;
- требования к тренеру объединяются: учитель нужен, если он нужен хотя бы
  одному слагаемому; хуки ставятся на объединение required_features;
- компоненты для логов не перетираются — ключи слагаемых уходят в history.csv
  с префиксом имени ("kd_hint", "seg_dice"), а их собственные total'ы —
  под самим именем ("kd", "seg").

ВАЖНО про двойной счёт CE. Почти каждый лосс проекта уже содержит своё
слагаемое CE (ce_weight). Собирая композицию, оставьте CE ровно в одном
месте, а у остальных поставьте ce_weight: 0 — иначе кросс-энтропия войдёт
в сумму дважды с непонятным итоговым весом.
"""

from collections.abc import Mapping

import torch
from torch import nn

from src.losses.base import DistillationLoss


class CompositeLoss(DistillationLoss):
    """total = sum_i weight_i * loss_i.total.

    Args:
        losses: {имя: лосс}. Имя попадает в ключи логов, поэтому лучше короткое
            и говорящее ("kd", "seg", "boundary").
        weights: {имя: вес}. Не перечисленные слагаемые идут с весом 1.0.
            Веса живут здесь, а не только внутри слагаемых, чтобы их можно
            было менять по ходу обучения планировщиком
            (src/training/loss_schedule.py, путь "weights.<имя>").
    """

    def __init__(
        self,
        losses: Mapping[str, DistillationLoss],
        weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        if not losses:
            raise ValueError("CompositeLoss требует хотя бы одно слагаемое в losses")

        weights = dict(weights or {})
        unknown = set(weights) - set(losses)
        if unknown:
            raise ValueError(
                f"В weights есть имена, которых нет в losses: {sorted(unknown)}"
            )

        self.losses = nn.ModuleDict(dict(losses))
        self.weights = {name: float(weights.get(name, 1.0)) for name in self.losses}

        self.requires_teacher = any(loss.requires_teacher for loss in self.losses.values())

        # Порядок тапов стабилен (первое появление), дубликаты убраны: хуки
        # ставятся по одному на слой, даже если его просят два слагаемых.
        required: list[str] = []
        for loss in self.losses.values():
            for name in loss.required_features:
                if name not in required:
                    required.append(name)
        self.required_features = tuple(required)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        labels: torch.Tensor,
        student_features: dict | None = None,
        teacher_features: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None

        for name, loss in self.losses.items():
            parts = loss(
                student_logits,
                teacher_logits,
                labels,
                student_features=student_features,
                teacher_features=teacher_features,
            )
            weighted = self.weights[name] * parts["total"]
            total = weighted if total is None else total + weighted

            result[name] = parts["total"]
            result.update(
                {f"{name}_{key}": value for key, value in parts.items() if key != "total"}
            )

        result["total"] = total
        return result
