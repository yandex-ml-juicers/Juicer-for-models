"""Базовый интерфейс лоссов дистилляции.

Ключевое решение: лосс — это nn.Module, а не функция. Причины:
- у feature-based дистилляции есть ОБУЧАЕМЫЕ параметры (адаптеры каналов),
  и они принадлежат лоссу, а не ученику; optimizer собирается из
  student.parameters() + criterion.parameters();
- лосс декларирует свои требования (нужен ли учитель, какие слои снимать
  хуками), и Trainer настраивается по этим декларациям, не зная деталей.
"""

import torch
from torch import nn


class DistillationLoss(nn.Module):
    """Контракт: forward принимает всё, что может понадобиться любому лоссу,
    и возвращает dict скаляров, где "total" — то, по чему идёт backward.
    Остальные ключи — компоненты для логирования.
    """

    # Нужен ли этому лоссу учитель (teacher_logits не None).
    requires_teacher: bool = True
    # Имена слоёв, чьи карты признаков должны быть сняты хуками
    # с ученика И учителя. Пустой кортеж — хуки не ставятся вовсе.
    required_features: tuple[str, ...] = ()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        labels: torch.Tensor,
        student_features: dict | None = None,
        teacher_features: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError
