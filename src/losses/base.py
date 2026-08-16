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
    # Какие ключи forward()-словаря считать независимыми слагаемыми total —
    # для GradientContributionTracker (см. DetectionTrainer). None (по
    # умолчанию) — все ключи, кроме "total", независимы, это верно для
    # большинства лоссов (YOLO: bbox+cls+dfl; DCKD: det+hekld+hokfd;
    # CLoCKDistill: det+lcmd+tcld). Если же словарь вперемешку содержит и
    # слагаемые total, И детальную раскладку ОДНОГО из них для логов
    # (например KDDETRLoss: total = det + distill, а distill сам разложен на
    # cls/l1/giou), то без этого поля трекер посчитал бы distill дважды —
    # один раз целиком, второй раз размазанным по cls/l1/giou. Переопредели
    # в подклассе кортежем только тех ключей, что реально суммируются в total.
    gradient_probe_keys: tuple[str, ...] | None = None

    @property
    def gradient_probe_weights(self) -> dict[str, float]:
        """λ, с которым каждый ключ gradient_probe_keys реально входит в
        total = Σ λ_i · L_i — для взвешенного зонда градиента
        (GradientContributionTracker.probe(..., weights=...) в
        DetectionTrainer). Ключ, которого здесь нет, получает вес 1.0: это
        верно для компонентов, что складываются в total без явного
        домножения (например bbox/cls/dfl у YOLO — они уже взвешены gain'ами
        ultralytics до того, как forward() их вернул).

        Свойство, а не статический класс-атрибут: λ обычно живут как
        self.lambda_* — конфигурируемые атрибуты конкретного экземпляра
        (значения из cfg.loss), а не константы класса. Подклассы с
        весами total, отличными от 1.0 (DCKD, CLoCKDistill, KDDETRLoss),
        обязаны переопределить это свойство — иначе взвешенный зонд молча
        деградирует до сырого (той же величины, что и до фикса "не работает
        учёт λ").
        """
        return {}

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        labels: torch.Tensor,
        student_features: dict | None = None,
        teacher_features: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError
