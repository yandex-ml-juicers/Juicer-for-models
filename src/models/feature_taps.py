"""Именованные точки съёма промежуточных признаков.

Зачем это нужно. FeatureExtractor ставит хуки по ИМЕНАМ подмодулей, а Trainer
использует один и тот же список имён и для ученика, и для учителя:

    layers = list(criterion.required_features)
    self.student_extractor = FeatureExtractor(self.student, layers)
    self.teacher_extractor = FeatureExtractor(self.teacher, layers)

Для однородной пары (ResNet -> ResNet) это работает: слои называются одинаково.
Для дистилляции ViT -> CNN (SegFormer -> SegNeXt) внутренние имена разные,
и общего имени слоя просто не существует.

Решение: обе модели-обёртки прогоняют свои стадии через FeatureTaps —
ModuleDict из nn.Identity с КАНОНИЧЕСКИМИ именами. Хук вешается на Identity,
а не на архитектурно-специфичный модуль, поэтому `taps.stage3` означает
"признаки на страйде 16" в любой модели, которая объявила такой тап.

Identity не имеет параметров и не влияет ни на forward, ни на state_dict
(в state_dict у ModuleDict из Identity нет ни одного ключа), поэтому
механизм ничего не ломает в уже обученных чекпоинтах.
"""

from collections.abc import Iterable

import torch
from torch import nn

# Канонические имена стадий энкодера. Цифра — номер стадии, страйд признаков
# относительно входа: stage1 -> 1/4, stage2 -> 1/8, stage3 -> 1/16, stage4 -> 1/32.
# Такая раскладка одинакова у MiT (SegFormer) и MSCAN (SegNeXt).
STAGE_TAPS: tuple[str, ...] = ("stage1", "stage2", "stage3", "stage4")


class FeatureTaps(nn.ModuleDict):
    """ModuleDict из nn.Identity: точки, к которым можно привязать forward-хук."""

    def __init__(self, names: Iterable[str] = STAGE_TAPS) -> None:
        super().__init__({name: nn.Identity() for name in names})

    def tap(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Пропускает тензор через тап `name` (возвращает его же).

        Вызов обязателен именно как проход через модуль: хук срабатывает
        на forward'е Identity, а не на присваивании.
        """
        return self[name](tensor)
