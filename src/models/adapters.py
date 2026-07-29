"""Адаптеры карт признаков ученика к размерностям учителя."""

from collections.abc import Mapping

from torch import nn


class ChannelAdapters(nn.Module):
    """1x1-свёртки, приводящие число каналов ученика к числу каналов учителя.

    Это ОБУЧАЕМЫЕ параметры: модуль живёт внутри лосса (FeatureKD), и его
    параметры попадают в optimizer через criterion.parameters(). Веса
    адаптеров — часть чекпоинта эксперимента, но не часть модели ученика.
    """

    def __init__(self, channels: Mapping[str, tuple[int, int]]) -> None:
        """
        Args:
            channels: {имя_слоя: (каналы_ученика, каналы_учителя)}.
        """
        super().__init__()
        self.adapters = nn.ModuleDict(
            {
                layer_name: nn.Conv2d(int(student_ch), int(teacher_ch), kernel_size=1, bias=False)
                for layer_name, (student_ch, teacher_ch) in channels.items()
            }
        )

    def forward(self, features: Mapping) -> dict:
        return {layer_name: self.adapters[layer_name](features[layer_name]) for layer_name in self.adapters}
