import copy
import math

import torch
from torch import nn

from src.utils.distributed import unwrap


class ModelEMA:
    """Экспоненциальное скользящее среднее весов студента.

    Официальный RT-DETR (как и ultralytics для YOLO) репортит именно
    EMA-веса. Оценка по мгновенному чекпоинту шумит: на дообучении RT-DETR
    на Cityscapes eval mAP гулял в полосе +-0.03 между соседними эпохами при
    монотонно падающем train loss, то есть шум сопоставим с разницей между
    сравниваемыми рецептами. EMA убирает его, не меняя траекторию обучения —
    градиенты считаются по живой модели, среднее только читается на валидации.

    decay растёт по прогреву tau (рецепт ultralytics): на первых шагах
    среднее почти догоняет модель, иначе случайно инициализированная голова
    держалась бы в нём тысячи шагов.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: int = 2000) -> None:
        self.module = copy.deepcopy(unwrap(model)).eval()
        self.module.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.decay * (1.0 - math.exp(-self.updates / self.tau)) if self.tau else self.decay

        model_state = unwrap(model).state_dict()
        for key, value in self.module.state_dict().items():
            source = model_state[key]
            if value.dtype.is_floating_point:
                value.lerp_(source.detach().to(value.device), 1.0 - decay)
            else:
                # Целочисленные буферы (счётчики шагов и т.п.) не усредняются.
                value.copy_(source)

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state["module"])
        self.updates = int(state.get("updates", 0))
