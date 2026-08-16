"""Расписание весов лосса: менять вклад слагаемых по ходу обучения.

Зачем. Дистилляционный член полезен в начале, когда ученик ещё ничего не
умеет и учитель для него — источник знаний, и вреден в конце, когда ученик
уже близок к своему потолку, а учитель тянет его к СВОИМ ошибкам (это в
литературе зовут "teacher-student gap"; на нём же построен early-stopped KD).
Симметричный случай — прогрев: hint-лосс FitNets имеет смысл придавить на
первых эпохах, пока регрессор внутри лосса сам не обучился.

Как это работает. Все лоссы проекта хранят веса слагаемых обычными float-
атрибутами (ce_weight, hint_weight, cwd_weight ...) и читают их в forward,
поэтому расписание — это просто присваивание перед каждой эпохой. Никакой
поддержки со стороны самих лоссов не требуется.

Конфиг (значения — доли от исходного веса? нет, абсолютные значения):

    loss_schedule:
      hint_weight:  {schedule: linear, start: 0.7, end: 0.0, start_epoch: 100}
      ce_weight:    {schedule: linear, start: 0.3, end: 1.0, start_epoch: 100}

Путь может быть вложенным — так достают слагаемые CompositeLoss:

    loss_schedule:
      weights.kd:          {schedule: cosine, start: 1.0, end: 0.0}
      losses.kd.ce_weight: {schedule: constant, start: 0.0}
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from torch import nn


def _resolve(container: object, name: str) -> object:
    """Шаг по пути: атрибут или элемент словаря/ModuleDict."""
    if isinstance(container, Mapping):
        return container[name]
    if isinstance(container, nn.ModuleDict):
        return container[name]
    return getattr(container, name)


@dataclass(frozen=True)
class WeightSchedule:
    """Одно правило: как меняется вес по пути `path` от эпохи к эпохе.

    Между start_epoch и end_epoch вес идёт от start к end, вне этого отрезка
    держится на границе. end_epoch=None означает "до конца обучения" и
    подставляется числом эпох прогона.

    schedule:
        constant — всегда start (способ выключить слагаемое, не трогая конфиг
            лосса, и способ задать вес одним числом);
        linear   — равномерно;
        cosine   — плавно на концах, быстро в середине; тот же профиль, что
            у косинусного расписания learning rate, и по той же причине:
            резкое переключение веса в середине обучения даёт скачок лосса.
    """

    path: str
    schedule: str = "linear"
    start: float = 1.0
    end: float = 0.0
    start_epoch: int = 1
    end_epoch: int | None = None

    def value_at(self, epoch: int, total_epochs: int) -> float:
        if self.schedule == "constant":
            return self.start

        last = self.end_epoch if self.end_epoch is not None else total_epochs
        if epoch <= self.start_epoch:
            return self.start
        if epoch >= last:
            return self.end

        progress = (epoch - self.start_epoch) / (last - self.start_epoch)
        if self.schedule == "linear":
            factor = progress
        elif self.schedule == "cosine":
            factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        else:
            raise ValueError(
                f"Неизвестное расписание {self.schedule!r}; доступны: constant, linear, cosine"
            )

        return self.start + (self.end - self.start) * factor


class LossWeightScheduler:
    """Применяет набор WeightSchedule к criterion перед каждой эпохой.

    Пути проверяются в конструкторе, а не при первом применении: опечатка
    в имени веса иначе всплыла бы через час обучения, причём молча — лосс
    просто продолжил бы считаться со старым весом.
    """

    def __init__(
        self,
        criterion: nn.Module,
        schedules: Mapping[str, Mapping],
        total_epochs: int,
    ) -> None:
        self.criterion = criterion
        self.total_epochs = int(total_epochs)
        self.schedules = [
            WeightSchedule(path=path, **dict(spec)) for path, spec in schedules.items()
        ]

        for schedule in self.schedules:
            container, name = self._locate(schedule.path)
            current = _resolve(container, name)
            if not isinstance(current, (int, float)):
                raise TypeError(
                    f"loss_schedule: {schedule.path!r} — это {type(current).__name__}, "
                    f"а расписание умеет менять только числовые веса"
                )

    def _locate(self, path: str) -> tuple[object, str]:
        """(контейнер, имя последнего элемента пути)."""
        *prefix, name = path.split(".")
        container: object = self.criterion
        for step in prefix:
            container = _resolve(container, step)

        try:
            _resolve(container, name)
        except (AttributeError, KeyError) as error:
            raise ValueError(
                f"loss_schedule: у лосса {type(self.criterion).__name__} нет веса {path!r}. "
                f"Путь — это цепочка атрибутов лосса, например 'hint_weight' "
                f"или 'weights.kd' для CompositeLoss."
            ) from error

        return container, name

    def step(self, epoch: int) -> dict[str, float]:
        """Выставляет веса на эпоху и возвращает их — для логов."""
        values: dict[str, float] = {}

        for schedule in self.schedules:
            container, name = self._locate(schedule.path)
            value = schedule.value_at(epoch, self.total_epochs)

            if isinstance(container, Mapping):
                container[name] = value
            else:
                setattr(container, name, value)

            values[f"loss_weight_{schedule.path.replace('.', '_')}"] = value

        return values
