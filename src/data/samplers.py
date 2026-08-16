"""Сэмплеры для распределённого запуска."""

from collections.abc import Iterator, Sized

from torch.utils.data import Sampler


class ShardSampler(Sampler[int]):
    """Режет выборку на непересекающиеся куски без дополнения дубликатами.

    Ранки могут получать разное число объектов

    ВАЖНОЕ ОГРАНИЧЕНИЕ: из-за разной длины шардов ранки делают разное число
    итераций. Значит внутри цикла по такому загрузчику НЕ должно быть
    коллективных операций — иначе ранки разойдутся в их числе и повиснут.
    Синхронизировать можно только после цикла.
    """

    def __init__(self, dataset: Sized, num_replicas: int = 1, rank: int = 0) -> None:
        self.total_size = len(dataset)
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.total_size, self.num_replicas))

    def __len__(self) -> int:
        return max(0, (self.total_size - self.rank + self.num_replicas - 1) // self.num_replicas)
