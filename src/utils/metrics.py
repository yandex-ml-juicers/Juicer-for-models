"""Метрики и агрегаторы."""

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Доля правильных ответов по argmax логитов, в [0, 1]."""
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


class AverageMeter:
    """Взвешенное скользящее среднее (среднее по всем объектам, не по батчам).

    Усреднение по батчам смещает метрику, если последний батч неполный;
    поэтому update() принимает размер батча как вес.
    """

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)
