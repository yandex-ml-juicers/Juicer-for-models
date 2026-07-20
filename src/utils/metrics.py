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
        return self.sum / self.count if self.count != 0 else self.sum

class ConfusionMatrixAccumulator:
    """Копит confusion matrix по батчам без хранения сырых предсказаний.
    Память — O(num_classes^2), не зависит от размера датасета/числа батчей.
    """

    def __init__(self, num_classes: int, device: torch.device) -> None:
        self.num_classes = num_classes
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """preds, labels — 1D LongTensor одинаковой длины (индексы классов).
        cm[i, j] = число примеров с истинным классом i, предсказанных как j.
        """
        preds = preds.long().to(self.matrix.device)
        labels = labels.long().to(self.matrix.device)
        indices = labels * self.num_classes + preds
        batch_counts = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.matrix += batch_counts.reshape(self.num_classes, self.num_classes)

    def reset(self) -> None:
        self.matrix.zero_()

    def compute(self) -> dict[str, float]:
        """precision/recall/f1 (macro), выведенные из накопленной матрицы
        """
        cm = self.matrix.float()
        tp = cm.diagonal()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp

        precision = tp / (tp + fp).clamp_min(1e-12)
        recall = tp / (tp + fn).clamp_min(1e-12)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

        # Усредняем только по классам, реально встретившимся в эпохе
        support = cm.sum(dim=1)
        present = support > 0

        return {
            "precision": precision[present].mean().item(),
            "recall": recall[present].mean().item(),
            "f1": f1[present].mean().item(),
        }
