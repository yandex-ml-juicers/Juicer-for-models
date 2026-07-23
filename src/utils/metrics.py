"""Метрики и агрегаторы."""

import pandas as pd
import torch
import torch.nn as nn

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Доля правильных ответов по argmax логитов, в [0, 1]."""
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()

def count_parameters(model: nn.Module) -> dict:
    '''Считает колиечество параметров у модели
    '''
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())

    params_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers_size = sum(b.numel() * b.element_size() for b in model.buffers())

    return {
        "total_M": total / 1e6,
        "trainable_M": trainable / 1e6,
        "buffers_M": buffers / 1e6,
        "size_MiB": (params_size + buffers_size) / (1024**2),
    }

def build_param_table(student=None, teacher=None, criterion=None) -> pd.DataFrame | None:
    cols = ["total_k", "trainable_k", "buffers_k", "size_MiB"]
    rows = {}
    if student is not None:
        rows["Student"] = count_parameters(student)
    if criterion is not None:
        rows["Criterion"] = count_parameters(criterion)
    if teacher is not None:
        rows["Teacher"] = count_parameters(teacher)
    if rows is {}:
        return None
    df = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    df.index.name = "module"
    return df.reset_index()

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
        """precision/recall/F1 (macro), выведенные из накопленной матрицы
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
            "F1": f1[present].mean().item(),
        }
