"""Метрики и агрегаторы.

Все накопители здесь хранят АДДИТИВНЫЕ величины -- суммы и счётчики (а не
готовые средние).
"""

from collections.abc import Mapping

import pandas as pd
import torch
import torch.nn as nn

from src.utils.distributed import all_reduce_sum_

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Доля правильных ответов по argmax логитов, в [0, 1]."""
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()

def count_parameters(model: nn.Module) -> dict:
    """Считает колиечество параметров у модели"""
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
    rows = {}
    if student is not None:
        rows["Student"] = count_parameters(student)
    if criterion is not None:
        rows["Criterion"] = count_parameters(criterion)
    if teacher is not None:
        rows["Teacher"] = count_parameters(teacher)
    if rows is {}:
        return None
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "module"
    df = df.reset_index()
    return df

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


def sync_meters(meters: Mapping[str, AverageMeter], device: torch.device) -> None:
    """Сводит метрики со всех процессов: складывает их суммы и счётчики"""
    if not meters:
        return

    # нужна сортировка, чтобы на всех GPU складывались одинаковые объекты
    names = sorted(meters)
    packed = torch.tensor(
        [[meters[name].sum, meters[name].count] for name in names],
        dtype=torch.float64, # берём float64 для меньшей погрешности
        device=device,
    )
    all_reduce_sum_(packed)

    for name, (total, count) in zip(names, packed.tolist()):
        meters[name].sum = total
        meters[name].count = int(count)

class IoUAccumulator:
    """Копит пиксельную confusion matrix для mIoU семантической сегментации.

    Отличия от ConfusionMatrixAccumulator, из-за которых это отдельный класс:
    - пиксели с ignore_index выбрасываются ДО bincount (метка 255 при
      num_classes=19 иначе вылезла бы за пределы матрицы);
    - агрегируется IoU = TP / (TP + FP + FN), а не precision/recall.

    Память — O(num_classes^2) независимо от разрешения и размера датасета.
    """

    def __init__(
        self,
        num_classes: int,
        device: torch.device,
        ignore_index: int = 255,
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """preds, labels — [B, H, W] (или любой формы) с индексами классов.
        cm[i, j] = число пикселей истинного класса i, предсказанных как j.
        """
        preds = preds.reshape(-1).long().to(self.matrix.device)
        labels = labels.reshape(-1).long().to(self.matrix.device)

        valid = labels != self.ignore_index
        preds = preds[valid]
        labels = labels[valid]

        indices = labels * self.num_classes + preds
        batch_counts = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.matrix += batch_counts.reshape(self.num_classes, self.num_classes)

    def reset(self) -> None:
        self.matrix.zero_()

    def compute(self) -> dict[str, float]:
        """mIoU по классам, встретившимся в выборке, и общая пиксельная точность."""
        cm = self.matrix.float()
        total = cm.sum()

        if total == 0:
            return {"miou": 0.0, "pixel_acc": 0.0}

        tp = cm.diagonal()
        union = cm.sum(dim=0) + cm.sum(dim=1) - tp
        iou = tp / union.clamp_min(1e-12)

        # Класс, ни разу не встретившийся в разметке, не должен занижать среднее.
        present = cm.sum(dim=1) > 0

        return {
            "miou": iou[present].mean().item() if present.any() else 0.0,
            "pixel_acc": (tp.sum() / total).item(),
        }

    def per_class_iou(self) -> torch.Tensor:
        cm = self.matrix.float()
        tp = cm.diagonal()
        union = cm.sum(dim=0) + cm.sum(dim=1) - tp
        return tp / union.clamp_min(1e-12)


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

    def synchronize(self) -> None:
        """Складывает матрицы всех процессов. Звать ДО compute()."""
        all_reduce_sum_(self.matrix)

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
