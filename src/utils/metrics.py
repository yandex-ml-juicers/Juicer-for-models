"""Метрики и агрегаторы."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    rows = {}
    if student is not None:
        rows["Student"] = count_parameters(student)
    if criterion is not None:
        rows["Criterion"] = count_parameters(criterion)
    if teacher is not None:
        rows["Teacher"] = count_parameters(teacher)
    if not rows:
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
        """IoU по каждому классу; NaN там, где класса не было в разметке.

        NaN, а не ноль: класс, которого не было в выборке, ничем себя не
        проявил, и нулевой столбик на графике читался бы как "модель его
        полностью провалила". Условие "класс был" — то же, что в compute(),
        поэтому среднее ненулевых столбиков в точности равно mIoU.
        """
        cm = self.matrix.float()
        tp = cm.diagonal()
        union = cm.sum(dim=0) + cm.sum(dim=1) - tp
        return torch.where(cm.sum(dim=1) > 0, tp / union.clamp_min(1e-12), torch.nan)

    def normalized_matrix(self) -> torch.Tensor:
        """Матрица ошибок, нормированная по строкам: доля пикселей класса i, ушедшая в класс j.

        Строки — истинные классы, поэтому диагональ читается как recall.
        Без нормировки карта бесполезна: road занимает треть кадра и давит
        абсолютными числами всё остальное.
        """
        cm = self.matrix.float()
        return cm / cm.sum(dim=1, keepdim=True).clamp_min(1e-12)


class TeacherSimilarity:
    """Насколько плотные предсказания ученика повторяют учительские.

    Аналог KL/agreement из классификации, но для карт [B, C, H, W]. Прямой
    перенос кода классификации сюда не годится по двум причинам:
    - `reduction="batchmean"` делит сумму на B, а не на число пикселей, то есть
      значение выросло бы в H*W раз и не сравнивалось бы между разрешениями;
    - честный проход по всем пикселям кропа 512x1024 при batch=8 — это ~80 млн
      значений на softmax, сопоставимо по цене с самим шагом обучения ради
      диагностики.

    Поэтому метрика считается по случайной подвыборке пикселей: `pixels_per_batch`
    позиций на батч (одни и те же позиции для всех картинок батча — картинки
    всё равно разные, а gather так дешевле). Оценка остаётся несмещённой,
    цена — доли процента шага.

    Пиксели с ignore_index выбрасываются: там нет разметки, и метрика должна
    быть сравнима с mIoU, который считается по тем же валидным пикселям.
    """

    def __init__(
        self,
        num_classes: int,
        device: torch.device,
        ignore_index: int = 255,
        pixels_per_batch: int = 8192,
        samples_per_update: int = 4096,
        max_samples: int = 131072,
    ) -> None:
        self.num_classes = num_classes
        self.device = device
        self.ignore_index = ignore_index
        self.pixels_per_batch = pixels_per_batch
        self.samples_per_update = samples_per_update
        self.max_samples = max_samples
        self.reset()

    def reset(self) -> None:
        # Всё копится тензорами на устройстве: .item() ни разу за эпоху,
        # то есть ни одной лишней синхронизации с GPU внутри цикла.
        self.kl_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        self.agree_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        self.pixels = torch.zeros((), dtype=torch.float64, device=self.device)
        self.class_agree = torch.zeros(self.num_classes, dtype=torch.float64, device=self.device)
        self.class_total = torch.zeros(self.num_classes, dtype=torch.float64, device=self.device)
        self.kl_samples: list[torch.Tensor] = []
        self.kept_samples = 0

    @torch.no_grad()
    def update(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """student_logits/teacher_logits — [B, C, H, W], targets — [B, H, W]."""
        if teacher_logits.shape[1] != student_logits.shape[1]:
            raise ValueError(
                f"У ученика {student_logits.shape[1]} классов, у учителя "
                f"{teacher_logits.shape[1]}: сравнивать предсказания нечем."
            )
        if teacher_logits.shape[2:] != student_logits.shape[2:]:
            # Голова учителя с другим страйдом (см. align_logits в losses):
            # приводим ДО прореживания, иначе выбранные позиции перестанут
            # соответствовать друг другу. Случай редкий, копия здесь допустима.
            teacher_logits = F.interpolate(
                teacher_logits.detach().float(),
                size=student_logits.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        student = student_logits.detach().flatten(2)  # [B, C, HW], view без копии
        teacher = teacher_logits.detach().flatten(2)
        labels = targets.detach().flatten(1)  # [B, HW]

        num_pixels = student.shape[2]
        if self.pixels_per_batch < num_pixels:
            index = torch.randint(num_pixels, (self.pixels_per_batch,), device=student.device)
            student = student.index_select(2, index)
            teacher = teacher.index_select(2, index)
            labels = labels.index_select(1, index)

        # fp32 независимо от AMP: KL живёт в хвостах распределения, а их fp16 съедает.
        log_student = F.log_softmax(student.float(), dim=1)
        log_teacher = F.log_softmax(teacher.float(), dim=1)

        valid = labels != self.ignore_index

        # KL(teacher || student) — направление то же, что в F.kl_div(log_s, p_t).
        kl = (log_teacher.exp() * (log_teacher - log_student)).sum(dim=1)[valid]
        agree = (log_student.argmax(dim=1) == log_teacher.argmax(dim=1))[valid]
        labels = labels[valid]

        self.kl_sum += kl.sum()
        self.agree_sum += agree.sum()
        self.pixels += kl.numel()
        self.class_agree += torch.bincount(
            labels[agree], minlength=self.num_classes
        ).to(self.class_agree.dtype)
        self.class_total += torch.bincount(
            labels, minlength=self.num_classes
        ).to(self.class_total.dtype)

        # Немного значений с каждого вызова — чтобы гистограмма описывала эпоху
        # целиком, а не первый её батч.
        if self.kept_samples < self.max_samples:
            self.kl_samples.append(kl[: self.samples_per_update].float())
            self.kept_samples += self.kl_samples[-1].numel()

    def compute(self) -> dict[str, float]:
        """Средние по эпохе. Пустой аккумулятор -> пустой словарь (метрик не было)."""
        if float(self.pixels) == 0.0:
            return {}
        return {
            "KL_divergence": float(self.kl_sum / self.pixels),
            "agreement_rate": float(self.agree_sum / self.pixels),
        }

    def per_class_agreement(self) -> np.ndarray:
        """Доля совпадений с учителем внутри каждого GT-класса; NaN — класса не было."""
        total = self.class_total.clamp_min(1.0)
        agreement = torch.where(self.class_total > 0, self.class_agree / total, torch.nan)
        return agreement.cpu().numpy()

    def kl_sample_values(self) -> np.ndarray:
        """Выборка попиксельных KL за эпоху — сырьё для гистограммы."""
        if not self.kl_samples:
            return np.empty(0, dtype=np.float32)
        return torch.cat(self.kl_samples).cpu().numpy()


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
