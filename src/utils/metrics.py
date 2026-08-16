"""Метрики и агрегаторы.

Все накопители здесь хранят АДДИТИВНЫЕ величины -- суммы и счётчики (а не
готовые средние).
"""

import logging
import math
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.distributed import all_reduce_sum_

from src.utils.distributed import all_reduce_sum_

log = logging.getLogger(__name__)

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


class GradientContributionTracker:
    """Норма градиента каждого именованного компонента лосса за эпоху.

    Компонент — любой тензор из словаря, который возвращает criterion.forward()
    (например bbox/cls/dfl у YOLO, det/hekld/hokfd у DCKD, det/lcmd/tcld у
    CLoCKDistill), кроме "total". Норма считается С УЧЁТОМ λ, с которым
    компонент реально входит в total = Σ λ_i · L_i (см. weights в probe() и
    DistillationLoss.gradient_probe_weights) — то есть это вклад именно в тот
    градиент, что реально уходит в total.backward(), а не сырая величина
    самого L_i (та уже видна на графике train_loss_<component>, но она не
    учитывает, что, например, DCKD с lambda_det=0.1 давит on task loss
    в 10 раз сильнее, чем говорит его собственная норма).

    Дорогая операция: probe() делает по torch.autograd.grad(retain_graph=True)
    на каждый компонент — K лишних backward-проходов по общей части графа на
    шаге зонда (K = число компонентов). Поэтому probe() не рассчитан на вызов
    на каждом шаге — троттлинг (раз в N шагов) остаётся на вызывающей стороне
    (см. DetectionTrainer.grad_contrib_every_n_steps).
    """

    def __init__(self) -> None:
        self.norms: dict[str, AverageMeter] = defaultdict(AverageMeter)
        self.non_finite: set[str] = set()

    def probe(
        self,
        losses: dict[str, torch.Tensor],
        params: list[torch.Tensor],
        grad_scale: float = 1.0,
        weights: dict[str, float] | None = None,
    ) -> None:
        """Считает |λ_i| · ‖∂L_i/∂params‖₂ для каждого L_i из losses (кроме "total").

        weights — {key: λ_i}, обычно criterion.gradient_probe_weights; ключ,
        которого там нет, получает λ=1.0. Умножение на |λ_i| — после
        autograd.grad, а не до (домножать сам тензор перед backward'ом было бы
        математически то же самое: ‖λ·∇L‖ = |λ|·‖∇L‖ для любого λ, но лишний
        тензорный op на графе не нужен).

        grad_scale — текущий scaler.get_scale() под AMP: L_i домножается на
        него перед autograd.grad, а итоговая норма делится обратно на
        grad_scale. Без этого на fp16 многие компоненты просто underflow'ят в
        0 — та же защита, которую настоящему backward'у даёт scaler.scale(...).
        allow_unused=True: не каждый компонент обязан задевать каждый
        параметр (например det не трогает criterion.feature_adapter, который
        существует только ради hokfd/lcmd) — такие параметры просто
        пропускаются, а не роняют probe с ошибкой. Но allow_unused не спасает
        от параметров с requires_grad=False в самом params (например conv
        DFL-слоя в ultralytics YOLO заморожен намеренно, а не unused) —
        autograd.grad падает на них с "does not require grad" ещё до всякого
        allow_unused, поэтому такие параметры отфильтровываются заранее.
        Компоненты с одинаковым тензором под разными ключами (bbox/box_loss —
        алиасы для обратной совместимости логов) считаются один раз.
        """
        components: list[tuple[str, torch.Tensor]] = []
        seen_tensors: set[int] = set()
        for key, value in losses.items():
            if key == "total" or not torch.is_tensor(value) or not value.requires_grad:
                continue
            if id(value) in seen_tensors:
                continue
            seen_tensors.add(id(value))
            components.append((key, value))

        trainable_params = [p for p in params if p.requires_grad]
        if not components or not trainable_params:
            return

        weights = weights or {}

        for key, value in components:
            # λ домножается ДО autograd.grad, а не после: под AMP компонент
            # должен пройти backward ровно с тем множителем, с каким он входит
            # в scaler.scale(total).backward(), то есть λ_i · grad_scale. Иначе
            # компонент с λ<1 (det с lambda_det=0.1 у DCKD) зондируется с
            # множителем в 1/λ раз большим, чем выдерживает настоящий backward,
            # fp16-градиенты переполняются в inf, и isfinite ниже молча
            # выбрасывает его из статистики. На результат порядок не влияет:
            # ‖λ·S·∇L‖ / S = |λ|·‖∇L‖.
            probe_scale = grad_scale * abs(weights.get(key, 1.0))
            scaled = value * probe_scale if probe_scale != 1.0 else value
            grads = torch.autograd.grad(scaled, trainable_params, retain_graph=True, allow_unused=True)
            per_param_norms = [g.detach().norm() for g in grads if g is not None]
            if not per_param_norms:
                continue

            # grad_scale сам может укатиться в 0.0 (float32 underflow при затяжном
            # расхождении: scaler без остановки ловит overflow и делит scale
            # пополам). Обычное деление на 0.0 не даёт inf, как у torch/numpy,
            # а валит ZeroDivisionError — заворачиваем в ту же ветку "не конечно".
            norm_value = float(torch.stack(per_param_norms).norm()) / grad_scale if grad_scale else math.inf
            if math.isfinite(norm_value):
                self.norms[key].update(norm_value)
            elif key not in self.non_finite:
                # Не-конечная норма = переполнение fp16 в backward'е зонда.
                # Компонент выпадает из averages, а доли остальных ренормируются
                # до 100%, поэтому на графике это выглядит не как ошибка, а как
                # «компонента просто нет». Предупреждаем один раз за эпоху.
                self.non_finite.add(key)
                log.warning(
                    "Градиентный зонд: норма компонента '%s' не конечна (grad_scale=%g); "
                    "компонент исключён из grad_contribution за эту эпоху.",
                    key, grad_scale,
                )

    def results(self) -> dict[str, float]:
        """gradnorm_<key> — средняя норма компонента за эпоху; gradshare_<key>
        — её доля среди всех замеренных компонентов, % (сумма долей = 100)."""
        averages = {key: meter.avg for key, meter in self.norms.items() if meter.count}
        if not averages:
            return {}

        out = {f"gradnorm_{key}": value for key, value in averages.items()}

        total = sum(averages.values())
        if total > 0:
            out.update({f"gradshare_{key}": 100.0 * value / total for key, value in averages.items()})

        return out


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

    def synchronize(self) -> None:
        """Складывает матрицы всех процессов. Звать ДО compute().

        Матрица аддитивна, поэтому суммы матриц достаточно: mIoU по ней
        получится ровно тот же, что и на однопроцессном прогоне по всей
        выборке. Усреднять готовые mIoU ранков было бы неверно — среднее
        отношений не равно отношению сумм.
        """
        all_reduce_sum_(self.matrix)

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

    def synchronize(self) -> None:
        """Складывает накопленное со всех процессов. Звать ДО compute().

        Все пять величин — суммы и счётчики, поэтому складываются напрямую,
        а деление на pixels происходит уже после сведения.

        kl_samples намеренно не трогаем: это выборка попиксельных KL для
        гистограммы, и выборка одного ранка описывает то же распределение,
        что и общая. Собирать её по всем процессам пришлось бы
        all_gather'ом переменной длины — цена несопоставима с пользой.
        """
        for tensor in (
            self.kl_sum,
            self.agree_sum,
            self.pixels,
            self.class_agree,
            self.class_total,
        ):
            all_reduce_sum_(tensor)

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
