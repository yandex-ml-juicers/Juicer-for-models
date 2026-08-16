"""Обобщённый цикл обучения с дистилляцией и без.

Один Trainer покрывает все режимы; чем именно он занят, определяют декларации
лосса (requires_teacher, required_features) и конфиг:
- ученик с нуля:            teacher=None,  loss=CrossEntropy
- ванильная KD (Хинтон):    teacher=model, loss=HintonKD
- feature-based KD:         teacher=model, loss=FeatureKD (хуки + адаптеры)
"""

import math
import time
import random
from types import SimpleNamespace
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp.grad_scaler import GradScaler
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_convert, generalized_box_iou

from tqdm import tqdm

from src.data.batch_augment import MixedBatch, interpolate_losses
from src.losses.base import DistillationLoss
from src.models.feature_extractor import FeatureExtractor
from src.utils.distributed import DistInfo, unwrap, all_reduce_sum_, all_reduce_max_
from src.utils.logger import MetricsHistory, get_logger

from src.utils.metrics import AverageMeter, accuracy, ConfusionMatrixAccumulator, GradientContributionTracker, IoUAccumulator, TeacherSimilarity, count_parameters, build_param_table, sync_meters
from src.utils.plots import Plot, bar_plot, distribution_plot, image_plot, matrix_plot
from src.utils.visualization import default_palette, prediction_panel
from src.utils.prepare_targets import prepare_targets
from src.utils.detection_visualization import visualize_detection

log = get_logger(__name__)


class NormTracker:
    """Нормы градиентов и весов за эпоху, устойчивые к пропущенным шагам AMP.

    На переполнении fp16 GradScaler пропускает optimizer.step(), а
    clip_grad_norm_ на таком шаге возвращает inf (и, если задан grad_clip_norm,
    домножает градиенты на 1/inf = 0, превращая их в NaN — их отбрасывает уже
    сам scaler, на веса это не влияет). Такой шаг — не измерение нормы, а
    сообщение "шага не было", и учитывать его в статистике нельзя:
    - AverageMeter, получив одно inf, до конца эпохи возвращает inf;
    - максимум залипает на inf навсегда;
    - ClearML не умеет рисовать inf и подменяет его нулём с предупреждением
      "inf value encountered. Reporting it as '0.0'".

    Поэтому непригодные шаги в средние и максимумы не попадают, а их число
    публикуется отдельной метрикой skipped_steps: по ней видно, как часто AMP
    срывается, и это уже само по себе полезный диагностический сигнал.
    """

    def __init__(self) -> None:
        self.grad = AverageMeter()
        self.weight = AverageMeter()
        self.max_grad = -math.inf
        self.max_weight = -math.inf
        self.skipped_steps = 0
        # Все пригодные нормы за эпоху: на графике avg/max видно только коридор,
        # а форму распределения (тяжёлый хвост, бимодальность) — нет.
        # Стоит это одного float на шаг, то есть ничего.
        self.grad_values: list[float] = []

    @torch.no_grad()
    def update(self, grad_norm: Tensor | float, params: Sequence[Tensor]) -> None:
        """Учитывает один шаг. grad_norm — ровно то, что вернул clip_grad_norm_."""
        grad_norm_value = float(grad_norm)
        if math.isfinite(grad_norm_value):
            self.grad.update(grad_norm_value, n=1)
            self.max_grad = max(self.max_grad, grad_norm_value)
            self.grad_values.append(grad_norm_value)
        else:
            self.skipped_steps += 1

        # sqrt(sum ||p||^2) — та же глобальная L2-норма, что у склейки всех
        # параметров в один вектор, но без её копии в памяти (для 27M параметров
        # такая копия стоила бы ~110 МБ VRAM и лишнего трафика каждый шаг).
        weight_norm_value = float(torch.stack([p.detach().norm() for p in params]).norm())
        if math.isfinite(weight_norm_value):
            self.weight.update(weight_norm_value, n=1)
            self.max_weight = max(self.max_weight, weight_norm_value)

    def synchronize(self, device: torch.device) -> None:
        """Сводит нормы со всех процессов. Звать ДО results().

        Средние складываются суммами и счётчиками, максимумы берутся
        максимумом, пропущенные шаги — суммой (сколько всего шагов эпохи
        потеряно на всех картах вместе).

        grad_values намеренно не собираем: это сырьё для гистограммы, и
        выборка одного ранка описывает то же распределение, что и общая.
        """
        sync_meters({"grad": self.grad, "weight": self.weight}, device)

        maxima = torch.tensor([self.max_grad, self.max_weight], device=device)
        all_reduce_max_(maxima)
        self.max_grad, self.max_weight = maxima.tolist()

        skipped = torch.tensor(float(self.skipped_steps), device=device)
        all_reduce_sum_(skipped)
        self.skipped_steps = int(skipped.item())

    def results(self) -> dict[str, float | int]:
        """Метрики эпохи. NaN = ни одного пригодного шага (ClearML такое не рисует)."""
        return {
            "avg_grad_norm": self.grad.avg if self.grad.count else math.nan,
            "max_grad_norm": self.max_grad if math.isfinite(self.max_grad) else math.nan,
            "avg_weight_norm": self.weight.avg if self.weight.count else math.nan,
            "max_weight_norm": self.max_weight if math.isfinite(self.max_weight) else math.nan,
            "skipped_steps": self.skipped_steps,
        }


def mix_batch(batch_augment: Callable | None, images: Tensor, targets: Tensor) -> MixedBatch:
    """Применяет Mixup/CutMix, если он задан; иначе отдаёт батч как есть."""
    if batch_augment is None:
        return MixedBatch(images, targets, targets, 1.0)

    return batch_augment(images, targets)


def compute_losses(
    criterion: DistillationLoss,
    student_logits,
    teacher_logits,
    mixed: MixedBatch,
    student_features: dict | None = None,
    teacher_features: dict | None = None,
) -> dict[str, Tensor]:
    """Значения лосса с учётом смешивания батча.

    При lam < 1 лосс считается по обоим наборам таргетов и линейно
    интерполируется: это эквивалент смешивания one-hot меток, но не требует,
    чтобы каждый лосс проекта умел работать с soft-таргетами
    (обоснование — в src/data/batch_augment.py). Модель при этом прогоняется
    ОДИН раз: второй вызов идёт по тем же логитам и тем же картам признаков,
    так что стоит он долю процента шага.
    """
    losses = criterion(
        student_logits,
        teacher_logits,
        mixed.targets_a,
        student_features=student_features,
        teacher_features=teacher_features,
    )

    if mixed.lam >= 1.0:
        return losses

    losses_b = criterion(
        student_logits,
        teacher_logits,
        mixed.targets_b,
        student_features=student_features,
        teacher_features=teacher_features,
    )
    return interpolate_losses(losses, losses_b, mixed.lam)


def wrap_ddp(
    module: nn.Module,
    dist: DistInfo,
    find_unused_parameters: bool,
    broadcast_buffers: bool,
) -> nn.Module:
    """Оборачивает модуль в DDP; при однопроцессном запуске отдаёт его как есть.

    Звать ПОСЛЕ того, как на модуль навешены хуки FeatureExtractor: хуки
    живут на подмодулях и переживают обёртку, а вот искать слои по именам
    в уже обёрнутой модели пришлось бы с префиксом "module.".
    """
    if not dist.is_distributed:
        return module
    return DistributedDataParallel(
        module,
        device_ids=[dist.local_rank] if dist.device.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit_batches: int | None = None, # кол-во батчей для eval на ранк
) -> tuple[float, float]:
    """Возвращает (средний CE-лосс, точность) на выборке. Всегда в fp32."""
    was_training = model.training
    model.eval()

    # [сумма лосса, верных ответов, объектов]
    totals = torch.zeros(3, device=device)

    for step, (images, labels) in enumerate(loader):
        if limit_batches is not None and step >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)

        totals[0] += F.cross_entropy(logits, labels, reduction="sum")
        totals[1] += (logits.argmax(dim=1) == labels).sum()
        totals[2] += labels.size(0)

    if was_training:
        model.train()

    all_reduce_sum_(totals)
    loss_sum, correct, count = totals.tolist()
    if count == 0:
        return 0.0, 0.0
    return loss_sum / count, correct / count


class Trainer:
    """Собирает воедино ученика, (опционально) учителя, лосс и оптимизацию.

    Инварианты, которые Trainer гарантирует:
    - учитель всегда заморожен (eval + requires_grad=False) и не попадает
      в optimizer;
    - хуки на промежуточные слои ставятся только если лосс их запросил,
      и очищаются перед каждым forward'ом;
    - в optimizer, переданный извне, обязаны входить параметры criterion
      (адаптеры) — это проверяется в конструкторе, а не остаётся на совести
      вызывающего кода.
    """

    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module | None,
        criterion: DistillationLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        num_classes: int,
        dist: DistInfo,
        output_dir: Path,
        epochs: int,
        amp: bool = False,
        grad_clip_norm: float | None = None,
        limit_train_batches: int | None = None,
        limit_eval_batches: int | None = None,
        save_best: bool = True,
        save_last: bool = True,
        progress_bar: bool = True,
        find_unused_parameters: bool = False,
        broadcast_buffers: bool = True,
        metrics_callback: tuple[Callable, ...] | None = None,
        scalars: dict[str, float | int | str],
        batch_augment: Callable | None = None,
    ) -> None:
        if criterion.requires_teacher and teacher is None:
            raise ValueError(
                f"Лосс {type(criterion).__name__} требует учителя, но model/teacher=null. "
                f"Либо задай учителя, либо возьми loss=ce."
            )

        self.student = student
        self.teacher = teacher
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.num_classes = num_classes
        self.dist = dist
        self.device = dist.device
        self.find_unused_parameters = find_unused_parameters
        self.broadcast_buffers = broadcast_buffers
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.grad_clip_norm = grad_clip_norm
        self.limit_train_batches = limit_train_batches
        self.limit_eval_batches = limit_eval_batches
        self.save_best = save_best
        self.save_last = save_last
        self.progress_bar = progress_bar
        # Mixup/CutMix: применяется к уже собранному батчу на устройстве,
        # только на обучении. На eval смешивания нет никогда — иначе метрика
        # измеряла бы качество на несуществующих картинках.
        self.batch_augment = batch_augment
        # Точка стыковки внешнего трекера (ClearML и т.п.): вызывается после
        # каждой эпохи со строкой метрик — той же, что уходит в history.csv.
        # Trainer ничего не знает о трекере, колбэк собирает scripts/train.py.
        self.metrics_callback_scalar = metrics_callback[0] if metrics_callback is not None else None
        self.metrics_callback_single = metrics_callback[1] if metrics_callback is not None else None
        self.metrics_callback_table = metrics_callback[2] if metrics_callback is not None else None
        self.scalars = scalars
        self.accumulation_steps = accumulation_steps
        # metrics = {'loss.train_loss': }
        self.train_confmat: ConfusionMatrixAccumulator | None = None
        if {"precision", "recall", "f1"} & set(self.scalars):
            self.train_confmat = ConfusionMatrixAccumulator(self.num_classes, self.device)
        

        self.amp_enabled = amp and self.device.type == "cuda"
        self.scaler = GradScaler(self.device.type, enabled=self.amp_enabled)

        if self.teacher is not None:
            self.teacher.eval()
            self.teacher.requires_grad_(False)

        if self.metrics_callback_table is not None:
            self.metrics_callback_table(build_param_table(self.student, self.teacher, self.criterion))

        criterion_params = list(self.criterion.parameters())
        if criterion_params:
            optimizer_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
            missing = [p for p in criterion_params if id(p) not in optimizer_params]
            if missing:
                raise ValueError(
                    "У лосса есть обучаемые параметры (адаптеры), не попавшие в optimizer. "
                    "Optimizer должен собираться из student.parameters() + criterion.parameters()."
                )

        self.student_extractor: FeatureExtractor | None = None
        self.teacher_extractor: FeatureExtractor | None = None
        if criterion.required_features:
            layers = list(criterion.required_features)
            self.student_extractor = FeatureExtractor(self.student, layers)
            if self.teacher is not None:
                self.teacher_extractor = FeatureExtractor(self.teacher, layers)

        self.student = wrap_ddp(
            self.student, self.dist, self.find_unused_parameters, self.broadcast_buffers
        )

        if any(parameter.requires_grad for parameter in self.criterion.parameters()):
            self.criterion = wrap_ddp(
                self.criterion, self.dist, self.find_unused_parameters, self.broadcast_buffers
            )
        # учителя не оборачиваем в DDP, т.к. синхронизация между процессами не нужна

    def fit(self) -> dict:
        history = MetricsHistory(self.output_dir / "history.csv") if self.dist.is_main else None
        best_acc, best_epoch = 0.0, 0

        try:
            for epoch in range(1, self.epochs + 1):

                start = time.time()
                lr=self.optimizer.param_groups[0]["lr"]

                train_loss_components, other_train_metrics = self._train_epoch(epoch)
                eval_loss, eval_acc = evaluate(
                    unwrap(self.student), self.eval_loader, self.device, self.limit_eval_batches
                )
                if self.scheduler is not None:
                    self.scheduler.step()

                all_values = {
                    "epoch": epoch,
                    "lr": lr,
                    "world_size": self.dist.world_size,
                    "global_batch_size": (self.train_loader.batch_size or 0) * self.dist.world_size,
                    "eval_loss": eval_loss,
                    "eval_acc": eval_acc,
                    "time_epoch": round(time.time() - start, 1),
                    **{f"train_loss_{key}": value for key, value in train_loss_components.items()},
                    **{f"train_{key}": value for key, value in other_train_metrics.items()},
                }

                if history is not None:
                    history.append(all_values)
                if self.metrics_callback_scalar is not None:
                    self.metrics_callback_scalar(all_values)

                is_best = eval_acc > best_acc
                if is_best:
                    best_acc, best_epoch = eval_acc, epoch
                    if self.save_best:
                        self._save_checkpoint("best.pt", epoch, best_acc)
                if self.save_last:
                    self._save_checkpoint("last.pt", epoch, best_acc)
                if epoch == 40:
                    self._save_checkpoint("checkpoint_40_epochs.pt", epoch, best_acc)
                if epoch == 80:
                    self._save_checkpoint("checkpoint_80_epochs.pt", epoch, best_acc)

                log.info(
                    "Эпоха %02d/%d | lr=%.6f | train loss=%.4f | train acc=%.2f%% | "
                    "eval loss=%.4f | eval acc=%.2f%%%s | %.1f c",
                    epoch,
                    self.epochs,
                    lr,
                    train_loss_components["total"],
                    other_train_metrics["acc"] * 100,
                    eval_loss,
                    eval_acc * 100,
                    " *" if is_best else "",
                    all_values["time_epoch"],
                )
        finally:
            if self.student_extractor is not None:
                self.student_extractor.remove()
            if self.teacher_extractor is not None:
                self.teacher_extractor.remove()

        log.info("Лучшая точность: %.2f%% (эпоха %d)", best_acc * 100, best_epoch)
        return {"best_acc": best_acc, "best_epoch": best_epoch}

    def _train_epoch(self, epoch: int) -> tuple:
        self.student.train()
        self.criterion.train()

        # Без set_epoch DistributedSampler выдаёт одну и ту же перестановку
        # каждую эпоху, то есть порядок данных перестаёт меняться.
        sampler = getattr(self.train_loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        norms = NormTracker()

        if self.train_confmat is not None:
            self.train_confmat.reset()

        meters_avg: dict[str, AverageMeter] = defaultdict(AverageMeter)
        meters_avg_loss: dict[str, AverageMeter] = defaultdict(AverageMeter)
        meters_other: dict[str, float | int | str] = {}
        iterator = tqdm(
            self.train_loader,
            desc=f"Эпоха {epoch}/{self.epochs}",
            disable=not self.progress_bar or not self.dist.is_main,
            leave=False,
        )

        for step, (images, labels) in enumerate(iterator):
            if self.limit_train_batches is not None and step >= self.limit_train_batches:
                iterator.close()
                break
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            batch_size = labels.size(0)

            # Смешивание идёт ДО прогона учителя: учитель обязан видеть ту же
            # картинку, что и ученик, иначе дистиллируются знания о кадре,
            # которого ученику не показывали.
            mixed = mix_batch(self.batch_augment, images, labels)
            images, labels = mixed.images, mixed.targets_a

            self.optimizer.zero_grad(set_to_none=True)
            if self.student_extractor is not None:
                self.student_extractor.clear()
            if self.teacher_extractor is not None:
                self.teacher_extractor.clear()

            teacher_logits = None
            if self.teacher is not None:
                with torch.no_grad(), torch.autocast(self.device.type, enabled=self.amp_enabled):
                    teacher_logits = self.teacher(images)

            with torch.autocast(self.device.type, enabled=self.amp_enabled):
                student_logits = self.student(images)
                losses = compute_losses(
                    self.criterion,
                    student_logits,
                    teacher_logits,
                    mixed,
                    student_features=(
                        self.student_extractor.features if self.student_extractor else None
                    ),
                    teacher_features=(
                        self.teacher_extractor.features if self.teacher_extractor else None
                    ),
                )

            if self.teacher is not None and teacher_logits is not None:
                if {"KL_divergence", "agreement_rate"} & set(self.scalars):
                    with torch.no_grad():
                        # обе метрики — в fp32, независимо от AMP, ради численной стабильности
                        s_logits = student_logits.detach().float()
                        t_logits = teacher_logits.detach().float()

                        KL = F.kl_div(
                            F.log_softmax(s_logits, dim=1),
                            F.softmax(t_logits, dim=1),
                            reduction="batchmean",
                        )
                        meters_avg["KL_divergence"].update(KL.item(), batch_size)

                        # доля примеров, где top-1 ученика совпал с top-1 учителя
                        agreement = (s_logits.argmax(dim=1) == t_logits.argmax(dim=1)).float().mean()
                        meters_avg["agreement_rate"].update(agreement.item(), batch_size)

            self.scaler.scale(losses["total"]).backward()

            self.scaler.unscale_(self.optimizer)

            params = [p for group in self.optimizer.param_groups for p in group["params"]]
            clip_threshold = self.grad_clip_norm if self.grad_clip_norm is not None else float("inf") 
            grad_norm = torch.nn.utils.clip_grad_norm_(params, clip_threshold)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            norms.update(grad_norm, params)

            for key, value in losses.items():
                meters_avg_loss[key].update(value.item(), batch_size)

            # При включённом Mixup/CutMix labels — это targets_a, доминирующая
            # половина смеси (lam >= 0.5 гарантируется sample_lambda). Train-acc
            # тогда становится оценкой снизу; eval-метрика этим не затронута,
            # на валидации смешивания нет.
            meters_avg["acc"].update(accuracy(student_logits.float(), labels), batch_size) #train_acc

            if self.train_confmat is not None:
                self.train_confmat.update(student_logits.detach().argmax(dim=1), labels)

            iterator.set_postfix({"loss": f"{losses['total'].item():.3f}"})

        sync_meters(meters_avg_loss, self.device)
        sync_meters(meters_avg, self.device)

        if self.train_confmat is not None:
            self.train_confmat.synchronize()
            meters_other.update(self.train_confmat.compute()) # pr, rec, F1

        train_loss_components = {key: meter.avg for key, meter in meters_avg_loss.items()} # train_loss_components

        norms.synchronize(self.device)
        meters_other.update(norms.results())

        other_train_metrics = {**{key: meter.avg for key, meter in meters_avg.items()}, **meters_other}

        return train_loss_components, other_train_metrics

    def _save_checkpoint(self, filename: str, epoch: int, best_acc: float) -> None:
        if not self.dist.is_main:
            return

        checkpoint = {
            "epoch": epoch,
            "best_acc": best_acc,
            "world_size": self.dist.world_size,
            "student_state": unwrap(self.student).state_dict(),
            # Состояние лосса = адаптеры каналов (у CrossEntropy/HintonKD пусто).
            "criterion_state": unwrap(self.criterion).state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict(),
        }
        torch.save(checkpoint, self.output_dir / filename)


def pick_debug_indices(dataset, count: int) -> list[int]:
    """Равномерно разбросанные по валидации индексы картинок для Debug Samples.

    Равномерно, а не случайно и не первые подряд: Cityscapes отсортирован по
    городам, поэтому первые N кадров — это N видов одной улицы, а случайная
    выборка на каждой отправке даёт новые кадры, которые не с чем сравнивать.
    Фиксированные кадры — единственный способ увидеть, как меняется предсказание.

    count <= 0 — картинки не отправляются вовсе.
    """
    if count <= 0:
        return []

    try:
        total = len(dataset)
    except TypeError:  # IterableDataset — индексов нет
        return []

    count = min(count, total)
    if count == 0:
        return []
    if count == 1:
        return [0]

    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


@torch.no_grad()
def detection_evaluate(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    prediction_postprocessor: Callable | None = None,
    label_offset: int = 1,
    limit_batches: int | None = None,
    targers_mode: str | None = None,
    class_names: dict[int, str] | None = None,
    amp: bool = False,
) -> tuple[float, float]:
    was_model_training = model.training
    was_criterion_training = criterion.training
    amp_enabled = amp and device.type == "cuda"

    model.eval()
    criterion.eval()

    loss_meter = AverageMeter()
    # sync_on_compute=False: состояние метрики лежит на CPU, а группа процессов
    # поднята с nccl-бэкендом даже при world_size=1 — иначе compute() уходит
    # в all_gather по CPU-тензорам и падает.
    # class_metrics=True включает map_per_class: на Cityscapes классы
    # различаются по числу объектов в 200 раз (car 27153 против train 171),
    # и средний mAP без разбивки не говорит ничего.
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
        sync_on_compute=False,
    )

    total_predictions = 0
    total_images = 0
    max_score = 0.0

    for step, (images, targets) in enumerate(loader):
        if limit_batches is not None and step >= limit_batches:
            break

        # Одна склейка на CPU + один H2D-перенос вместо переноса каждой
        # картинки батча по отдельности (см. тот же приём в _train_epoch).
        images = torch.stack(images, dim=0).to(device, non_blocking=True)

        targets_device = prepare_targets(targets, device, targers_mode, label_offset)

        # Train-шаг уже считался в autocast при amp=true — eval гонялся в fp32
        # и терял ускорение на тензорных ядрах для той же модели.
        with torch.autocast(device.type, enabled=amp_enabled):
            if targers_mode == "lw-detr-small":
                outputs = model(pixel_values=images, labels=targets_device)
            else:
                outputs = model(images)
            losses = criterion(outputs, teacher_outputs=None, labels=targets_device)

        batch_size = len(images)
        loss_meter.update(losses["total"].item(), batch_size)

        if prediction_postprocessor is not None:
            predictions = prediction_postprocessor(outputs, images)
        else:
            predictions = outputs

        predictions_for_metric = []
        targets_for_metric = []

        for prediction, target in zip(predictions, targets):
            scores = prediction["scores"]
            total_predictions += int(scores.numel())
            if scores.numel():
                max_score = max(max_score, float(scores.max()))

            # Постпроцессор уже вернул метки в системе датасета (label_offset
            # прибавлен там), таргеты в ней же — вычитать его здесь нечего.
            predictions_for_metric.append({
                "boxes": prediction["boxes"].detach().cpu(),
                "scores": prediction["scores"].detach().cpu(),
                "labels": prediction["labels"].detach().cpu(),
            })

            metric_target = {
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu(),
            }

            if "area" in target:
                metric_target["area"] = (target["area"].detach().cpu())

            targets_for_metric.append(metric_target)

        total_images += len(predictions_for_metric)
        metric.update(predictions_for_metric, targets_for_metric)

    computed_metrics = metric.compute()

    metrics = {
        "map": computed_metrics["map"].item(),
        "map_50": computed_metrics["map_50"].item(),
        "map_75": computed_metrics["map_75"].item(),
        "mar_100": computed_metrics["mar_100"].item(),
        # Разбивка по размерам: после ресайза 1024x2048 -> 512x1024 около 70%
        # объектов Cityscapes попадают в COCO-категорию "small".
        "map_small": computed_metrics["map_small"].item(),
        "map_medium": computed_metrics["map_medium"].item(),
        "map_large": computed_metrics["map_large"].item(),
        # Диагностика коллапса: и число детекций, и максимум score падают
        # раньше, чем mAP успевает дойти до нуля.
        "predictions_per_image": total_predictions / max(total_images, 1),
        "max_score": max_score,
    }

    # torchmetrics отдаёт -1 для классов, которых нет в выборке.
    classes = torch.atleast_1d(computed_metrics["classes"])
    per_class = torch.atleast_1d(computed_metrics["map_per_class"])

    for class_id, value in zip(classes.tolist(), per_class.tolist()):
        name = class_names.get(class_id, class_id) if class_names else class_id
        metrics[f"ap_{name}"] = float(value)

    if was_model_training:
        model.train()

    if was_criterion_training:
        criterion.train()

    return loss_meter.avg, metrics

class DetectionTrainer:
    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module | None,
        criterion: DistillationLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        num_classes: int,
        device: torch.device,
        output_dir: Path,
        epochs: int,
        amp: bool = False,
        grad_clip_norm: float | None = None,
        limit_train_batches: int | None = None,
        limit_eval_batches: int | None = None,
        save_best: bool = True,
        save_last: bool = True,
        progress_bar: bool = True,
        metrics_callback: tuple[Callable, ...] | None = None,
        scalars: dict[str, float | int | str],
        prediction_postprocessor: Callable | None = None,
        label_offset: int = 1,
        targers_mode: str | None = None,
        track_train_map: bool = True,
        class_names: dict[int, str] | None = None,
        normalize: tuple[Sequence[float], Sequence[float]] | None = None,
        plots: dict | None = None,
    ) -> None:

        if criterion.requires_teacher and teacher is None:
            raise ValueError(
                f"Лосс {type(criterion).__name__} требует учителя, но model/teacher=null. "
            )

        self.prediction_postprocessor = prediction_postprocessor

        self.label_offset = label_offset
        self.targers_mode = targers_mode
        self.track_train_map = track_train_map
        self.class_names = class_names
        # (mean, std) из конфига датасета или None, если нормализации нет:
        # возвращает Debug Samples исходные цвета. То же, что у SegmentationTrainer.
        self.normalize = normalize

        plots = dict(plots) if plots is not None else {}
        # Debug Samples выключаются любым из двух нулей: debug_samples: 0 —
        # «картинки не нужны совсем», debug_every_n_epochs: 0 — «не слать
        # периодически». Раньше отправка была захардкожена каждые 5 эпох.
        self.debug_every_n_epochs = int(plots.get("debug_every_n_epochs", 10))
        self.debug_indices = pick_debug_indices(eval_loader.dataset, int(plots.get("debug_samples", 4)))

        # Зонд вклада каждого компонента лосса в градиент (см.
        # GradientContributionTracker): 0 — не считать совсем, N>0 — раз в N
        # шагов внутри эпохи. Каждый зонд — это K лишних backward-проходов
        # (K = число именованных компонентов лосса), поэтому по умолчанию
        # выключено.
        self.grad_contrib_every_n_steps = int(plots.get("grad_contrib_every_n_steps", 0))

        self.student = student
        self.teacher = teacher
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.num_classes = num_classes
        self.device = device
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        # Стартовая эпоха и лучший результат "с нуля"; load_checkpoint() их перезатрёт.
        self.start_epoch = 1
        self.best_map = -1.0
        self.best_epoch = 0
        self.grad_clip_norm = grad_clip_norm
        self.limit_train_batches = limit_train_batches
        self.limit_eval_batches = limit_eval_batches
        self.save_best = save_best
        self.save_last = save_last
        self.progress_bar = progress_bar
        # Точка стыковки внешнего трекера (ClearML и т.п.): вызывается после
        # каждой эпохи со строкой метрик — той же, что уходит в history.csv.
        # Trainer ничего не знает о трекере, колбэк собирает scripts/train.py.
        self.metrics_callback_scalar = metrics_callback[0] if metrics_callback is not None else None
        self.metrics_callback_single = metrics_callback[1] if metrics_callback is not None else None
        self.metrics_callback_table = metrics_callback[2] if metrics_callback is not None else None
        self.metrics_callback_debug = metrics_callback[4] if metrics_callback is not None and len(metrics_callback) > 4 else None

        self.scalars = scalars

        self.amp_enabled = amp and device.type == "cuda"
        self.scaler = GradScaler(device.type, enabled=self.amp_enabled)

        self.student.to(self.device)
        self.criterion.to(self.device)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.teacher is not None:
            self.teacher.to(self.device)
            self.teacher.eval()
            self.teacher.requires_grad_(False)

        if self.metrics_callback_table is not None:
            self.metrics_callback_table(build_param_table(self.student, self.teacher, self.criterion))

        criterion_params = list(self.criterion.parameters())
        if criterion_params:
            optimizer_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
            missing = [p for p in criterion_params if id(p) not in optimizer_params]
            if missing:
                raise ValueError(
                    "У лосса есть обучаемые параметры (адаптеры), не попавшие в optimizer. "
                    "Optimizer должен собираться из student.parameters() + criterion.parameters()."
                )

        self.student_extractor: FeatureExtractor | None = None
        self.teacher_extractor: FeatureExtractor | None = None
        # Большинство лоссов снимают одноимённые слои у ученика и учителя
        # (required_features). DCKD — исключение: у ученика и учителя разные
        # архитектуры и разные имена слоёв, поэтому у него есть
        # student_required_features/teacher_required_features. getattr с
        # фоллбэком на required_features не меняет поведение остальных
        # лоссов (FitNets, FeatureKD, MGD, SP) — у них имена общие.
        student_layers = list(getattr(criterion, "student_required_features", criterion.required_features))
        teacher_layers = list(getattr(criterion, "teacher_required_features", criterion.required_features))
        if student_layers:
            self.student_extractor = FeatureExtractor(self.student, student_layers)
        if teacher_layers and self.teacher is not None:
            self.teacher_extractor = FeatureExtractor(self.teacher, teacher_layers)

    def fit(self) -> dict:
        history = MetricsHistory(self.output_dir / "history.csv", resume=self.start_epoch > 1)
        # -1.0, а не 0.0: при коллапсе модели mAP ровно 0.0 и условие `>` никогда
        # не сработало бы — best.pt не создавался вовсе.
        best_map, best_epoch = self.best_map, self.best_epoch

        if history.rows and self.metrics_callback_scalar is not None:
            # history.csv пишется синхронно и пережил крэш целиком, а отчёт в
            # ClearML асинхронный — часть уже посчитанных эпох могла не успеть
            # долететь до сервера. Реплеим все локально сохранённые эпохи
            # заново: те же title/series/iteration в ClearML — это перезапись
            # той же точки графика, а не дубль, так что безопасно послать и то,
            # что уже долетело.
            for row in history.rows:
                self.metrics_callback_scalar(row)

        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                start = time.time()
                lr = self.optimizer.param_groups[0]["lr"]

                train_loss_components, other_train_metrics = self._train_epoch(epoch)
                eval_loss, eval_metrics = detection_evaluate(
                    model=self.student,
                    criterion=self.criterion, 
                    loader=self.eval_loader, 
                    device=self.device,
                    prediction_postprocessor=self.prediction_postprocessor,
                    label_offset=self.label_offset,
                    limit_batches=self.limit_eval_batches,
                    targers_mode=self.targers_mode,
                    class_names=self.class_names,
                    amp=self.amp_enabled,
                )

                self._report_detection_samples(epoch)

                if self.scheduler is not None:
                    self.scheduler.step()

                all_values = {
                    "epoch": epoch,
                    "lr": lr,
                    "eval_loss": eval_loss,
                    # Перечисление ключей руками означало бы, что новые метрики
                    # (per-class AP, разбивка по размерам) не доедут ни до
                    # history.csv, ни до колбэков.
                    **{f"eval_{key}": value for key, value in eval_metrics.items()},
                    "time_epoch": round(time.time() - start, 1),
                    **{
                        f"train_loss_{key}": value
                        for key, value in train_loss_components.items()
                    },
                    **{
                        f"train_{key}": value 
                        for key, value in other_train_metrics.items()
                    },
                }

                history.append(all_values)
                if self.metrics_callback_scalar is not None:
                    self.metrics_callback_scalar(all_values)

                is_best = eval_metrics["map"] > best_map
                if is_best:
                    best_map = eval_metrics["map"]
                    best_epoch = epoch
                    if self.save_best:
                        self._save_checkpoint("best.pt", epoch, best_map, best_epoch)
                if self.save_last:
                    self._save_checkpoint("last.pt", epoch, best_map, best_epoch)

                log.info(
                    "Эпоха %02d/%d | lr=%.6f | "
                    "train loss=%.4f | eval loss=%.4f | "
                    "mAP=%.4f | mAP@50=%.4f | mAP@75=%.4f%s | %.1f c",
                    epoch,
                    self.epochs,
                    lr,
                    train_loss_components["total"],
                    eval_loss,
                    eval_metrics["map"],
                    eval_metrics["map_50"],
                    eval_metrics["map_75"],
                    " *" if is_best else "",
                    all_values["time_epoch"],
                )
        
        finally:
            if self.student_extractor is not None:
                self.student_extractor.remove()
            if self.teacher_extractor is not None:
                self.teacher_extractor.remove()

        log.info("Лучший mAP: %.4f (эпоха %d)", best_map, best_epoch)
        return {"best_map": best_map, "best_epoch": best_epoch}

    def _train_epoch(self, epoch: int) -> tuple:
        self.student.train()
        self.criterion.train()

        norms = NormTracker()
        grad_contrib = GradientContributionTracker()

        agreement_correct = 0
        agreement_total = 0

        meters_avg: dict[str, AverageMeter] = defaultdict(AverageMeter)
        meters_avg_loss: dict[str, AverageMeter] = defaultdict(AverageMeter)
        meters_other: dict[str, float | int | str] = {}
        meters_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", sync_on_compute=False)

        iterator = tqdm(
            self.train_loader,
            desc=f"Эпоха {epoch}/{self.epochs}",
            disable=not self.progress_bar,
            leave=False,
        )

        # Обнуляем градиенты ПЕРЕД началом эпохи
        self.optimizer.zero_grad(set_to_none=True)

        for step, (images, targets) in enumerate(iterator):
            if self.limit_train_batches is not None and step >= self.limit_train_batches:
                iterator.close()
                break

            targets_for_meters_map = targets

            # Одна склейка на CPU + один H2D-перенос вместо переноса каждой
            # картинки батча по отдельности. torch.stack и раньше требовал
            # одинакового размера всех картинок батча — трансформы всех
            # детекционных экспериментов ресайзят к фиксированному
            # data.dataset.image_size, так что условие не меняется, меняется
            # только порядок операций (стек -> перенос, а не перенос -> стек).
            images = torch.stack(images, dim=0).to(self.device, non_blocking=True)
            targets = prepare_targets(targets, self.device, self.targers_mode, self.label_offset)
            batch_size = images.size(0)

            self.optimizer.zero_grad(set_to_none=True)
            if self.student_extractor is not None:
                self.student_extractor.clear()
            if self.teacher_extractor is not None:
                self.teacher_extractor.clear()

            teacher_outputs = None
            if self.teacher is not None:
                with torch.no_grad(), torch.autocast(self.device.type, enabled=self.amp_enabled):
                    teacher_outputs = self.teacher(images)

            with torch.autocast(self.device.type, enabled=self.amp_enabled):
                if self.teacher is not None:
                    student_outputs = self.student(images)
                else:
                    if self.targers_mode == "lw-detr-small":
                        student_outputs = self.student(pixel_values=images, labels=targets)
                    else:
                        student_outputs = self.student(images)

                losses = self.criterion(
                    student_outputs,
                    teacher_outputs,
                    targets,
                    student_features=(
                        self.student_extractor.features if self.student_extractor else None
                    ),
                    teacher_features=(
                        self.teacher_extractor.features if self.teacher_extractor else None
                    ),
                )

            params = [p for group in self.optimizer.param_groups for p in group["params"]]

            # Зонд вклада компонентов лосса в градиент — ДО настоящего
            # backward'а: probe() держит граф живым через retain_graph=True,
            # чтобы backward() ниже мог пройти по нему как обычно.
            if self.grad_contrib_every_n_steps and step % self.grad_contrib_every_n_steps == 0:
                # gradient_probe_keys сужает набор до реально независимых
                # слагаемых total, если словарь лосса вперемешку содержит и
                # их, и детальную раскладку одного из них для логов (см.
                # DistillationLoss.gradient_probe_keys, KDDETRLoss).
                probe_keys = getattr(self.criterion, "gradient_probe_keys", None)
                losses_for_probe = (
                    losses if probe_keys is None
                    else {key: losses[key] for key in probe_keys if key in losses}
                )
                grad_contrib.probe(
                    losses_for_probe, params,
                    grad_scale=self.scaler.get_scale() if self.amp_enabled else 1.0,
                    weights=getattr(self.criterion, "gradient_probe_weights", None),
                )

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)

            clip_threshold = self.grad_clip_norm if self.grad_clip_norm is not None else float("inf") 
            grad_norm = torch.nn.utils.clip_grad_norm_(params, clip_threshold)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            norms.update(grad_norm, params)

            if self.track_train_map and step % 5 == 0:
                self._update_detection_meters_map(
                    metric=meters_map,
                    student_outputs=student_outputs,
                    images=images,
                    targets=targets_for_meters_map,
                )

            if self.teacher is not None:
                correct, total = self._detection_agreement_rate(student_outputs, teacher_outputs, num_classes=8, teacher_topk=100, student_topk=1000)
                agreement_correct += correct
                agreement_total += total

            # Один .tolist() на все компоненты лосса вместо отдельного .item()
            # на каждую (total/bbox/cls/dfl) — каждый .item() это отдельная
            # синхронизация с GPU, а на train-шаге они дороже, чем на eval.
            loss_keys = list(losses.keys())
            loss_values = torch.stack([losses[key] for key in loss_keys]).tolist()
            loss_values_by_key = dict(zip(loss_keys, loss_values))
            for key, value in loss_values_by_key.items():
                meters_avg_loss[key].update(value, batch_size)

            iterator.set_postfix({"loss": f"{loss_values_by_key['total']:.3f}"})

        train_loss_components = {key: meter.avg for key, meter in meters_avg_loss.items()}
        meters_other.update(norms.results())
        meters_other.update(grad_contrib.results())

        if self.teacher is not None:
            agreement_rate = agreement_correct / max(agreement_total, 1)

        other_train_metrics = {
            **{key: meter.avg for key, meter in meters_avg.items()},
            **meters_other,
        }

        if self.track_train_map:
            computed_meters_map = meters_map.compute()
            other_train_metrics |= {
                key: computed_meters_map[key].item()
                for key in ("map", "map_50", "map_75", "mar_100")
            }

        if self.teacher is not None:
            other_train_metrics["agreement_rate"] = agreement_rate

        return train_loss_components, other_train_metrics

    def _update_detection_meters_map(
        self,
        metric: MeanAveragePrecision,
        student_outputs,
        images,
        targets,
    ) -> None:
        with torch.no_grad():

            # LW-DETR использует несколько query-групп во время train.
            # Для метрики берем только первую группу, как при inference.
            config = getattr(self.student, "config", None)

            if (
                config is not None
                and getattr(config, "group_detr", 1) > 1
                and hasattr(student_outputs, "logits")
                and hasattr(student_outputs, "pred_boxes")
            ):
                num_queries = config.num_queries

                logits = student_outputs.logits[:, :num_queries]
                pred_boxes = student_outputs.pred_boxes[:, :num_queries]

                query_scores = logits.sigmoid().amax(dim=-1)

                top_k = min(100, logits.shape[1])
                topk_indices = torch.topk(query_scores, k=top_k, dim=1).indices
                logits = torch.gather(logits, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, logits.shape[-1]))

                pred_boxes = torch.gather(pred_boxes, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, 4))

                student_outputs = SimpleNamespace(
                    logits=logits.detach(),
                    pred_boxes=pred_boxes.detach(),
                )

            if self.prediction_postprocessor is not None:
                predictions = self.prediction_postprocessor(student_outputs, images)
            else:
                predictions = student_outputs

            predictions_metric = []
            targets_metric = []

            for prediction, target in zip(predictions, targets):
                boxes = prediction["boxes"]
                scores = prediction["scores"]
                labels = prediction["labels"]

                # Оставляем максимум 100 лучших detections
                if scores.numel() > 100:
                    topk_indices = torch.topk(scores, k=100).indices

                    boxes = boxes[topk_indices]
                    scores = scores[topk_indices]
                    labels = labels[topk_indices]

                predictions_metric.append(
                    {
                        "boxes": boxes.detach().float().cpu(),
                        "scores": scores.detach().float().cpu(),
                        "labels": labels.detach().long().cpu(),
                    }
                )

                metric_target = {
                    "boxes": (target["boxes"].detach().float().cpu()),
                    "labels": target["labels"].detach().long().cpu(),
                }

                if "area" in target:
                    metric_target["area"] = (target["area"].detach().float().cpu())

                targets_metric.append(metric_target)

            metric.update(predictions_metric, targets_metric)

    @torch.no_grad()
    def _detection_agreement_rate(self, student_outputs, teacher_outputs, num_classes: int, teacher_topk: int=100, student_topk: int=1000) -> tuple[int, int]:
        student_logits = student_outputs["kd_logits"]
        student_boxes = student_outputs["kd_boxes"]
        teacher_logits = teacher_outputs.logits
        teacher_boxes = teacher_outputs.pred_boxes

        if student_logits.shape[-1] != num_classes and student_logits.shape[1] == num_classes:
            student_logits = student_logits.transpose(1, 2)

        if student_boxes.shape[-1] != 4 and student_boxes.shape[1] == 4:
            student_boxes = student_boxes.transpose(1, 2)

        student_probs = torch.sigmoid(student_logits[..., :num_classes])
        teacher_probs = F.softmax(teacher_logits, dim=-1)[..., :num_classes]

        correct = 0
        total = 0

        for batch_idx in range(student_logits.shape[0]):
            student_scores = student_probs[batch_idx].max(dim=-1).values
            teacher_scores = teacher_probs[batch_idx].max(dim=-1).values

            student_idx = torch.topk(student_scores, k=min(student_topk, student_scores.numel()), sorted=False).indices
            teacher_idx = torch.topk(teacher_scores, k=min(teacher_topk, teacher_scores.numel()), sorted=False).indices

            if student_idx.numel() == 0 or teacher_idx.numel() == 0:
                continue

            sp = student_probs[batch_idx, student_idx]
            tp = teacher_probs[batch_idx, teacher_idx]
            sb = student_boxes[batch_idx, student_idx]
            tb = teacher_boxes[batch_idx, teacher_idx]

            teacher_soft = tp[:, None, :]
            student_soft = sp[None, :, :].clamp(1e-6, 1.0 - 1e-6)

            cls_cost = -(teacher_soft * torch.log(student_soft) + (1.0 - teacher_soft) * torch.log(1.0 - student_soft)).mean(dim=-1)
            l1_cost = torch.cdist(tb, sb, p=1)
            giou_cost = -generalized_box_iou(self._cxcywh_to_xyxy(tb), self._cxcywh_to_xyxy(sb))

            cost = cls_cost + 5.0 * l1_cost + 2.0 * giou_cost

            teacher_match, student_match = linear_sum_assignment(cost.cpu().numpy())

            teacher_match = torch.as_tensor(teacher_match, device=teacher_logits.device)
            student_match = torch.as_tensor(student_match, device=student_logits.device)

            matched_teacher_classes = tp[teacher_match].argmax(dim=-1)
            matched_student_classes = sp[student_match].argmax(dim=-1)

            correct += (matched_teacher_classes == matched_student_classes).sum().item()
            total += matched_teacher_classes.numel()

        return correct, total


    def _cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)

    @torch.no_grad()
    def _report_detection_samples(self, epoch: int) -> None:

        if self.metrics_callback_debug is None or not self.debug_indices:
            return
        # Последняя эпоха отправляется всегда: иначе итог прогона зависел бы от
        # того, кратно ли число эпох периоду.
        if not (self.debug_every_n_epochs > 0 and (epoch % self.debug_every_n_epochs == 0 or epoch == self.epochs)):
            return

        dataset = self.eval_loader.dataset
        indices = self.debug_indices

        samples = [dataset[i] for i in indices]

        images_cpu = [image for image, _ in samples]
        targets = [target for _, target in samples]

        images = torch.stack(images_cpu, dim=0).to(self.device)

        was_training = self.student.training
        self.student.eval()
        
        outputs = self.student(images)

        if self.prediction_postprocessor is not None:
            predictions = self.prediction_postprocessor(outputs, images)
        else:
            predictions = outputs

        for number, (image, target, prediction) in enumerate(zip(images_cpu, targets, predictions)):
            debug_image = visualize_detection(
                image=image,
                target=target,
                prediction=prediction,
                label_to_name=dataset.label_to_name,
                mean=self.normalize[0] if self.normalize else None,
                std=self.normalize[1] if self.normalize else None,
                score_threshold=0.3,
            )

            self.metrics_callback_debug(
                image=debug_image,
                series=f"sample_{number}",
                iteration=epoch,
            )

        if was_training:
            self.student.train()

    def load_checkpoint(self, path: Path) -> None:
        """Восстанавливает student/criterion/optimizer/scheduler/scaler из
        чекпоинта и выставляет эпоху и лучший mAP, с которых продолжать fit().
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        unwrap(self.student).load_state_dict(checkpoint["student_state"])
        unwrap(self.criterion).load_state_dict(checkpoint["criterion_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.scheduler is not None and checkpoint.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_map = checkpoint["best_map"]
        self.best_epoch = checkpoint.get("best_epoch", checkpoint["epoch"])

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        best_map: float,
        best_epoch: int,
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "best_map": best_map,
            "best_epoch": best_epoch,
            # unwrap: под DDP ключи иначе ушли бы с префиксом "module." (сейчас
            # для detection DDP запрещён выше по стеку, так что unwrap() здесь
            # no-op, но так чекпоинт останется совместим, если это снимут).
            "student_state": unwrap(self.student).state_dict(),
            "criterion_state": unwrap(self.criterion).state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "scaler_state": self.scaler.state_dict(),
        }
        torch.save(checkpoint, self.output_dir / filename)


@torch.no_grad()
def segmentation_evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
    limit_batches: int | None = None,
    amp: bool = False,
) -> tuple[float, IoUAccumulator]:
    """Возвращает (средний CE-лосс, аккумулятор IoU).

    Отдаётся именно аккумулятор, а не готовый {miou, pixel_acc}: из него, кроме
    скаляров, достаются per-class IoU и матрица ошибок для графиков, и всё это
    без второго прохода по валидации.

    Лосс считается обычной кросс-энтропией, а не self.criterion: на валидации
    интересна метрика самой модели, а не значение дистилляционного лосса,
    которое к тому же требовало бы учителя.

    amp здесь обязан совпадать с обучением: валидация идёт в полном разрешении
    1024x2048, а у U-Net skip-связи живут до самого декодера и не освобождаются
    по ходу forward'а — в fp32 это лишний двукратный расход памяти на пустом месте.
    Сам лосс всё равно считается в fp32 (logits.float()).

    Под DDP модель приходит уже развёрнутой (без обёртки), а результаты
    сводятся по всем процессам ПОСЛЕ цикла. Внутри цикла коллективных операций
    быть не должно: eval идёт по ShardSampler без дополнения дубликатами, у
    ранков разное число батчей, и они разошлись бы в числе операций.

    mIoU при этом совпадает с однопроцессным прогоном точно — матрица
    аддитивна. Лосс может разойтись в последних знаках: он взвешен по
    картинкам батча, а границы батчей у шардов другие.
    """
    was_training = model.training
    model.eval()

    loss_meter = AverageMeter()
    iou = IoUAccumulator(num_classes, device, ignore_index)

    for step, (images, masks) in enumerate(loader):
        if limit_batches is not None and step >= limit_batches:
            break

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(device.type, enabled=amp):
            logits = model(images)

        loss = F.cross_entropy(logits.float(), masks, ignore_index=ignore_index)

        loss_meter.update(loss.item(), images.size(0))
        iou.update(logits.argmax(dim=1), masks)

    if was_training:
        model.train()

    sync_meters({"loss": loss_meter}, device)
    iou.synchronize()

    return loss_meter.avg, iou


class SegmentationTrainer:
    """Тренер семантической сегментации.

    Отличия от Trainer, ради которых он отдельный:
    - цель — mIoU, а не accuracy; лучший чекпоинт выбирается по нему;
    - метрика копится пиксельной confusion matrix с выбрасыванием ignore_index;
    - таргет — карта [B, H, W], а не вектор меток.

    Учитель здесь опционален и не используется до появления дистилляционного
    лосса — интерфейс оставлен тем же, чтобы этап KD не потребовал правок.
    """

    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module | None,
        criterion: DistillationLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        num_classes: int,
        dist: DistInfo,
        output_dir: Path,
        epochs: int,
        amp: bool = False,
        grad_clip_norm: float | None = None,
        limit_train_batches: int | None = None,
        limit_eval_batches: int | None = None,
        save_best: bool = True,
        save_last: bool = True,
        progress_bar: bool = True,
        find_unused_parameters: bool = False,
        broadcast_buffers: bool = True,
        metrics_callback: tuple[Callable, ...] | None = None,
        scalars: dict[str, float | int | str],
        ignore_index: int = 255,
        batch_augment: Callable | None = None,
        plots: dict | None = None,
        class_names: Sequence[str] | None = None,
        palette: Sequence[Sequence[int]] | None = None,
        normalize: tuple[Sequence[float], Sequence[float]] | None = None,
    ) -> None:
        if criterion.requires_teacher and teacher is None:
            raise ValueError(
                f"Лосс {type(criterion).__name__} требует учителя, но model/teacher=null. "
                f"Либо задай учителя, либо возьми loss=ce."
            )

        self.student = student
        self.teacher = teacher
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dist = dist
        self.device = dist.device
        self.find_unused_parameters = find_unused_parameters
        self.broadcast_buffers = broadcast_buffers
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.grad_clip_norm = grad_clip_norm
        self.limit_train_batches = limit_train_batches
        self.limit_eval_batches = limit_eval_batches
        self.save_best = save_best
        self.save_last = save_last
        self.progress_bar = progress_bar
        # Mixup/CutMix по батчу. Для сегментации рабочий вариант — CutMix:
        # он переносит вместе с куском изображения и кусок маски, поэтому
        # таргет остаётся точным (см. src/data/batch_augment.py).
        self.batch_augment = batch_augment
        self.metrics_callback_scalar = metrics_callback[0] if metrics_callback is not None else None
        self.metrics_callback_single = metrics_callback[1] if metrics_callback is not None else None
        self.metrics_callback_table = metrics_callback[2] if metrics_callback is not None else None
        self.metrics_callback_plots = metrics_callback[3] if metrics_callback is not None else None
        self.scalars = scalars
        self.class_names = list(class_names) if class_names is not None else None

        plots = dict(plots) if plots is not None else {}
        # Графики-картинки (per-class IoU, матрица ошибок, распределения) строятся
        # из уже накопленных агрегатов, поэтому почти бесплатны. Дорогая часть —
        # только сравнение с учителем, и её объём задаётся probe_batches/probe_pixels.
        self.plots_enabled = bool(plots.get("enabled", True)) and self.metrics_callback_plots is not None
        self.plots_every_n_epochs = int(plots.get("every_n_epochs", 5))
        self.probe_batches = int(plots.get("probe_batches", 20))
        self.debug_every_n_epochs = int(plots.get("debug_every_n_epochs", 10))
        self.debug_width = int(plots.get("debug_width", 512))
        self.palette = list(palette) if palette is not None else default_palette(self.num_classes)
        self.normalize = normalize
        # Картинки для Debug Samples берутся с равным шагом по валидации и
        # фиксируются на весь прогон: Cityscapes отсортирован по городам, так
        # что первые N подряд — это N кадров одной улицы, а случайные каждый
        # раз не с чем сравнивать. Одни и те же кадры на каждой отправке —
        # единственный способ увидеть, как меняется предсказание.
        self.debug_indices = self._pick_debug_indices(int(plots.get("debug_samples", 4)))

        self.train_iou = IoUAccumulator(self.num_classes, self.device, self.ignore_index)

        # Схожесть с учителем — то же, что KL/agreement в классификации, но по
        # пикселям. Включается теми же ключами в clearml.scalars и, разумеется,
        # только когда учитель вообще есть.
        self.similarity: TeacherSimilarity | None = None
        if teacher is not None and {"KL_divergence", "agreement_rate"} & set(self.scalars):
            self.similarity = TeacherSimilarity(
                self.num_classes,
                self.device,
                self.ignore_index,
                pixels_per_batch=int(plots.get("probe_pixels", 8192)),
            )

        self.amp_enabled = amp and self.device.type == "cuda"
        self.scaler = GradScaler(self.device.type, enabled=self.amp_enabled)

        self.student.to(self.device)
        self.criterion.to(self.device)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.teacher is not None:
            self.teacher.to(self.device)
            self.teacher.eval()
            self.teacher.requires_grad_(False)

        if self.metrics_callback_table is not None:
            self.metrics_callback_table(build_param_table(self.student, self.teacher, self.criterion))

        criterion_params = list(self.criterion.parameters())
        if criterion_params:
            optimizer_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
            missing = [p for p in criterion_params if id(p) not in optimizer_params]
            if missing:
                raise ValueError(
                    "У лосса есть обучаемые параметры (адаптеры), не попавшие в optimizer. "
                    "Optimizer должен собираться из student.parameters() + criterion.parameters()."
                )

        self.student_extractor: FeatureExtractor | None = None
        self.teacher_extractor: FeatureExtractor | None = None
        if criterion.required_features:
            layers = list(criterion.required_features)
            self.student_extractor = FeatureExtractor(self.student, layers)
            if self.teacher is not None:
                self.teacher_extractor = FeatureExtractor(self.teacher, layers)

        self.student = wrap_ddp(
            self.student, self.dist, self.find_unused_parameters, self.broadcast_buffers
        )

        if any(parameter.requires_grad for parameter in self.criterion.parameters()):
            self.criterion = wrap_ddp(
                self.criterion, self.dist, self.find_unused_parameters, self.broadcast_buffers
            )
        # учителя не оборачиваем в DDP, т.к. синхронизация между процессами не нужна

    def fit(self) -> dict:
        history = MetricsHistory(self.output_dir / "history.csv") if self.dist.is_main else None
        best_miou, best_epoch = 0.0, 0

        # Срез до первого шага (iteration 0): у необученной модели предсказание
        # шумовое, и именно с ним потом сравниваются все последующие эпохи.
        if self.plots_enabled and self.debug_indices:
            self.metrics_callback_plots(self._debug_samples(), 0)

        try:
            for epoch in range(1, self.epochs + 1):
                start = time.time()
                lr = self.optimizer.param_groups[0]["lr"]

                train_loss_components, other_train_metrics, norms = self._train_epoch(epoch)
                eval_loss, eval_iou = segmentation_evaluate(
                    # Именно развёрнутая модель: forward через DDP-обёртку —
                    # коллективная операция, а число батчей у ранков на eval
                    # разное (ShardSampler), и они бы разошлись.
                    model=unwrap(self.student),
                    loader=self.eval_loader,
                    device=self.device,
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index,
                    limit_batches=self.limit_eval_batches,
                    amp=self.amp_enabled,
                )
                eval_metrics = eval_iou.compute()

                if self.scheduler is not None:
                    self.scheduler.step()

                all_values = {
                    "epoch": epoch,
                    "lr": lr,
                    "world_size": self.dist.world_size,
                    "global_batch_size": (self.train_loader.batch_size or 0) * self.dist.world_size,
                    "eval_loss": eval_loss,
                    "eval_miou": eval_metrics["miou"],
                    "eval_pixel_acc": eval_metrics["pixel_acc"],
                    "time_epoch": round(time.time() - start, 1),
                    **{f"train_loss_{key}": value for key, value in train_loss_components.items()},
                    **{f"train_{key}": value for key, value in other_train_metrics.items()},
                }

                if history is not None:
                    history.append(all_values)
                if self.metrics_callback_scalar is not None:
                    self.metrics_callback_scalar(all_values)
                if self.plots_enabled:
                    self.metrics_callback_plots(self._build_plots(epoch, eval_iou, norms), epoch)

                is_best = eval_metrics["miou"] > best_miou
                if is_best:
                    best_miou, best_epoch = eval_metrics["miou"], epoch
                    if self.save_best:
                        self._save_checkpoint("best.pt", epoch, best_miou)
                if self.save_last:
                    self._save_checkpoint("last.pt", epoch, best_miou)

                log.info(
                    "Эпоха %02d/%d | lr=%.6f | train loss=%.4f | train mIoU=%.4f | "
                    "eval loss=%.4f | eval mIoU=%.4f | pixel acc=%.4f%s | %.1f c",
                    epoch,
                    self.epochs,
                    lr,
                    train_loss_components["total"],
                    other_train_metrics["miou"],
                    eval_loss,
                    eval_metrics["miou"],
                    eval_metrics["pixel_acc"],
                    " *" if is_best else "",
                    all_values["time_epoch"],
                )
        finally:
            if self.student_extractor is not None:
                self.student_extractor.remove()
            if self.teacher_extractor is not None:
                self.teacher_extractor.remove()

        log.info("Лучший mIoU: %.4f (эпоха %d)", best_miou, best_epoch)
        return {"best_miou": best_miou, "best_epoch": best_epoch}

    def _train_epoch(self, epoch: int) -> tuple:
        self.student.train()
        self.criterion.train()

        # Без set_epoch DistributedSampler выдаёт одну и ту же перестановку
        # каждую эпоху, то есть порядок данных перестаёт меняться.
        sampler = getattr(self.train_loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        norms = NormTracker()

        self.train_iou.reset()
        if self.similarity is not None:
            self.similarity.reset()
        # Сравнение с учителем считается не на каждом шаге: за эпоху достаточно
        # probe_batches замеров, чтобы среднее устоялось, а гистограмма набрала
        # форму. Шаги берутся с равным интервалом по эпохе — иначе метрика
        # описывала бы только её начало.
        probe_every = self._probe_interval()

        meters_avg: dict[str, AverageMeter] = defaultdict(AverageMeter)
        meters_avg_loss: dict[str, AverageMeter] = defaultdict(AverageMeter)
        meters_other: dict[str, float | int | str] = {}

        iterator = tqdm(
            self.train_loader,
            desc=f"Эпоха {epoch}/{self.epochs}",
            disable=not self.progress_bar or not self.dist.is_main,
            leave=False,
        )

        for step, (images, masks) in enumerate(iterator):
            if self.limit_train_batches is not None and step >= self.limit_train_batches:
                iterator.close()
                break

            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            batch_size = images.size(0)

            # До прогона учителя: он обязан видеть ту же склейку, что и ученик.
            mixed = mix_batch(self.batch_augment, images, masks)
            images, masks = mixed.images, mixed.targets_a

            self.optimizer.zero_grad(set_to_none=True)
            if self.student_extractor is not None:
                self.student_extractor.clear()
            if self.teacher_extractor is not None:
                self.teacher_extractor.clear()

            teacher_logits = None
            if self.teacher is not None:
                with torch.no_grad(), torch.autocast(self.device.type, enabled=self.amp_enabled):
                    teacher_logits = self.teacher(images)

            with torch.autocast(self.device.type, enabled=self.amp_enabled):
                student_logits = self.student(images)
                losses = compute_losses(
                    self.criterion,
                    student_logits,
                    teacher_logits,
                    mixed,
                    student_features=(
                        self.student_extractor.features if self.student_extractor else None
                    ),
                    teacher_features=(
                        self.teacher_extractor.features if self.teacher_extractor else None
                    ),
                )

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)

            params = [p for group in self.optimizer.param_groups for p in group["params"]]
            clip_threshold = self.grad_clip_norm if self.grad_clip_norm is not None else float("inf")
            grad_norm = torch.nn.utils.clip_grad_norm_(params, clip_threshold)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            norms.update(grad_norm, params)

            if (
                self.similarity is not None
                and teacher_logits is not None
                and step % probe_every == 0
            ):
                self.similarity.update(student_logits, teacher_logits, masks)

            for key, value in losses.items():
                meters_avg_loss[key].update(value.item(), batch_size)

            with torch.no_grad():
                # masks — это targets_a. Для CutMix маска точная (кусок перенесён
                # вместе с пикселями), поэтому train mIoU остаётся честным;
                # для Mixup он превращается в оценку снизу.
                self.train_iou.update(student_logits.detach().argmax(dim=1), masks)

            iterator.set_postfix({"loss": f"{losses['total'].item():.3f}"})

        # Сведение по процессам — здесь, после цикла: число шагов у ранков
        # одинаково (DistributedSampler дополняет выборку), поэтому в саму
        # эпоху коллективные операции добавлять не нужно.
        sync_meters(meters_avg_loss, self.device)
        sync_meters(meters_avg, self.device)
        self.train_iou.synchronize()
        norms.synchronize(self.device)
        if self.similarity is not None:
            self.similarity.synchronize()

        train_loss_components = {key: meter.avg for key, meter in meters_avg_loss.items()}

        meters_other.update(self.train_iou.compute())  # miou, pixel_acc
        meters_other.update(norms.results())
        if self.similarity is not None:
            meters_other.update(self.similarity.compute())  # KL_divergence, agreement_rate

        other_train_metrics = {**{key: meter.avg for key, meter in meters_avg.items()}, **meters_other}

        return train_loss_components, other_train_metrics, norms

    def _pick_debug_indices(self, count: int) -> list[int]:
        """Равномерно разбросанные по валидации индексы картинок для Debug Samples."""
        if count <= 0:
            return []
        try:
            total = len(self.eval_loader.dataset)
        except TypeError:  # IterableDataset — индексов нет
            return []
        count = min(count, total)
        if count == 0:
            return []
        if count == 1:
            return [0]
        return [round(i * (total - 1) / (count - 1)) for i in range(count)]

    @torch.no_grad()
    def _debug_samples(self) -> list[Plot]:
        """[кадр | разметка | предсказание | ошибки] по фиксированным картинкам валидации.

        Картинки прогоняются по одной: их единицы, а полный кадр 1024x2048
        под U-Net с его skip-связями — не тот тензор, который стоит собирать
        в батч ради экономии сотых долей секунды.

        Под DDP сюда заходит только главный ранк (у остальных нет колбэка
        графиков), поэтому модель берётся развёрнутой: forward через
        DDP-обёртку на одном ранке из нескольких — коллективная операция,
        которую остальные не сделают.
        """
        dataset = self.eval_loader.dataset
        student = unwrap(self.student)
        was_training = student.training
        student.eval()

        plots: list[Plot] = []
        try:
            for order, index in enumerate(self.debug_indices):
                image, mask = dataset[index]
                image = image.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                with torch.autocast(self.device.type, enabled=self.amp_enabled):
                    logits = student(image[None])
                prediction = logits.float().argmax(dim=1)[0]

                panel = prediction_panel(
                    image,
                    mask,
                    prediction,
                    palette=self.palette,
                    mean=self.normalize[0] if self.normalize else None,
                    std=self.normalize[1] if self.normalize else None,
                    ignore_index=self.ignore_index,
                    max_width=self.debug_width,
                )
                plot = image_plot("predictions", f"sample_{order}", panel)
                if plot is not None:
                    plots.append(plot)
        finally:
            if was_training:
                student.train()
            # Хуки лосса сработали и на этих кадрах: карты признаков полного
            # разрешения незачем держать до следующего шага обучения.
            if self.student_extractor is not None:
                self.student_extractor.clear()

        return plots

    def _probe_interval(self) -> int:
        """Через сколько шагов делать замер схожести с учителем."""
        if self.probe_batches <= 0:
            return 1
        try:
            steps = len(self.train_loader)
        except TypeError:  # IterableDataset — длина неизвестна заранее
            return 20
        if self.limit_train_batches is not None:
            steps = min(steps, self.limit_train_batches)
        return max(1, steps // self.probe_batches)

    def _is_due(self, epoch: int, period: int) -> bool:
        """Пора ли слать периодический график. Последняя эпоха — всегда.

        Иначе итог прогона зависел бы от того, кратно ли число эпох периоду:
        на 100 эпохах с периодом 7 последний срез оказался бы на 98-й.
        """
        return period > 0 and (epoch % period == 0 or epoch == self.epochs)

    def _build_plots(self, epoch: int, eval_iou: IoUAccumulator, norms: NormTracker) -> list[Plot]:
        """Графики эпохи.

        Всё, кроме Debug Samples, строится из уже накопленных агрегатов — второго
        прохода по данным нет. Картинки требуют форварда по нескольким кадрам,
        поэтому идут по своему, более редкому расписанию.
        """
        names = self.class_names
        # Матрица ошибок — 361 число и самая тяжёлая карточка в UI, шлём реже.
        heavy = self._is_due(epoch, self.plots_every_n_epochs)

        candidates = [
            bar_plot(
                "per_class_iou", "train",
                self.train_iou.per_class_iou().cpu().numpy(), names,
                xaxis="class", yaxis="IoU",
            ),
            bar_plot(
                "per_class_iou", "eval",
                eval_iou.per_class_iou().cpu().numpy(), names,
                xaxis="class", yaxis="IoU",
            ),
            distribution_plot(
                "grad_norm_distribution", "train",
                norms.grad_values, xaxis="grad norm (before clipping)",
            ),
        ]

        if heavy:
            candidates.append(
                matrix_plot(
                    "confusion_matrix", "eval",
                    eval_iou.normalized_matrix().cpu().numpy(), names,
                    xaxis="predicted", yaxis="ground truth",
                )
            )

        if self.similarity is not None and self.similarity.compute():
            candidates += [
                bar_plot(
                    "teacher_agreement_per_class", "train",
                    self.similarity.per_class_agreement(), names,
                    xaxis="class", yaxis="agreement rate",
                ),
                distribution_plot(
                    "teacher_kl_distribution", "train",
                    self.similarity.kl_sample_values(), xaxis="per-pixel KL(teacher || student)",
                ),
            ]

        plots = [plot for plot in candidates if plot is not None]

        if self.debug_indices and self._is_due(epoch, self.debug_every_n_epochs):
            plots += self._debug_samples()

        return plots

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        best_miou: float,
    ) -> None:
        if not self.dist.is_main:
            return

        checkpoint = {
            "epoch": epoch,
            "best_miou": best_miou,
            "world_size": self.dist.world_size,
            # unwrap: под DDP ключи иначе ушли бы с префиксом "module.",
            # и чекпоинт не встал бы в обычную модель.
            "student_state": unwrap(self.student).state_dict(),
            "criterion_state": unwrap(self.criterion).state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "scaler_state": self.scaler.state_dict(),
        }
        torch.save(checkpoint, self.output_dir / filename)
