"""Обобщённый цикл обучения с дистилляцией и без.

Один Trainer покрывает все режимы; чем именно он занят, определяют декларации
лосса (requires_teacher, required_features) и конфиг:
- ученик с нуля:            teacher=None,  loss=CrossEntropy
- ванильная KD (Хинтон):    teacher=model, loss=HintonKD
- feature-based KD:         teacher=model, loss=FeatureKD (хуки + адаптеры)
"""

import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses.base import DistillationLoss
from src.models.feature_extractor import FeatureExtractor
from src.utils.logger import MetricsHistory, get_logger
from src.utils.metrics import AverageMeter, accuracy

log = get_logger(__name__)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit_batches: int | None = None,
) -> tuple[float, float]:
    """Возвращает (средний CE-лосс, точность) на выборке. Всегда в fp32."""
    was_training = model.training
    model.eval()

    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    for step, (images, labels) in enumerate(loader):
        if limit_batches is not None and step >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        batch_size = labels.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy(logits, labels), batch_size)

    if was_training:
        model.train()
    return loss_meter.avg, acc_meter.avg


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
        metrics_callback: Callable[[dict], None] | None = None,
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
        self.device = device
        self.output_dir = Path(output_dir)

        self.epochs = epochs
        self.grad_clip_norm = grad_clip_norm
        self.limit_train_batches = limit_train_batches
        self.limit_eval_batches = limit_eval_batches
        self.save_best = save_best
        self.save_last = save_last
        self.progress_bar = progress_bar
        # Точка стыковки внешнего трекера (ClearML и т.п.): вызывается после
        # каждой эпохи со строкой метрик — той же, что уходит в history.csv.
        # Trainer ничего не знает о трекере, колбэк собирает scripts/train.py.
        self.metrics_callback = metrics_callback

        # AMP имеет смысл только на CUDA; на CPU молча работаем в fp32.
        self.amp_enabled = amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp_enabled)

        if self.teacher is not None:
            self.teacher.eval()
            self.teacher.requires_grad_(False)

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
            self.teacher_extractor = FeatureExtractor(self.teacher, layers)

    def fit(self) -> dict:
        history = MetricsHistory(self.output_dir / "history.csv")
        best_acc, best_epoch = 0.0, 0

        try:
            for epoch in range(1, self.epochs + 1):
                start = time.time()
                lr = self.optimizer.param_groups[0]["lr"]

                train_stats = self._train_epoch(epoch)
                eval_loss, eval_acc = evaluate(
                    self.student, self.eval_loader, self.device, self.limit_eval_batches
                )
                if self.scheduler is not None:
                    self.scheduler.step()

                row = {
                    "epoch": epoch,
                    "lr": lr,
                    **{f"train_{key}": value for key, value in train_stats.items()},
                    "eval_loss": eval_loss,
                    "eval_acc": eval_acc,
                    "time_sec": round(time.time() - start, 1),
                }
                history.append(row)
                if self.metrics_callback is not None:
                    self.metrics_callback(row)

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
                    train_stats["total"],
                    train_stats["acc"] * 100,
                    eval_loss,
                    eval_acc * 100,
                    " *" if is_best else "",
                    row["time_sec"],
                )
        finally:
            if self.student_extractor is not None:
                self.student_extractor.remove()
                self.teacher_extractor.remove()

        log.info("Лучшая точность: %.2f%% (эпоха %d)", best_acc * 100, best_epoch)
        return {"best_acc": best_acc, "best_epoch": best_epoch}

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.student.train()
        self.criterion.train()

        meters: dict[str, AverageMeter] = defaultdict(AverageMeter)
        iterator = tqdm(
            self.train_loader,
            desc=f"Эпоха {epoch}/{self.epochs}",
            disable=not self.progress_bar,
            leave=False,
        )

        for step, (images, labels) in enumerate(iterator):
            if self.limit_train_batches is not None and step >= self.limit_train_batches:
                iterator.close()
                break
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            batch_size = labels.size(0)

            self.optimizer.zero_grad(set_to_none=True)
            if self.student_extractor is not None:
                self.student_extractor.clear()
                self.teacher_extractor.clear()

            teacher_logits = None
            if self.teacher is not None:
                with torch.no_grad(), torch.autocast(self.device.type, enabled=self.amp_enabled):
                    teacher_logits = self.teacher(images)

            with torch.autocast(self.device.type, enabled=self.amp_enabled):
                student_logits = self.student(images)
                losses = self.criterion(
                    student_logits,
                    teacher_logits,
                    labels,
                    student_features=(
                        self.student_extractor.features if self.student_extractor else None
                    ),
                    teacher_features=(
                        self.teacher_extractor.features if self.teacher_extractor else None
                    ),
                )

            self.scaler.scale(losses["total"]).backward()
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                params = [p for group in self.optimizer.param_groups for p in group["params"]]
                torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for key, value in losses.items():
                meters[key].update(value.item(), batch_size)
            meters["acc"].update(accuracy(student_logits.float(), labels), batch_size)

            iterator.set_postfix({"loss": f"{losses['total'].item():.3f}"})

        return {key: meter.avg for key, meter in meters.items()}

    def _save_checkpoint(self, filename: str, epoch: int, best_acc: float) -> None:
        checkpoint = {
            "epoch": epoch,
            "best_acc": best_acc,
            "student_state": self.student.state_dict(),
            # Состояние лосса = адаптеры каналов (у CrossEntropy/HintonKD пусто).
            "criterion_state": self.criterion.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict(),
        }
        torch.save(checkpoint, self.output_dir / filename)
