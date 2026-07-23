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
from torch.amp.grad_scaler import GradScaler
import torch.nn.functional as F
from tqdm import tqdm

from src.losses.base import DistillationLoss
from src.models.feature_extractor import FeatureExtractor
from src.utils.logger import MetricsHistory, get_logger
from src.utils.metrics import AverageMeter, accuracy, ConfusionMatrixAccumulator, count_parameters, build_param_table

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
        loss = F.cross_entropy(logits, labels)

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
        metrics_callback: tuple[Callable, Callable, Callable] | None = None,
        scalars: dict[str, float | int | str],
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
        self.metrics_callback_scalar = metrics_callback[0] if metrics_callback is not None else None
        self.metrics_callback_single = metrics_callback[1] if metrics_callback is not None else None
        self.metrics_callback_table = metrics_callback[2] if metrics_callback is not None else None
        self.scalars = scalars
        # metrics = {'loss.train_loss': }
        self.train_confmat: ConfusionMatrixAccumulator | None = None
        if {"precision", "recall", "f1"} & set(self.scalars):
            self.train_confmat = ConfusionMatrixAccumulator(self.num_classes, self.device)
        


        # AMP имеет смысл только на CUDA; на CPU молча работаем в fp32.
        self.amp_enabled = amp and device.type == "cuda"
        self.scaler = GradScaler(device.type, enabled=self.amp_enabled)

        if self.teacher is not None:
            self.teacher.eval()
            self.teacher.requires_grad_(False)

        parameters_names = ["total_parameters_in-thousands", "trainable_parameters_in-thousands", "buffers_parameters_in-thousands", "size_in-MiB"]
        #total_student_parameters, trainable_student_parameters, buffers_student_parameters
        if self.metrics_callback_single is not None:
            for role, model in (("Student", self.student), ("Criterion", self.criterion)):
                stats = count_parameters(model)
                self.metrics_callback_single({f"{role}_{k}": v for k, v in stats.items()})
            if self.teacher is not None:
                stats = count_parameters(self.teacher)
                self.metrics_callback_single({f"Teacher_{k}": v for k, v in stats.items()})

        if self.metrics_callback_table is not None:
            self.metrics_callback_table(build_param_table(self.student, self.teacher, self.criterion))
        """
        if self.metrics_callback_single is not None:
            student_count_parameters = count_parameters(self.student)
            criterion_count_parameters = count_parameters(self.criterion)
            self.metrics_callback_single({f'Student_'+parameters_names[i]: student_count_parameters[i] for i in range(len(parameters_names))})
            self.metrics_callback_single({f'Criterion_'+parameters_names[i]: criterion_count_parameters[i] for i in range(len(parameters_names))})
            if self.teacher is not None:
                teacher_count_parameters = count_parameters(self.teacher)
                self.metrics_callback_single({f'Teacher_'+parameters_names[i]: teacher_count_parameters[i] for i in range(len(parameters_names))})
        """

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

    def fit(self) -> dict:
        history = MetricsHistory(self.output_dir / "history.csv")
        best_acc, best_epoch = 0.0, 0

        try:
            for epoch in range(1, self.epochs + 1):

                start = time.time()
                lr=self.optimizer.param_groups[0]["lr"]

                train_loss_components, other_train_metrics = self._train_epoch(epoch)
                eval_loss, eval_acc = evaluate(
                    self.student, self.eval_loader, self.device, self.limit_eval_batches
                )
                if self.scheduler is not None:
                    self.scheduler.step()

                all_values = {
                    "epoch": epoch,
                    "lr": lr,
                    "eval_loss": eval_loss,
                    "eval_acc": eval_acc,
                    "time_epoch": round(time.time() - start, 1),
                    **{f"train_loss_{key}": value for key, value in train_loss_components.items()},
                    **{f"train_{key}": value for key, value in other_train_metrics.items()},
                }

                print('ВСЕ ЗНАЧЕНИЯ = ', all_values)

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

        max_grad_norm = -1
        max_weight_norm = -1

        if self.train_confmat is not None:
            self.train_confmat.reset()

        meters_avg: dict[str, AverageMeter] = defaultdict(AverageMeter)
        meters_avg_loss: dict[str, AverageMeter] = defaultdict(AverageMeter)
        meters_other: dict[str, float | int | str] = {}
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
            if self.teacher_extractor is not None:
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


            grad_norm_value = grad_norm.item()
            meters_avg["avg_grad_norm"].update(grad_norm_value, n=1)    # средняя градиент
            if grad_norm_value > max_grad_norm:
                max_grad_norm = grad_norm_value # максимальный градиент
            
            with torch.no_grad():
                weight_norm = torch.norm(torch.stack([p.detach().norm() for p in params]))
            weight_norm_value = weight_norm.item()
            meters_avg["avg_weight_norm"].update(weight_norm_value, n=1)
            if weight_norm_value > max_weight_norm: 
                max_weight_norm = weight_norm_value

            for key, value in losses.items():
                meters_avg_loss[key].update(value.item(), batch_size)

            meters_avg["acc"].update(accuracy(student_logits.float(), labels), batch_size) #train_acc

            probs = torch.softmax(student_logits.detach().float(), dim=1)

            if self.train_confmat is not None:
                self.train_confmat.update(probs.argmax(dim=1), labels)

            iterator.set_postfix({"loss": f"{losses['total'].item():.3f}"})

        if self.train_confmat is not None:
            meters_other.update(self.train_confmat.compute()) # pr, rec, F1

        train_loss_components = {key: meter.avg for key, meter in meters_avg_loss.items()} # train_loss_components

        meters_other["max_grad_norm"] = max_grad_norm
        meters_other["max_weight_norm"] = max_weight_norm

        other_train_metrics = {**{key: meter.avg for key, meter in meters_avg.items()}, **meters_other}

        return train_loss_components, other_train_metrics

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
