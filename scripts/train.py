"""Единая точка входа обучения.

Артефакты запуска (конфиг, логи, history.csv, чекпоинты) складываются
в outputs/<name>/<дата_время>/.
"""

import logging
import math
from pathlib import Path

import torch

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn

from src.data import base_loader
from src.models import ChunkedTeacher, MultiScaleInference, NativeResolutionTeacher
from src.training import (
    DetectionTrainer,
    LossWeightScheduler,
    SegmentationTrainer,
    Trainer,
    build_param_groups,
    describe_param_groups,
)
from src.utils import distributed, resolve_device, seed_everything
from src.utils.distributed import DistInfo

log = logging.getLogger(__name__)

BEST_METRIC_KEY = {
    "classification": "best_acc",
    "detection": "best_map",
    "segmentation": "best_miou",
}

def _plain(value):
    """DictConfig/ListConfig -> dict/list, остальное как есть.

    ClearML проверяет тип аргументов строго (`isinstance(x, dict)`), а контейнеры
    omegaconf от dict/list не наследуются. Непреобразованный DictConfig в
    auto_connect_frameworks молча трактуется как "истина", то есть настройка
    {"pytorch": false} не срабатывала бы вовсе.
    """
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _normalize_stats(dataset_cfg: DictConfig):
    """(mean, std) из конфига датасета или None, если нормировки нет."""
    normalize = dataset_cfg.get("normalize")
    if normalize is None:
        return None
    return list(normalize.mean), list(normalize.std)


def build_optimizer_params(cfg: DictConfig, student: nn.Module, criterion: nn.Module):
    """Параметры для optimizer: плоский список (по умолчанию) или группы
    с раздельным lr/weight_decay (cfg.param_groups, см. docs/param_groups.md).

    Обучаемые параметры лосса (адаптеры FitNets, проекторы HeteroAKD) идут
    вместе со студентом в обоих случаях — SegmentationTrainer/Trainer в
    конструкторе проверяют, что КАЖДЫЙ параметр criterion попал в optimizer,
    и падают, если нет (иначе адаптер тихо не обучался бы).
    """
    if not cfg.get("param_groups"):
        return list(student.parameters()) + list(criterion.parameters())

    groups = build_param_groups(
        student,
        criterion,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.get("weight_decay", 0.0),
        **_plain(cfg.param_groups),
    )
    log.info("Группы параметров оптимизатора:\n%s", describe_param_groups(groups))
    return groups


def configure_rank_logging(dist: DistInfo) -> None:
    """
    Разводит логи ранков, чтобы они не мешали друг другу.
    """
    if dist.is_main:
        return

    root = logging.getLogger()
    for handler in list(root.handlers):
        # ищем обработчик, который пишет в файл
        if isinstance(handler, logging.FileHandler):
            path = Path(handler.baseFilename)
            root.removeHandler(handler)
            handler.close() # честно закрываем логгер
            # создаём личный файл для логгера, чтобы избежать гонки
            rank_handler = logging.FileHandler(
                path.with_name(f"{path.stem}.rank{dist.rank}{path.suffix}")
            )
            rank_handler.setFormatter(handler.formatter)
            rank_handler.setLevel(handler.level)
            root.addHandler(rank_handler)
        elif isinstance(handler, logging.StreamHandler):
            # в консоль от ранков будем выводить только WARNINGs
            handler.setLevel(logging.WARNING)


def resolve_output_dir(dist: DistInfo) -> Path:
    """Выбор output_dir, а также проверка
       на то, что ранки не расходятся
    """
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    main_dir = distributed.broadcast_object(str(output_dir), device=dist.device, src=0)
    if main_dir != str(output_dir):
        raise RuntimeError(
            f"Ранки разошлись в директории запуска: rank {dist.rank} получил "
            f"{output_dir}, а rank 0 — {main_dir}. Значит hydra.run.dir был "
            f"вычислен в каждом процессе отдельно. Проверь, что задана "
            f"переменная JUICER_RUN_ID или TORCHELASTIC_RUN_ID."
        )
    return output_dir


def init_clearml(cfg: DictConfig, dist: DistInfo):
    # таска создаётся только на главном ранке
    if not dist.is_main or not cfg.clearml.enabled:
        return None
    from clearml import Task

    # resume=true (detection и segmentation, см. DetectionTrainer.load_checkpoint /
    # SegmentationTrainer.load_checkpoint): без continue_last_task=True ClearML
    # завёл бы новый таск, и график в UI разъехался бы на "до крэша" и "после"
    # вместо одной линии.
    continue_last_task = (
        True
        if (cfg.resume and cfg.task_type in ("detection", "segmentation"))
        else cfg.clearml.continue_last_task
    )

    task = Task.init(
        project_name=cfg.clearml.project,                      # проект
        task_name=f"{cfg.name}",                               # имя таски
        task_type=Task.TaskTypes.training,                     # тип таски
        tags=_plain(cfg.clearml.tags),                         # теги
        reuse_last_task_id = cfg.clearml.reuse_last_task_id,   # перезаписывать ли таску с таким же именем
        continue_last_task=continue_last_task,                 # Подхватит предыдущий ID и продолжит логирование
        output_uri=cfg.clearml.output_uri,                     # складывать ли артефакты/модели и если куда-то базово, то url
        auto_connect_frameworks=_plain(cfg.clearml.auto_connect_frameworks), # авто-перехват фреймворков
        auto_connect_arg_parser=cfg.clearml.auto_connect_arg_parser, # авто-перехват аргументов из argparse
        auto_resource_monitoring=cfg.clearml.auto_resource_monitoring, # графики :monitor:gpu/:monitor:machine
    )

    # Полный разрешённый конфиг — в Configuration objects задачи.
    task.connect_configuration(OmegaConf.to_container(cfg, resolve=True), name="hydra_config")
    return task


def clearml_reporter(task):
    """
    title = график в UI, series = линия на нём
    """
    logger = task.get_logger()

    def report_scalar(row: dict, iteration_key: str = "epoch") -> None:
        if iteration_key not in row:
            return

        iterate = int(row[iteration_key])

        def _safe_report(title: str, series: str, val):
            """Логирует только конечные числа.

            Отсутствующие ключи (None) и inf/NaN отбрасываются молча: ClearML
            рисовать их не умеет и подменяет нулём с предупреждением
            "inf value encountered. Reporting it as '0.0'" — в UI это выглядит
            как настоящий ноль метрики и путает сильнее, чем разрыв линии.
            """
            if val is None:
                return
            try:
                val = float(val)
            except (TypeError, ValueError):
                return
            if math.isfinite(val):
                logger.report_scalar(title=title, series=series, value=val, iteration=iterate)

        # 1. Основные лоссы
        if "train_loss_total" in row:
            _safe_report("loss", "train", row["train_loss_total"])
        if "eval_loss" in row:
            _safe_report("loss", "eval", row["eval_loss"])

        # 2. Learning Rate
        if "lr" in row:
            _safe_report("lr", "lr", row["lr"])

        # 3. Метрики точности (Classification)
        if "train_acc" in row:
            _safe_report("accuracy", "train", row["train_acc"])
        if "eval_acc" in row:
            _safe_report("accuracy", "eval", row["eval_acc"])

        # 4. Precision / Recall / F1
        if "train_precision" in row:
            _safe_report("precision", "train", row["train_precision"])
            _safe_report("recall", "train", row["train_recall"])
            _safe_report("F1", "train", row["train_F1"])

        # 5. Дистилляция / KL
        if "train_KL_divergence" in row:
            _safe_report("KL", "train_KL", row["train_KL_divergence"])
        if "train_agreement_rate" in row.keys():
            _safe_report("agreement_rate", "train_agreement_rate", row["train_agreement_rate"])

        # 6. Нормы градиентов и весов
        _safe_report("grad_norm", "train_avg_grad_norm", row.get("train_avg_grad_norm"))
        _safe_report("grad_norm", "train_max_grad_norm", row.get("train_max_grad_norm"))
        _safe_report("weight_norm", "train_avg_weight_norm", row.get("train_avg_weight_norm"))
        _safe_report("weight_norm", "train_max_weight_norm", row.get("train_max_weight_norm"))

        # 6a. Здоровье AMP и скорость: сколько шагов эпохи GradScaler отбросил
        # из-за переполнения fp16 и сколько эпоха заняла секунд.
        _safe_report("amp_skipped_steps", "train", row.get("train_skipped_steps"))
        _safe_report("time_epoch", "seconds", row.get("time_epoch"))

        # 7. Компоненты лосса (ОДИН ЦИКЛ вместо двух)
        for key, value in row.items():
            # Логируем все отдельные компоненты лосса (например: train_loss_ce, train_loss_kd)
            if key.startswith("train_loss_") and key != "train_loss_total":
                series_name = key.removeprefix("train_loss_")
                # Вклады слагаемых композиции — то же, что компоненты, но
                # домноженное на веса (loss_contributions) и отнормированное
                # к единице (loss_shares). Именно по ним подбираются веса:
                # в loss_components лежат СЫРЫЕ значения, на которые вес
                # не влияет, и по ним не видно, кто тянет сумму.
                if series_name.endswith("_weighted"):
                    _safe_report(
                        "loss_contributions", series_name.removesuffix("_weighted"), value
                    )
                elif series_name.endswith("_share"):
                    _safe_report("loss_shares", series_name.removesuffix("_share"), value)
                else:
                    _safe_report("loss_components", series_name, value)
            # Веса слагаемых, если они меняются по расписанию: без этого
            # графика падение дистилляционного члена не отличить от того,
            # что ученик догнал учителя.
            elif key.startswith("loss_weight_"):
                _safe_report("loss_weights", key.removeprefix("loss_weight_"), value)

        # 7a. Вклад компонентов лосса в градиент (только detection, только если
        # включён clearml.plots.grad_contrib_every_n_steps — см.
        # GradientContributionTracker). gradnorm — абсолютная норма ‖∂L_i/∂θ‖
        # компонента ДО умножения на λ (сравнима 1-в-1 с loss_components выше),
        # gradshare — её доля среди всех компонентов в %, сумма долей = 100.
        for key, value in row.items():
            if key.startswith("train_gradnorm_"):
                _safe_report("grad_contribution", key.removeprefix("train_gradnorm_"), value)
            elif key.startswith("train_gradshare_"):
                _safe_report("grad_contribution_share_pct", key.removeprefix("train_gradshare_"), value)

        # 8. Метрики детекции
        detection_metrics = {
            "train_map": ("mAP", "train"),
            "eval_map": ("mAP", "eval"),

            "train_map_50": ("mAP@50", "train"),
            "eval_map_50": ("mAP@50", "eval"),

            "train_map_75": ("mAP@75", "train"),
            "eval_map_75": ("mAP@75", "eval"),

            "train_mar_100": ("mAR@100", "train"),
            "eval_mar_100": ("mAR@100", "eval"),

            # Разбивка по размерам: на Cityscapes после ресайза большая часть
            # объектов мелкая, и именно эта тройка показывает, где теряется mAP.
            "eval_map_small": ("mAP by size", "small"),
            "eval_map_medium": ("mAP by size", "medium"),
            "eval_map_large": ("mAP by size", "large"),

            # Диагностика коллапса: обе величины падают раньше, чем mAP
            # успевает дойти до нуля.
            "eval_predictions_per_image": ("detections", "per_image"),
            "eval_max_score": ("detections", "max_score"),
        }

        for key, (title, series_name) in detection_metrics.items():
            if key in row:
                # title берётся из словаря: с захардкоженным именем графика все
                # метрики писались в одну серию и затирали друг друга.
                _safe_report(title, series_name, row[key])

        # per-class AP приходит ключами вида eval_ap_person. Перечислить их
        # в словаре нельзя: имена классов зависят от датасета.
        for key, value in row.items():
            if key.startswith("eval_ap_"):
                _safe_report("AP per class", key.removeprefix("eval_ap_"), value)

        # 9. Метрики сегментации
        segmentation_metrics = {
            "train_miou": ("segmentation_metrics", "train_mIoU"),
            "eval_miou": ("segmentation_metrics", "eval_mIoU"),
            "train_pixel_acc": ("pixel_accuracy", "train"),
            "eval_pixel_acc": ("pixel_accuracy", "eval"),
            # Диагностика дистилляции — собственное качество учителя, а не
            # схожесть с ним. Два независимых флага в clearml.scalars:
            # train_teacher_mIoU (флаг "teacher_miou") — по эпохам, учитель на
            # тех же кропах/масштабе/аугментациях, что ПРЯМО СЕЙЧАС видит
            # ученик; teacher_native_mIoU (отдельный флаг "teacher_native_miou")
            # — плоская линия-ориентир, учитель на eval_loader (родное
            # разрешение, без кропа) — не включена по умолчанию вместе с
            # первой, т.к. не меняется по эпохам и обычно и так известна.
            # Разница между ними — прямая проверка гипотезы про разрешение
            # учителя при дистилляции.
            "train_teacher_miou": ("segmentation_metrics", "train_teacher_mIoU"),
            "teacher_native_miou": ("segmentation_metrics", "teacher_native_mIoU"),
        }
        for key, (title, series_name) in segmentation_metrics.items():
            if key in row:
                _safe_report(title, series_name, row[key])

    def report_single(single_values: dict):
        for key, val in single_values.items():
            logger.report_single_value(key, val)

    def report_table(df):
        # build_param_table объявлен как DataFrame | None
        if df is None:
            return
        logger.report_table(
            title="parameters",
            series="param_counts",
            iteration=0,
            table_plot=df,
        )

    def report_plots(plots, iteration: int) -> None:
        """Графики раздела Plots: столбики и гистограммы -> report_histogram, карты -> confusion_matrix.

        Тренер отдаёт src.utils.plots.Plot — чистые данные без знания о ClearML;
        весь перевод в вызовы SDK собран здесь.
        """
        for plot in plots:
            if plot.kind == "image":
                logger.report_image(
                    title=plot.title,
                    series=plot.series,
                    image=plot.values,
                    iteration=iteration,
                    # По умолчанию ClearML хранит 5 последних картинок на серию —
                    # срез до обучения вытеснился бы, а он нужен как точка отсчёта.
                    max_image_history=-1,
                )
            elif plot.kind == "matrix":
                logger.report_confusion_matrix(
                    title=plot.title,
                    series=plot.series,
                    matrix=plot.values,
                    iteration=iteration,
                    xlabels=plot.xlabels,
                    ylabels=plot.ylabels,
                    xaxis=plot.xaxis,
                    yaxis=plot.yaxis,
                    # (0,0) в левом верхнем углу — привычная ориентация матрицы ошибок
                    yaxis_reversed=True,
                )
            else:
                # ClearML сам гистограмму не считает: values — уже готовые высоты
                # столбиков, биннинг сделан в src/utils/plots.py.
                logger.report_histogram(
                    title=plot.title,
                    series=plot.series,
                    values=plot.values,
                    iteration=iteration,
                    xlabels=plot.xlabels,
                    xaxis=plot.xaxis,
                    yaxis=plot.yaxis,
                )

    def report_debug_sample(
        image,
        series: str,
        iteration: int,
    ) -> None:
        image = image.detach().cpu()

        if image.dtype == torch.uint8:
            image = image.permute(1, 2, 0).numpy()
        else:
            image = image.clamp(0, 1)
            image = image.permute(1, 2, 0).numpy()

        logger.report_image(
            title="Validation Detection",
            series=series,
            iteration=iteration,
            image=image,
    )

    return report_scalar, report_single, report_table, report_plots, report_debug_sample


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    seed_everything(cfg.seed, deterministic=cfg.deterministic, warn_only=cfg.deterministic_warn_only)
    dist = distributed.setup(
        device_cfg=cfg.device,
        backend=cfg.distributed.backend,
        timeout_minutes=cfg.distributed.timeout_minutes,
    )
    try:
        configure_rank_logging(dist)

        if cfg.task_type not in BEST_METRIC_KEY:
            raise ValueError(
                f"task_type={cfg.task_type!r} неизвестен. "
                f"Доступны: {sorted(BEST_METRIC_KEY)}."
            )

        output_dir = resolve_output_dir(dist)
        device = dist.device

        task = init_clearml(cfg, dist)

        log.info("Конфиг запуска:\n%s", OmegaConf.to_yaml(cfg))
        log.info("Артефакты запуска: %s", output_dir)

        # DDP портирован в Trainer и SegmentationTrainer; DetectionTrainer
        # его пока не знает и под torchrun обучал бы N независимых моделей,
        # ничем это не показывая. Падаем на старте, а не через сутки.
        if dist.is_distributed and cfg.task_type == "detection":
            raise ValueError(
                "DetectionTrainer пока не поддерживает распределённый запуск "
                f"(world_size={dist.world_size}). Запускай детекцию одним процессом: "
                f"python scripts/train.py ..."
            )

        # Mixup/CutMix живут не в трансформе, а в тренере: они смешивают разные
        # примеры между собой, а трансформ видит только один (см. src/data/batch_augment.py).
        batch_augment = instantiate(cfg.augment) if cfg.get("augment") is not None else None
        if batch_augment is not None and cfg.task_type == "detection":
            raise ValueError(
                "Mixup/CutMix для детекции не поддержаны: смешивание меняет набор боксов, "
                "а не только таргет попиксельно. Уберите augment из конфига (augment=null)."
            )

        with distributed.main_process_first(dist):
            train_loader, eval_loader = base_loader(
                cfg.data, cfg.task_type, seed=cfg.seed, dist=dist
            )

            student = instantiate(cfg.model.student).to(device)
            teacher = None
            if cfg.model.get("teacher") is not None:
                teacher = instantiate(cfg.model.teacher).to(device)
                # Учителя апсемплим до разрешения, на котором его мерили/
                # дообучали (см. src/models/native_resolution_teacher.py) —
                # ПЕРЕД мультимасштабным прогоном, если оба заданы: тогда
                # каждый масштаб MultiScaleInference берётся уже от родного
                # разрешения, а не от кропа студента.
                teacher_native_resolution = _plain(cfg.model.get("teacher_native_resolution"))
                if teacher_native_resolution:
                    teacher = NativeResolutionTeacher(teacher, **teacher_native_resolution)
                    log.info(
                        "Учитель апсемплится до родного разрешения: %s", teacher_native_resolution
                    )
                # Мультимасштабный прогон учителя: несколько forward'ов вместо
                # одного, зато таргет заметно чище (см. src/models/multi_scale.py).
                teacher_inference = _plain(cfg.model.get("teacher_inference"))
                if teacher_inference:
                    teacher = MultiScaleInference(teacher, **teacher_inference)
                    log.info("Учитель считает таргеты мультимасштабно: %s", teacher_inference)
                # Режем батч учителя на куски ПОСЛЕДНИМ (снаружи всех
                # обёрток выше) — учитель без градиентов, чанкинг ничего не
                # меняет в результате (см. src/models/chunked_teacher.py),
                # только развязывает batch_size, под который тюнились lr и
                # расписание оптимизатора, от памяти/лимитов CUDA-ядер на
                # ОДИН forward тяжёлого учителя (особенно актуально вместе с
                # teacher_native_resolution — апсемпленный батч целиком легко
                # не помещается).
                teacher_micro_batch_size = cfg.model.get("teacher_micro_batch_size")
                if teacher_micro_batch_size:
                    teacher = ChunkedTeacher(teacher, micro_batch_size=teacher_micro_batch_size)
                    log.info(
                        "Учитель считается кусками батча по %s", teacher_micro_batch_size
                    )
            criterion = instantiate(cfg.loss).to(device)

        # заменяем слои BatchNorm до сборки optimizer
        if cfg.distributed.sync_bn and dist.is_distributed:
            student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
            log.info("BatchNorm заменён на SyncBatchNorm")

        # Обучаемые параметры лосса (адаптеры feature-KD) оптимизируются вместе с учеником.
        # lwdetr_adamw группирует параметры по имени (backbone/decoder/...,
        # см. src/optimizers/lwdetr_optimizer.py), поэтому ему нужны (name, param),
        # а не build_param_groups — тот отдаёт голые nn.Parameter/группы-словари.
        if cfg.task_type == "detection" and cfg.data.dataset.targets_format_mode == "lw-detr-small":
            params = (
                list(student.named_parameters())
                + [(f"criterion.{name}", param) for name, param in criterion.named_parameters()]
            )
        else:
            params = build_optimizer_params(cfg, student, criterion)

        optimizer = instantiate(cfg.optimizer)(params)
        scheduler = instantiate(cfg.scheduler)(optimizer) if cfg.get("scheduler") is not None else None

        # Расписание весов слагаемых лосса. Строится до тренера: там criterion
        # уже может уехать под DDP-обёртку, а планировщику нужен сам модуль.
        loss_schedule = None
        if cfg.get("loss_schedule"):
            if cfg.task_type == "detection":
                raise ValueError("loss_schedule пока поддержан только для классификации и сегментации")
            loss_schedule = LossWeightScheduler(
                criterion, _plain(cfg.loss_schedule), total_epochs=cfg.trainer.epochs
            )

        # cfg.get(...): при `prediction_postprocessors: null` Hydra ключ в struct не создаёт,
        # прямое обращение падает с ConfigAttributeError.
        postprocessor_cfg = cfg.get("prediction_postprocessors")
        prediction_postprocessor = instantiate(postprocessor_cfg) if postprocessor_cfg is not None else None

        if cfg.task_type == "classification":
            trainer = Trainer(
                student=student,
                teacher=teacher,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                eval_loader=eval_loader,
                dist=dist,
                find_unused_parameters=cfg.distributed.find_unused_parameters,
                broadcast_buffers=cfg.distributed.broadcast_buffers,
                output_dir=output_dir,
                metrics_callback=clearml_reporter(task) if task is not None else None,
                num_classes=cfg.data.dataset.num_classes,
                scalars=cfg.clearml.scalars,
                batch_augment=batch_augment,
                loss_schedule=loss_schedule,
                **cfg.trainer,
            )
        elif cfg.task_type == "detection":
            trainer = DetectionTrainer(
                student=student,
                teacher=teacher,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                eval_loader=eval_loader,
                device=device,
                output_dir=output_dir,
                metrics_callback=clearml_reporter(task) if task is not None else None,
                num_classes=cfg.data.dataset.num_classes,
                scalars=cfg.clearml.scalars,
                prediction_postprocessor=prediction_postprocessor,
                label_offset=cfg.data.dataset.build.label_offset,
                targers_mode=cfg.data.dataset.targets_format_mode,
                # Подписывают per-class AP; без них в логе останутся индексы 0..7.
                class_names=getattr(train_loader.dataset, "label_to_name", None),
                # Возвращает Debug Samples исходные цвета; None, если
                # нормализации не было (у YOLO вход остаётся в 0..1).
                normalize=_normalize_stats(cfg.data.dataset),
                plots=_plain(cfg.clearml.get("plots")),
                **cfg.trainer,
            )
            if cfg.resume:
                checkpoint_path = output_dir / "last.pt"
                if checkpoint_path.exists():
                    trainer.load_checkpoint(checkpoint_path)
                    log.info(
                        "Продолжаем с чекпоинта %s: эпоха %d, лучший mAP=%.4f",
                        checkpoint_path, trainer.start_epoch - 1, trainer.best_map,
                    )
                else:
                    log.warning("resume=true, но %s не найден — стартуем с нуля.", checkpoint_path)
        elif cfg.task_type == "segmentation":
            trainer = SegmentationTrainer(
                student=student,
                teacher=teacher,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                eval_loader=eval_loader,
                dist=dist,
                find_unused_parameters=cfg.distributed.find_unused_parameters,
                broadcast_buffers=cfg.distributed.broadcast_buffers,
                output_dir=output_dir,
                metrics_callback=clearml_reporter(task) if task is not None else None,
                num_classes=cfg.data.dataset.num_classes,
                scalars=cfg.clearml.scalars,
                ignore_index=cfg.data.dataset.ignore_index,
                batch_augment=batch_augment,
                loss_schedule=loss_schedule,
                plots=_plain(cfg.clearml.get("plots")),
                # Имена классов подписывают столбики per-class графиков и оси
                # матрицы ошибок; без них останутся индексы 0..18.
                class_names=getattr(train_loader.dataset, "classes", None),
                # Палитра и нормировка нужны Debug Samples: первая красит маски,
                # вторая возвращает кадру исходные цвета после Normalize.
                palette=getattr(train_loader.dataset, "palette", None),
                normalize=_normalize_stats(cfg.data.dataset),
                **cfg.trainer,
            )
            if cfg.resume:
                checkpoint_path = output_dir / "last.pt"
                if checkpoint_path.exists():
                    trainer.load_checkpoint(checkpoint_path)
                    log.info(
                        "Продолжаем с чекпоинта %s: эпоха %d, лучший mIoU=%.4f",
                        checkpoint_path, trainer.start_epoch - 1, trainer.best_miou,
                    )
                else:
                    log.warning("resume=true, но %s не найден — стартуем с нуля.", checkpoint_path)
        else:
            # Недостижимо, пока task_type проверяется в начале main(). Нужно
            # на случай, когда новую задачу добавят в BEST_METRIC_KEY, а ветку
            # с тренером здесь завести забудут: без этого trainer остался бы
            # неопределённым и падение случилось бы ниже, с UnboundLocalError.
            raise AssertionError(cfg.task_type)

        summary = trainer.fit()

        # Целевая метрика запуска: своя на каждую задачу, имя должно совпадать
        # с тем, что реально измерено, иначе mIoU уезжает в ClearML как "acc".
        best_metric = summary[BEST_METRIC_KEY[cfg.task_type]]

        if task is not None:
            logger = task.get_logger()
            logger.report_single_value(BEST_METRIC_KEY[cfg.task_type], best_metric)
            logger.report_single_value("best_epoch", summary["best_epoch"])
            logger.report_single_value("world_size", dist.world_size)
            task.close()

    finally:
        # при падении одного ранка остальные должны корректно
        # закрыть группу, а не висеть в коллективной операции до таймаута.
        distributed.cleanup()

    # Возврат метрики делает скрипт совместимым с hydra-свиперами
    # (optuna и т.п. максимизируют возвращаемое значение).
    return best_metric


if __name__ == "__main__":
    main()
