"""Единая точка входа обучения.

Артефакты запуска (конфиг, логи, history.csv, чекпоинты) складываются
в outputs/<name>/<дата_время>/.
"""

import logging
import math
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.data import base_loader
from src.training import Trainer, DetectionTrainer, SegmentationTrainer
from src.utils import resolve_device, seed_everything, prediction_postprocessor

log = logging.getLogger(__name__)


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


def init_clearml(cfg: DictConfig):
    if not cfg.clearml.enabled:
        return None
    from clearml import Task

    task = Task.init(
        project_name=cfg.clearml.project,                      # проект
        task_name=f"{cfg.name}",                               # имя таски
        task_type=Task.TaskTypes.training,                     # тип таски
        tags=_plain(cfg.clearml.tags),                         # теги
        reuse_last_task_id = cfg.clearml.reuse_last_task_id,   # перезаписывать ли таску с таким же именем
        continue_last_task=cfg.clearml.continue_last_task,     # Подхватит предыдущий ID и продолжит логирование
        output_uri=cfg.clearml.output_uri,                     # складывать ли артефакты/модели и если куда-то базово, то url
        auto_connect_frameworks=_plain(cfg.clearml.auto_connect_frameworks), # авто-перехват фреймворков
        auto_connect_arg_parser=cfg.clearml.auto_connect_arg_parser, # авто-перехват аргументов из argparse
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
                _safe_report("loss_components", series_name, value)

        # 8. Метрики детекции
        detection_metrics = {
            "eval_map": "mAP",
            "eval_map_50": "mAP@50",
            "eval_map_75": "mAP@75",
            "eval_mar_100": "mAR@100",
        }
        for key, series_name in detection_metrics.items():
            if key in row:
                _safe_report("detection_metrics", series_name, row[key])

        # 9. Метрики сегментации
        segmentation_metrics = {
            "train_miou": ("segmentation_metrics", "train_mIoU"),
            "eval_miou": ("segmentation_metrics", "eval_mIoU"),
            "train_pixel_acc": ("pixel_accuracy", "train"),
            "eval_pixel_acc": ("pixel_accuracy", "eval"),
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

    return report_scalar, report_single, report_table, report_plots


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    task = init_clearml(cfg)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info("Конфиг запуска:\n%s", OmegaConf.to_yaml(cfg))
    log.info("Артефакты запуска: %s", output_dir)

    seed_everything(cfg.seed, deterministic=cfg.deterministic, warn_only=cfg.deterministic_warn_only)
    device = resolve_device(cfg.device)

    train_loader, eval_loader = base_loader(cfg.data, cfg.task_type, seed=cfg.seed)

    # Mixup/CutMix живут не в трансформе, а в тренере: они смешивают разные
    # примеры между собой, а трансформ видит только один (см. src/data/batch_augment.py).
    batch_augment = instantiate(cfg.augment) if cfg.get("augment") is not None else None
    if batch_augment is not None and cfg.task_type == "detection":
        raise ValueError(
            "Mixup/CutMix для детекции не поддержаны: смешивание меняет набор боксов, "
            "а не только таргет попиксельно. Уберите augment из конфига (augment=null)."
        )

    student = instantiate(cfg.model.student).to(device)
    teacher = None
    if cfg.model.get("teacher") is not None:
        teacher = instantiate(cfg.model.teacher).to(device)
    criterion = instantiate(cfg.loss).to(device)

    # Обучаемые параметры лосса (адаптеры feature-KD) оптимизируются вместе с учеником.
    params = list(student.parameters()) + list(criterion.parameters())
    optimizer = instantiate(cfg.optimizer)(params)  
    scheduler = instantiate(cfg.scheduler)(optimizer) if cfg.get("scheduler") is not None else None

    if cfg.task_type == "classification":
        trainer = Trainer(
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
            batch_augment=batch_augment,
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
            **cfg.trainer,
        )
    elif cfg.task_type == "segmentation":
        trainer = SegmentationTrainer(
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
            ignore_index=cfg.data.dataset.ignore_index,
            batch_augment=batch_augment,
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
    summary = trainer.fit()

    # Целевая метрика запуска: своя на каждую задачу, имя должно совпадать
    # с тем, что реально измерено, иначе mIoU уезжает в ClearML как "acc".
    target_metric = {
        "classification": "best_acc",
        "detection": "best_map",
        "segmentation": "best_miou",
    }[cfg.task_type]
    result = summary[target_metric]

    if task is not None:
        logger = task.get_logger()
        logger.report_single_value(target_metric, result)
        logger.report_single_value("best_epoch", summary["best_epoch"])
        task.close()

    # Возврат метрики делает скрипт совместимым с hydra-свиперами
    # (optuna и т.п. максимизируют возвращаемое значение).
    return result


if __name__ == "__main__":
    main()
