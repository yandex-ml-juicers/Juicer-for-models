"""Единая точка входа обучения.

Артефакты запуска (конфиг, логи, history.csv, чекпоинты) складываются
в outputs/<name>/<дата_время>/.
"""

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.data import base_loader
from src.training import Trainer
from src.utils import distributed, seed_everything
from src.utils.distributed import DistInfo

log = logging.getLogger(__name__)


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
            f"переменная JUICER_RUN_DIR или TORCHELASTIC_RUN_ID."
        )
    return output_dir


def init_clearml(cfg: DictConfig, dist: DistInfo):
    # таска создаётся только на главном ранке
    if not dist.is_main or not cfg.clearml.enabled:
        return None
    from clearml import Task

    task = Task.init(
        project_name=cfg.clearml.project,                      # проект
        task_name=f"{cfg.name}",                               # имя таски
        task_type=Task.TaskTypes.training,                     # тип таски
        tags=cfg.clearml.tags,                                 # теги
        reuse_last_task_id = cfg.clearml.reuse_last_task_id,   # перезаписывать ли таску с таким же именем
        continue_last_task=cfg.clearml.continue_last_task,     # Подхватит предыдущий ID и продолжит логирование
        output_uri=cfg.clearml.output_uri,                     # складывать ли артефакты/модели и если куда-то базово, то url
        auto_connect_frameworks=cfg.clearml.auto_connect_frameworks, # авто-перехват фреймворков
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

    def report_scalar(row: dict, iteration: str = 'epoch') -> None:

        iterate = row[iteration]
        logger.report_scalar(title="loss", series="train", value=row["train_loss_total"], iteration=iterate)
        logger.report_scalar(title="loss", series="eval", value=row["eval_loss"], iteration=iterate)
        logger.report_scalar(title="accuracy", series="train", value=row["train_acc"], iteration=iterate)
        logger.report_scalar(title="accuracy", series="eval", value=row["eval_acc"], iteration=iterate)
        logger.report_scalar(title="lr", series="lr", value=row["lr"], iteration=iterate)
        if "train_precision" in row.keys():
            logger.report_scalar(title="precision", series="train", value=row["train_precision"], iteration=iterate)
            logger.report_scalar(title="recall", series="train", value=row["train_recall"], iteration=iterate)
            logger.report_scalar(title="F1", series="train", value=row["train_F1"], iteration=iterate)
        if "train_KL_divergence" in row.keys():
            logger.report_scalar(title="KL", series="train_KL", value=row["train_KL_divergence"], iteration=iterate)
            logger.report_scalar(title="agreement_rate", series="train_agreement_rate", value=row["train_agreement_rate"], iteration=iterate)
        logger.report_scalar(title="grad_norm", series="train_avg_grad_norm", value=row["train_avg_grad_norm"], iteration=iterate)
        logger.report_scalar(title="grad_norm", series="train_max_grad_norm", value=row["train_max_grad_norm"], iteration=iterate)
        logger.report_scalar(title="weight_norm", series="train_avg_weight_norm", value=row["train_avg_weight_norm"], iteration=iterate)
        logger.report_scalar(title="weight_norm", series="train_max_weight_norm", value=row["train_max_weight_norm"], iteration=iterate)

        # Компоненты лосса (train_ce, train_kd, train_feature_*) — одним графиком.
        for key, value in row.items():
            if key.startswith("train_loss"):
                logger.report_scalar(
                    "loss_components", key.removeprefix("train_"), value, iteration=iterate
                )

    def report_single(single_values: dict):
        for key, val in single_values.items():
            task.get_logger().report_single_value(key, val)

    def report_table(df):
        task.get_logger().report_table(
        title="parameters",
        series="param_counts",
        iteration=0,
        table_plot=df,
        )

    return report_scalar, report_single, report_table


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
        output_dir = resolve_output_dir(dist)
        device = dist.device

        task = init_clearml(cfg, dist)

        log.info("Конфиг запуска:\n%s", OmegaConf.to_yaml(cfg))
        log.info("Артефакты запуска: %s", output_dir)

        train_loader, eval_loader = base_loader(cfg.data, seed=cfg.seed, dist=dist)

        student = instantiate(cfg.model.student).to(device)
        teacher = None
        if cfg.model.get("teacher") is not None:
            teacher = instantiate(cfg.model.teacher).to(device)
        criterion = instantiate(cfg.loss).to(device)

        # заменяем слои BatchNorm до сборки optimizer
        if cfg.distributed.sync_bn and dist.is_distributed:
            student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
            log.info("BatchNorm заменён на SyncBatchNorm")

        # Обучаемые параметры лосса (адаптеры feature-KD) оптимизируются вместе с учеником.
        params = list(student.parameters()) + list(criterion.parameters())
        optimizer = instantiate(cfg.optimizer)(params)
        scheduler = instantiate(cfg.scheduler)(optimizer) if cfg.get("scheduler") is not None else None

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
            **cfg.trainer,
        )
        result = trainer.fit()

        if task is not None:
            task.get_logger().report_single_value("best_eval_acc", result["best_acc"])
            task.get_logger().report_single_value("world_size", dist.world_size)
            task.close()
    finally:
        # при падении одного ранка остальные должны корректно
        # закрыть группу, а не висеть в коллективной операции до таймаута.
        distributed.cleanup()

    # Возврат метрики делает скрипт совместимым с hydra-свиперами
    # (optuna и т.п. максимизируют возвращаемое значение).
    return result["best_acc"]


if __name__ == "__main__":
    main()
