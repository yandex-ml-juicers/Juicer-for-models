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

from src.data import build_dataloaders
from src.training import Trainer
from src.utils import resolve_device, seed_everything

log = logging.getLogger(__name__)


def init_clearml(cfg: DictConfig):
    if not cfg.clearml.enabled:
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

    def report(row: dict, iteration: str = 'epoch') -> None:
    
        iterate = row[iteration]
        logger.report_scalar(title="loss", series="train", value=row["train_total"], iteration=iterate)
        logger.report_scalar(title="loss", series="eval", value=row["eval_loss"], iteration=iterate)
        logger.report_scalar(title="accuracy", series="train", value=row["train_acc"], iteration=iterate)
        logger.report_scalar(title="accuracy", series="eval", value=row["eval_acc"], iteration=iterate)
        logger.report_scalar(title="lr", series="lr", value=row["lr"], iteration=iterate)
        logger.report_scalar(title="precision", series="train", value=row["train_precision"], iteration=iterate)
        logger.report_scalar(title="recall", series="train", value=row["train_recall"], iteration=iterate)
        logger.report_scalar(title="F1", series="train", value=row["train_F1"], iteration=iterate)
        logger.report_scalar(title="KL", series="KL", value=row["KL_divergence"], iteration=iterate)
        logger.report_scalar(title="agreement_rate", series="agreement_rate", value=row["agreement_rate"], iteration=iterate)
        logger.report_scalar(title="grad_norm", series="avg_grad_norm", value=row["avg_grad_norm"], iteration=iterate)
        logger.report_scalar(title="grad_norm", series="max_grad_norm", value=row["max_grad_norm"], iteration=iterate)
        logger.report_scalar(title="weight_norm", series="avg_weight_norm", value=row["avg_weight_norm"], iteration=iterate)
        logger.report_scalar(title="weight_norm", series="max_weight_norm", value=row["max_weight_norm"], iteration=iterate)
        

        # Компоненты лосса (train_ce, train_kd, train_feature_*) — одним графиком.
        for key, value in row.items():
            if key.startswith("train_") and key not in ('train_total', 'train_acc', 'train_precision', 'train_recall', 'train_F1'):
                logger.report_scalar(
                    "loss_components", key.removeprefix("train_"), value, iteration=iterate
                )

    return report


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    # ClearML — первой строкой, до создания моделей и тренера (см. base_docs.md).
    task = init_clearml(cfg)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info("Конфиг запуска:\n%s", OmegaConf.to_yaml(cfg))
    log.info("Артефакты запуска: %s", output_dir)

    # Первым делом, до любых созданий тензоров и загрузок данных.
    seed_everything(cfg.seed, deterministic=cfg.deterministic, warn_only=cfg.deterministic_warn_only)
    device = resolve_device(cfg.device)

    train_loader, eval_loader = build_dataloaders(cfg.data, seed=cfg.seed)

    student = instantiate(cfg.model.student).to(device)
    teacher = None
    if cfg.model.get("teacher") is not None:
        teacher = instantiate(cfg.model.teacher).to(device)
    criterion = instantiate(cfg.loss).to(device)

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
        device=device,
        output_dir=output_dir,
        metrics_callback=clearml_reporter(task) if task is not None else None,
        **cfg.trainer,
    )
    result = trainer.fit()

    if task is not None:
        task.get_logger().report_single_value("best_eval_acc", result["best_acc"])
        task.close()

    # Возврат метрики делает скрипт совместимым с hydra-свиперами
    # (optuna и т.п. максимизируют возвращаемое значение).
    return result["best_acc"]


if __name__ == "__main__":
    main()
