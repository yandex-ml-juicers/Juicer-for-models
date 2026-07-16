"""Единая точка входа обучения.

Примеры:
    python scripts/train.py experiment=b1_vanilla_kd
    python scripts/train.py experiment=b2_feature_kd trainer.epochs=30
    python scripts/train.py loss=hinton_kd loss.temperature=8 seed=1
    python scripts/train.py data=fake '~model/teacher' loss=ce \
        trainer.epochs=1 trainer.limit_train_batches=3   # смоук без сети/GPU

(`~model/teacher` — CLI-синтаксис удаления группы из defaults; в yaml-файлах
экспериментов то же самое пишется как `- override /model/teacher: null`.)

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


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info("Конфиг запуска:\n%s", OmegaConf.to_yaml(cfg))
    log.info("Артефакты запуска: %s", output_dir)

    # Первым делом, до любых созданий тензоров и загрузок данных.
    seed_everything(cfg.seed, deterministic=cfg.deterministic, warn_only=cfg.deterministic_warn_only)
    device = resolve_device(cfg.device)

    train_loader, eval_loader = build_dataloaders(cfg.data, seed=cfg.seed)

    student = instantiate(cfg.model.s`tudent).to(device)
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
        **cfg.trainer,
    )
    result = trainer.fit()

    # Возврат метрики делает скрипт совместимым с hydra-свиперами
    # (optuna и т.п. максимизируют возвращаемое значение).
    return result["best_acc"]


if __name__ == "__main__":
    main()
