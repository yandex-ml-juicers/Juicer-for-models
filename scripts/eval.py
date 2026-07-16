"""Оценка сохранённого чекпоинта на тестовой выборке.

Пример:
    python scripts/eval.py experiment=b2_feature_kd \
        ckpt_path=outputs/b2_resnet50_to_resnet18_feature_kd/2026-07-16_12-00-00/best.pt

experiment должен совпадать с тем, которым обучали: из него берутся
архитектура ученика и датасет.
"""

import logging

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig

from src.data import build_dataloaders
from src.training import evaluate
from src.utils import resolve_device, seed_everything

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    if cfg.ckpt_path is None:
        raise ValueError("Укажи чекпоинт: python scripts/eval.py ckpt_path=outputs/…/best.pt")

    seed_everything(cfg.seed, deterministic=cfg.deterministic, warn_only=cfg.deterministic_warn_only)
    device = resolve_device(cfg.device)

    _, eval_loader = build_dataloaders(cfg.data, seed=cfg.seed)

    student = instantiate(cfg.model.student).to(device)
    checkpoint = torch.load(to_absolute_path(cfg.ckpt_path), map_location=device, weights_only=True)
    student.load_state_dict(checkpoint["student_state"])
    log.info("Загружен чекпоинт эпохи %d (best_acc=%.4f)", checkpoint["epoch"], checkpoint["best_acc"])

    eval_loss, eval_acc = evaluate(student, eval_loader, device)
    log.info("eval loss=%.4f | eval acc=%.2f%%", eval_loss, eval_acc * 100)
    return eval_acc


if __name__ == "__main__":
    main()
