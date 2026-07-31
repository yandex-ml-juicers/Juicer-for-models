import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.transforms import base_transform
from src.utils.seed import make_generator, seed_worker


def base_loader(cfg: DictConfig, seed: int) -> tuple[DataLoader, DataLoader]:
    """Возвращает (train_loader, eval_loader)."""
    train_transform = instantiate(cfg.transform.train)
    eval_transform = instantiate(cfg.transform.eval)

    train_dataset = instantiate(cfg.dataset.build, train=True, transform=train_transform)
    eval_dataset = instantiate(cfg.dataset.build, train=False, transform=eval_transform)

    num_workers = cfg.loader.num_workers
    common = dict(
        num_workers=num_workers,
        pin_memory=cfg.loader.pin_memory and torch.cuda.is_available(),
        # persistent_workers несовместим с num_workers=0
        persistent_workers=cfg.loader.persistent_workers and num_workers > 0,
        worker_init_fn=seed_worker,
    )

    if cfg.task_type == "detection":
        common["collate_fn"] = detection_collate_fn

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.loader.batch_size,
        shuffle=True,
        generator=make_generator(seed),
        **common,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.loader.batch_size,
        shuffle=False,
        **common,
    )
    return train_loader, eval_loader


def detection_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[
    tuple[torch.Tensor, ...],
    tuple[dict[str, torch.Tensor], ...],
]:
    return tuple(zip(*batch))