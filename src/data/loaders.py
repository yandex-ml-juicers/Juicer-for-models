import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.transforms import build_transforms
from src.utils.seed import make_generator, seed_worker


def build_dataloaders(cfg: DictConfig, seed: int) -> tuple[DataLoader, DataLoader]:
    """Возвращает (train_loader, eval_loader).

    Воспроизводимость порядка данных обеспечивают два механизма:
    - выделенный generator для shuffle (не зависит от глобального генератора);
    - seed_worker, сидирующий NumPy/random в каждом воркере DataLoader.
    """
    train_transform = build_transforms(
        mean=cfg.normalize.mean, std=cfg.normalize.std, image_size=cfg.image_size
    )
    eval_transform = build_transforms(
        mean=cfg.normalize.mean, std=cfg.normalize.std, image_size=cfg.image_size
    )

    train_dataset = instantiate(cfg.dataset, train=True, transform=train_transform)
    eval_dataset = instantiate(cfg.dataset, train=False, transform=eval_transform)

    num_workers = cfg.loader.num_workers
    common = dict(
        num_workers=num_workers,
        pin_memory=cfg.loader.pin_memory and torch.cuda.is_available(),
        # persistent_workers несовместим с num_workers=0
        persistent_workers=cfg.loader.persistent_workers and num_workers > 0,
        worker_init_fn=seed_worker,
    )
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
