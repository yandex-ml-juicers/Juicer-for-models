import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.samplers import ShardSampler
from src.data.transforms import base_transform
from src.utils.distributed import DistInfo
from src.utils.seed import make_generator, seed_worker

log = logging.getLogger(__name__)


def base_loader(
        cfg: DictConfig, seed: int, dist: DistInfo | None = None
    ) -> tuple[DataLoader, DataLoader]:
    """Возвращает (train_loader, eval_loader).

    Args:
        cfg: cfg.data
        seed: базовый seed запуска
        dist: параметры процесса в группе

    Сэмплеры собираются прямо здесь, а не в вызывающем коде

    `set_epoch()` на train-сэмплере зовёт Trainer, доставая его через
    `train_loader.sampler`.
    """
    world_size = dist.world_size if dist is not None else 1
    rank = dist.rank if dist is not None else 0

    train_transform = instantiate(cfg.transform.train)
    eval_transform = instantiate(cfg.transform.eval)

    train_dataset = instantiate(cfg.dataset.build, train=True, transform=train_transform)
    eval_dataset = instantiate(cfg.dataset.build, train=False, transform=eval_transform)

    # batch_size в конфиге - глобальный размер батча, т.е. число объектов
    # на один шаг оптимизации. Каждому процессу достаётся 
    # batch_size = global_batch_size // world_size
    global_batch_size = cfg.loader.batch_size
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"batch_size={global_batch_size} не делится на число процессов "
            f"({world_size}). batch_size задаётся глобально, на все карты сразу — "
            f"возьми ближайшее кратное, например {global_batch_size // world_size * world_size}."
        )
    batch_size = global_batch_size // world_size

    num_workers = cfg.loader.num_workers
    common = dict(
        num_workers=num_workers,
        pin_memory=cfg.loader.pin_memory and torch.cuda.is_available(),
        # persistent_workers = True - значит воркеры не умирают между эпохами
        persistent_workers=cfg.loader.persistent_workers and num_workers > 0,
        worker_init_fn=seed_worker,
    )

    if world_size > 1:
        # drop_last=False дополняет выборку повтором первых
        # примеров до кратности world_size.
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=False
        )
        # Для eval — свой сэмплер без дополнения дубликатами, 
        # иначе метрика окажется смещённой
        eval_sampler = ShardSampler(eval_dataset, num_replicas=world_size, rank=rank)
        train_shuffle = None  # sampler и shuffle взаимоисключающи
    else:
        train_sampler = None
        eval_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        # необходимо равное число батчей
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        # Сдвиг на ранг разводит источники случайности в воркерах. l
        generator=make_generator(seed + rank),
        **common,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=eval_sampler,
        **common,
    )

    log.info(
        "Батч: %d глобально, %d на процесс (world_size=%d) | train: %d объектов, %d на процесс",
        global_batch_size,
        batch_size,
        world_size,
        len(train_dataset),
        len(train_sampler) if train_sampler is not None else len(train_dataset),
    )
    return train_loader, eval_loader
