"""Полный контроль источников случайности для воспроизводимых запусков.

Бит-в-бит воспроизводимость гарантируется при одинаковых: железе (модель GPU),
версиях torch/CUDA/cuDNN и конфиге запуска. Между разными GPU или версиями
библиотек совпадение результатов не гарантируется — это ограничение CUDA-ядер,
а не кода.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True, warn_only: bool = False) -> int:
    """Фиксирует все источники случайности: Python, NumPy, PyTorch, CUDA.

    Args:
        seed: базовый seed для всех генераторов.
        deterministic: если True, принуждает PyTorch использовать только
            детерминированные ядра (бит-в-бит между запусками ценой
            ~10-20% скорости). Если False — включает cudnn.benchmark
            (автоподбор быстрых свёрточных ядер, недетерминированный).
        warn_only: аварийный клапан. Если у операции нет детерминированной
            реализации, вместо RuntimeError будет warning. Использовать
            только когда падение блокирует работу, и осознавать, что
            воспроизводимость в этой точке теряется.

    Returns:
        Тот же seed (удобно логировать).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # сидирует и все CUDA-устройства тоже
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Требование cuBLAS >= 10.2 для детерминированных matmul.
        # Переменная должна быть выставлена ДО первой CUDA-операции,
        # поэтому seed_everything() вызывается первой строкой в main().
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

    return seed


def seed_worker(worker_id: int) -> None:
    """worker_init_fn для DataLoader: сидирует NumPy/random в каждом воркере.

    PyTorch сам раздаёт воркерам производные seed'ы только для torch-генератора;
    NumPy и random внутри воркеров без этого хука остаются несидированными —
    классическая дыра в воспроизводимости при num_workers > 0.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Отдельный генератор для shuffle в DataLoader.

    Изолирует порядок семплирования от глобального torch-генератора:
    добавление dropout'а в модель не изменит порядок батчей.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
