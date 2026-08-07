"""Единственный модуль, знающий про torch.distributed.

Остальной код работает с DDP через объекты: 
1) `DistInfo` (кто я в группе и на какой карте считаю) 
2) вспомогательные функции для коллективных операций

Важное решение: **однопроцессный запуск тоже поднимает группу**, просто из
одного участника. 
"""

import logging
import os
import socket
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, overload

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from src.utils.device import resolve_device

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistInfo:
    # указатель на процесс в рамках Кластеров (rank=0 — лидер процессов)
    rank: int

    # указатель внутри одного Кластера
    local_rank: int

    # кол-во процессов в группе
    world_size: int

    # Устройство, где происходят вычисления
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def find_free_port() -> int:
    """Ищем свободный TCP-порт для rendezvous.
        IP уже знаем, нужно найти порт
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0)) # привязываем сокет к новому адресу.
        return sock.getsockname()[1] # возвращаем порт


def setup(device_cfg: str = "auto", backend: str = "auto", timeout_minutes: int = 30) -> DistInfo:
    """Подключает процесс к группе и возвращает его параметры"""

    # setdefault нужны для обычного запуска python scripts/train.py на один процесс
    # при групповом запуске переменные окружения перезапишут дефолты
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    # если один процесс (1 GPU), то не будем выходить за пределы сервера
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1") 
    os.environ.setdefault("MASTER_PORT", str(find_free_port()))

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Явный индекс карты в конфиге под DDP (cuda:2) = всегда ошибка: 
    # все ранки сядут на одну карту. 
    # ловим эту ошибку здесь
    if world_size > 1 and device_cfg not in ("auto", "cuda", "cpu"):
        requested = torch.device(device_cfg)
        if requested.type == "cuda" and requested.index is not None:
            raise ValueError(
                f"device={device_cfg} несовместимо с распределённым запуском "
                f"(world_size={world_size}): все ранки заняли бы одну карту. "
                f"Используй device=auto — каждый процесс возьмёт карту по своему "
                f"LOCAL_RANK, а выбор карт задаётся через CUDA_VISIBLE_DEVICES."
            )

    device = resolve_device(device_cfg, local_rank=local_rank)
    if device.type == "cuda":
        # ДО init_process_group: NCCL привязывает выч.устройство к текущему процессу
        # без этой строки все ранки узла работали бы
        # с cuda:0 независимо от того, что лежит в device.
        torch.cuda.set_device(device)

    if backend == "auto":
        backend = "nccl" if device.type == "cuda" else "gloo"

    if not dist.is_initialized():
        # Блокируется, пока все world_size процессов не дойдут до этой строки, —
        # первая точка синхронизации запуска.
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))

    log.info(
        "Распределённый запуск: rank=%d/%d, local_rank=%d, device=%s, backend=%s",
        rank, world_size, local_rank, device, backend,
    )
    return DistInfo(rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def cleanup() -> None:
    """закрытие группы и завершение процессов"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@overload
def unwrap(module: nn.Module) -> nn.Module: ...


@overload
def unwrap(module: None) -> None: ...

def unwrap(module: nn.Module | None) -> nn.Module | None:
    """Достаёт исходный модуль из-под DDP-обёртки.

       ВНИМАНИЕ: Получаем указатель на голую модель. 
       При изменении/forward не происходит синхронизации 
       с другими нодами и процессами.
       Работа с локальной моделью.

       None на входе допустим ради учителя, которого может не быть. Перегрузки
       выше нужны, чтобы это послабление не протекало в остальные вызовы:
       unwrap(student) статически остаётся Module, а не Module | None.

    """
    if isinstance(module, DistributedDataParallel):
        return module.module
    return module


def _is_active() -> bool:
    """Проверка на смысл в коллективных операциях.
    
    False в двух случаях: группа не поднята вовсе или
    в ней один участник. И там, и там all_reduce — тождественная операция,
    и вызывать её незачем.
    """
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


# на функции all_reduce просиходит синхронизация по всем процессам
def all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    """сумма по всем ранкам"""
    if _is_active():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def all_reduce_max_(tensor: torch.Tensor) -> torch.Tensor:
    """Максимум по всем ранкам"""
    if _is_active():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor


# ждём когда все ранки дойдут до этой точки
def barrier() -> None:
    """Создание барьера
        Остальные процессы из группы не могут пройти дальше
        пока не дошёл хотя бы один"""
    if _is_active():
        dist.barrier()


def broadcast_object(obj: Any, device: torch.device, src: int = 0) -> Any:
    """Рассылает произвольный python-объект с ранка src всем остальным.

    Медленнее тензорных операций, поэтому годится только
    для разовых вещей вроде согласования путей на старте -
    но не для метрик внутри цикла обучения. 
    Под капотом переводит объект в сырой массив байтов и перекидывает по шинам
    """
    if not _is_active():
        return obj
    holder = [obj]
    dist.broadcast_object_list(holder, src=src, device=device)
    return holder[0]
