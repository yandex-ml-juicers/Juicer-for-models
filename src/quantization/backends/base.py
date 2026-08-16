"""Общий интерфейс раннеров инференса"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import Tensor

log = logging.getLogger(__name__)

# quantize.backend
KINDS = ("torch", "onnxruntime", "tensorrt")


class Runner(ABC):
    """Одна модель, готовая считать: [runner.infer(batch) -> logits]"""

    name: str
    device: torch.device

    @abstractmethod
    def infer(self, batch: Tensor) -> Tensor:
        """Прогон одного батча. Вход и выход — torch-тензоры на self.device"""

    def close(self) -> None:
        """Освободить ресурсы (контексты, сессии). По умолчанию нечего"""

    def __enter__(self) -> "Runner":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def make_runner(kind: str, **kwargs) -> Runner:
    if kind == "torch":
        from src.quantization.backends.torch_backend import TorchRunner
        return TorchRunner(**kwargs)
    
    if kind == "onnxruntime":
        from src.quantization.backends.onnxruntime_backend import OnnxRuntimeRunner
        return OnnxRuntimeRunner(**kwargs)
    
    if kind == "tensorrt":
        from src.quantization.backends.tensorrt_backend import TensorRTRunner
        return TensorRTRunner(**kwargs)
    
    raise ValueError(f"Неизвестный бэкенд {kind!r}. Доступны: {KINDS}.")


def resolve_artifact(path: str | Path) -> Path:
    """Проверка существования артефакта с внятным сообщением"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Артефакт не найден: {path}. Стадия, которая его создаёт, не выполнялась "
            f"или писала в другую директорию."
        )
    return path
