"""TensorRT-раннер поверх torch-тензоров, без pycuda"""

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.quantization.backends.base import Runner, resolve_artifact

log = logging.getLogger(__name__)

MIN_TRT_VERSION = (8, 5)


def require_tensorrt() -> Any:
    """точка импорта TensorRT: ленивая, с проверкой версии"""
    try:
        import tensorrt as trt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Нужен tensorrt, а он не установлен. Собирать и гонять движок имеет смысл "
            "только на целевой машине с GPU (у нас — сервер с V100). Локально доступны "
            "стадии export/validate/benchmark на бэкендах torch и onnxruntime."
        ) from error

    version = tuple(int(part) for part in trt.__version__.split(".")[:2])
    if version < MIN_TRT_VERSION:
        raise RuntimeError(
            f"TensorRT {trt.__version__} слишком старый: нужен "
            f">= {'.'.join(map(str, MIN_TRT_VERSION))}."
        )
    return trt


def _torch_dtype(trt: Any, trt_dtype: Any) -> torch.dtype:
    return torch.from_numpy(np.empty(0, dtype=trt.nptype(trt_dtype))).dtype


class TensorRTRunner(Runner):
    def __init__(
        self,
        engine_path: str,
        *,
        device: torch.device | str = "cuda",
        name: str = "trt",
    ) -> None:
        trt = require_tensorrt()

        self._trt = trt
        self._closed = False
        self.path = resolve_artifact(engine_path)
        self.device = torch.device(device)
        self.name = name

        if self.device.type != "cuda":
            raise ValueError("TensorRT считает только на CUDA-устройстве.")

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise RuntimeError(
                f"Не удалось десериализовать {self.path}. Почти всегда это движок с чужого "
                f"железа или собранный другой версией TensorRT — сверь <имя>.meta.json "
                f"с текущей машиной."
            )
        self.context = self.engine.create_execution_context()

        inputs, outputs = [], []
        for index in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(index)
            target = (
                inputs
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
                else outputs
            )
            target.append(tensor_name)

        if len(inputs) != 1 or len(outputs) != 1:
            raise NotImplementedError(
                f"Движок с {len(inputs)} входами и {len(outputs)} выходами; раннер "
                f"рассчитан на один тензор туда и один обратно."
            )
        self.input_name, self.output_name = inputs[0], outputs[0]
        self.input_dtype = _torch_dtype(trt, self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = _torch_dtype(trt, self.engine.get_tensor_dtype(self.output_name))
        self.profile = self.engine.get_tensor_profile_shape(self.input_name, 0)

        log.info(
            "Движок загружен: %s | вход %s %s, профиль %s..%s | выход %s %s",
            self.path.name,
            self.input_name,
            self.input_dtype,
            tuple(self.profile[0]),
            tuple(self.profile[-1]),
            self.output_name,
            self.output_dtype,
        )

    def _check_shape(self, batch: Tensor) -> None:
        low, _, high = self.profile
        shape = tuple(batch.shape)
        if len(shape) != len(low) or any(
            not lo <= dim <= hi for dim, lo, hi in zip(shape, low, high)
        ):
            raise ValueError(
                f"Форма батча {shape} вне профиля движка {tuple(low)}..{tuple(high)}. "
                f"Движок собран под другой диапазон — либо меняй batch_size, либо "
                f"пересобирай движок с нужным профилем."
            )

    def infer(self, batch: Tensor) -> Tensor:
        self._check_shape(batch)

        source = batch.to(device=self.device, dtype=self.input_dtype, non_blocking=True)
        source = source.contiguous()

        self.context.set_input_shape(self.input_name, tuple(source.shape))
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        destination = torch.empty(output_shape, dtype=self.output_dtype, device=self.device)

        self.context.set_tensor_address(self.input_name, source.data_ptr())
        self.context.set_tensor_address(self.output_name, destination.data_ptr())

        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError("execute_async_v3 вернул False — движок не отработал батч.")

        return destination.float()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        del self.context
        del self.engine
