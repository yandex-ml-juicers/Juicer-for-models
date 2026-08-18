"""TensorRT-раннер поверх torch-тензоров, без pycuda"""

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.quantization.backends.base import Runner, resolve_artifact

log = logging.getLogger(__name__)

MIN_TRT_VERSION = (8, 5)

# Volta (V100). В 10.x объявлена устаревшей, в 11.x уже удалена.
VOLTA = (7, 0)

# Минимальная архитектура карты, начиная с версии TensorRT. Проверено на V100
# 2026-08-16:
#   11.2  — билдер вообще не создаётся: ассерт `SmVersion{0x0705} <= smVersion`
#           (0x0705 = sm_75), обёрнутый в невнятное pybind-nullptr;
#   10.16 — билдер создаётся, но buildSerializedNetwork возвращает пустой
#           движок с "Target GPU SM 70 is not supported by this TensorRT release".
# Где именно внутри 10.x выпилили Volta — не выяснено, поэтому версии ниже
# 10.16 пропускаем: там ошибка билдера уже внятная и скажет сама за себя.
SM_REQUIREMENTS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (((10, 16), (7, 5)),)


def tensorrt_cuda_variant() -> str | None:
    """Под какую CUDA собрано установленное колесо TensorRT: 'cu12', 'cu13', ...

    Пакет `tensorrt` — метапакет, а настоящие библиотеки лежат в
    `tensorrt_cu<N>_libs`. Иначе версию CUDA у TensorRT узнать неоткуда:
    `trt.__version__` говорит только про сам TensorRT.
    """
    from importlib.metadata import distributions

    for dist in distributions():
        name = (dist.metadata["Name"] or "").replace("-", "_").lower()
        if name.startswith("tensorrt_cu") and name.endswith("_libs"):
            return name.split("_")[1]
    return None


def unsupported_gpu(version: tuple[int, int], capability: tuple[int, int]) -> str | None:
    """Сообщение, если эта версия TensorRT уже не поддерживает карту, иначе None."""
    required = None
    for since, minimum in SM_REQUIREMENTS:
        if version >= since:
            required = minimum
    if required is None or capability >= required:
        return None

    card = f"sm_{capability[0]}{capability[1]}"
    return (
        f"TensorRT {version[0]}.{version[1]} не поддерживает {card}: ядер под эту "
        f"архитектуру в сборке нет, и настройками это не обходится.\n"
        f"Варианты, по убыванию скорости получения результата:\n"
        f"  1) quantize=torch_fp16 — fp16 средствами torch. Работает на этой карте "
        f"прямо сейчас, стадия build не нужна.\n"
        f"  2) quantize=ort_cuda — onnxruntime с CUDA EP (нужен onnxruntime-gpu). Пока "
        f"это fp32-граф на карте: fp16 там требует fp16-графа ONNX, отдельная работа.\n"
        f"  3) более старый TensorRT: 'pip install tensorrt-cu12==10.3.0' и ниже. "
        f"Точная версия, где выпилили {card}, не выяснена — проверять придётся перебором, "
        f"каждая установка это ~4 ГБ.\n"
        f"Проверить версию TensorRT, не трогая данные и не гоняя валидацию:\n"
        f"  python scripts/quantize.py <те же аргументы> quantize.stages='[build]'"
    )


def cuda_variant_mismatch(variant: str | None, torch_cuda: str | None) -> str | None:
    """Сообщение о несовпадении сборок TensorRT и torch по CUDA, иначе None."""
    if not variant or not torch_cuda:
        return None

    major = torch_cuda.split(".")[0]
    if variant == f"cu{major}":
        return None

    return (
        f"TensorRT собран под {variant}, а torch — под CUDA {torch_cuda}. Драйвер машины "
        f"поддерживает ту CUDA, под которую собран torch, поэтому инициализация TensorRT "
        f"провалится с cudaError 35 ('CUDA driver version is insufficient').\n"
        f"Нужна сборка под ту же CUDA:\n"
        f"  pip uninstall -y tensorrt tensorrt_{variant} "
        f"tensorrt_{variant}_bindings tensorrt_{variant}_libs\n"
        f"  pip install 'tensorrt-cu{major}'"
    )


def require_tensorrt() -> Any:
    """точка импорта TensorRT: ленивая, с проверкой версии и сборки под CUDA"""
    try:
        import tensorrt as trt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Нужен tensorrt, а он не установлен. Собирать и гонять движок имеет смысл "
            "только на целевой машине с GPU (у нас — сервер с V100). Локально доступны "
            "стадии export/validate/benchmark на бэкендах torch и onnxruntime."
        ) from error

    major, minor = (int(part) for part in trt.__version__.split(".")[:2])
    version = (major, minor)
    if version < MIN_TRT_VERSION:
        raise RuntimeError(
            f"TensorRT {trt.__version__} слишком старый: нужен "
            f">= {'.'.join(map(str, MIN_TRT_VERSION))}."
        )

    # `pip install tensorrt` тянет САМУЮ СВЕЖУЮ сборку — сейчас под CUDA 13.
    # Если драйвер и torch стоят на CUDA 12, TensorRT падает не здесь, а глубже,
    # в C++: "cudaError 35: CUDA driver version is insufficient" и следом
    # `pybind11::init(): factory function returned nullptr`. По такому сообщению
    # причину не угадать, поэтому ловим несовпадение заранее.
    mismatch = cuda_variant_mismatch(tensorrt_cuda_variant(), torch.version.cuda)
    if mismatch:
        raise RuntimeError(mismatch)

    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        unsupported = unsupported_gpu(version, capability)
        if unsupported:
            raise RuntimeError(unsupported)
        if capability == VOLTA:
            log.warning(
                "Карта — Volta (sm_70), а TensorRT %s из веток, где её уже выпиливают. "
                "Если сборка вернёт пустой движок с 'Target GPU SM 70 is not supported' — "
                "ставь версию младше или переходи на quantize=torch_fp16.",
                trt.__version__,
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
