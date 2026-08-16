"""Сборка TensorRT-движка из ONNX -- stage build PTQ-пайплайна"""

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
import torch

from src.quantization.backends.tensorrt_backend import require_tensorrt
from src.quantization.export import DynamicAxisSpec, sha256_file

log = logging.getLogger(__name__)

DEFAULT_WORKSPACE_BYTES = 2 * 1024**3

PRECISIONS = ("fp32", "fp16")


@dataclass(frozen=True)
class BuildResult:
    path: Path
    meta: dict


def hardware_tag(device: int | None = None) -> str:
    """Метка железа и тулчейна, например: `Tesla-V100-PCIE-32GB_sm70_trt10.7.0_cu126`.
    Должны собираться под конкретное железо, версию TRT"""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Сборка движка требует видимой CUDA-карты: тактики подбираются "
            "замерами на том железе, под которое собираем."
        )

    # По умолчанию текущее устройство процесса, а не нулевое
    device = torch.cuda.current_device() if device is None else device
    
    name = torch.cuda.get_device_name(device).replace(" ", "-")
    major, minor = torch.cuda.get_device_capability(device)
    cuda = (torch.version.cuda or "unknown").replace(".", "")

    try:
        import tensorrt as trt

        trt_version = trt.__version__
    except ModuleNotFoundError:
        trt_version = "none"

    return f"{name}_sm{major}{minor}_trt{trt_version}_cu{cuda}"


def onnx_inputs(onnx_path: str | Path) -> list[tuple[str, list[int | None]]]:
    """Входы графа: [(имя, [размерности])], где None — динамическая ось"""
    model = onnx.load(str(onnx_path), load_external_data=False)
    initializers = {tensor.name for tensor in model.graph.initializer}

    inputs = []
    for entry in model.graph.input:
        if entry.name in initializers:
            continue
        dims: list[int | None] = []
        for dim in entry.type.tensor_type.shape.dim:
            # dim_param — символьное имя ('batch'), dim_value — конкретное число.
            dims.append(dim.dim_value if dim.WhichOneof("value") == "dim_value" else None)
        inputs.append((entry.name, dims))
    return inputs


def profile_shapes(
    dims: Sequence[int | None],
    dynamic_axes: DynamicAxisSpec | None,
    opt_shape: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """(min, opt, max) для профиля оптимизации TensorRT"""
    spec = dict(dynamic_axes or {})

    unresolved = [i for i, dim in enumerate(dims) if dim is None and i not in spec]
    if unresolved:
        raise ValueError(
            f"В графе есть динамические оси {unresolved}, для которых не задан диапазон. "
            f"TensorRT не соберёт движок без профиля min/opt/max — укажи их в "
            f"quantize.export.dynamic_axes."
        )

    stale = [i for i in spec if i < len(dims) and dims[i] is not None]
    if stale:
        raise ValueError(
            f"Оси {stale} объявлены динамическими в конфиге, но в графе они статические "
            f"({[dims[i] for i in stale]}). Значит .onnx собран другим конфигом — "
            f"пересобери его, иначе профиль опишет не тот движок."
        )

    minimum, optimum, maximum = [], [], []
    for axis, dim in enumerate(dims):
        if dim is not None:
            minimum.append(dim)
            optimum.append(dim)
            maximum.append(dim)
            continue
        _, low, high = spec[axis]
        minimum.append(low)
        optimum.append(high)
        maximum.append(high)

    if opt_shape is not None:
        if len(opt_shape) != len(dims):
            raise ValueError(
                f"opt_shape={tuple(opt_shape)} не совпадает по рангу с входом графа "
                f"({len(dims)} осей)."
            )
        for axis, value in enumerate(opt_shape):
            if not minimum[axis] <= value <= maximum[axis]:
                raise ValueError(
                    f"opt_shape[{axis}]={value} вне диапазона профиля "
                    f"[{minimum[axis]}, {maximum[axis]}]."
                )
        optimum = list(opt_shape)

    return tuple(minimum), tuple(optimum), tuple(maximum)


def _make_logger(trt: Any, verbose: bool) -> Any:
    """Мост из логгера TensorRT в стандартный logging"""
    levels = {
        trt.Logger.INTERNAL_ERROR: logging.ERROR,
        trt.Logger.ERROR: logging.ERROR,
        trt.Logger.WARNING: logging.WARNING,
        trt.Logger.INFO: logging.INFO if verbose else logging.DEBUG,
        trt.Logger.VERBOSE: logging.DEBUG,
    }

    class _BridgeLogger(trt.ILogger):
        def __init__(self) -> None:
            trt.ILogger.__init__(self)

        def log(self, severity, message) -> None:
            log.log(levels.get(severity, logging.INFO), "TRT: %s", message)

    return _BridgeLogger()


def graph_precisions(network: Any) -> set[str]:
    """Какие типы реально встречаются на выходах слоёв разобранного графа"""
    found = set()
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        for output in range(layer.num_outputs):
            found.add(layer.get_output(output).dtype.name)
    return found


def _network_flags(trt: Any, precision: str) -> int:
    flags = 0
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    if explicit_batch is not None:
        flags |= 1 << int(explicit_batch)

    if precision == "fp16" and not hasattr(trt.BuilderFlag, "FP16"):
        strongly_typed = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
        if strongly_typed is not None:
            flags |= 1 << int(strongly_typed)
    return flags


def _apply_precision(trt: Any, builder: Any, config: Any, network: Any, precision: str) -> str:
    if precision == "fp32":
        return "fp32"

    flag = getattr(trt.BuilderFlag, "FP16", None)
    if flag is not None:
        if not getattr(builder, "platform_has_fast_fp16", True):
            log.warning("У карты нет быстрого fp16: движок соберётся, но ускорения не будет.")
        config.set_flag(flag)
        return "BuilderFlag.FP16"

    precisions = graph_precisions(network)
    if "HALF" not in precisions:
        raise RuntimeError(
            f"TensorRT {trt.__version__} не умеет слабо типизированные сети: в BuilderFlag "
            f"нет FP16, и точность берётся из типов графа. А в графе типы "
            f"{sorted(precisions)} — fp16 там нет, движок вышел бы обычным fp32.\n"
            f"Значит понижать точность нужно на стадии экспорта (fp16-граф ONNX), "
            f"а не на сборке."
        )
    return "типы графа (strongly typed)"


def _parse_onnx(trt: Any, network: Any, logger: Any, onnx_path: Path) -> None:
    parser = trt.OnnxParser(network, logger)
    if parser.parse_from_file(str(onnx_path)):
        return

    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    raise RuntimeError(
        "TensorRT не смог разобрать ONNX:\n  " + "\n  ".join(errors)
    )


def _layer_precision_summary(trt: Any, engine: Any, output_path: Path) -> dict | None:
    """Сколько слоёв реально собралось в fp16, а сколько осталось в fp32"""
    try:
        inspector = engine.create_engine_inspector()
        raw = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        info = json.loads(raw)
    except Exception as error:
        log.warning("Не удалось получить разбор слоёв у EngineInspector: %s", error)
        return None

    sidecar = output_path.with_suffix(".layers.json")
    sidecar.write_text(raw)

    layers = info.get("Layers", info) if isinstance(info, dict) else info
    if not isinstance(layers, list):
        return None

    summary: dict[str, int] = {}
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        precision = layer.get("Precision") or layer.get("precision") or "unknown"
        summary[str(precision)] = summary.get(str(precision), 0) + 1

    log.info("Слои движка по точности: %s (полный разбор: %s)", summary, sidecar.name)
    return summary


def build_engine(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    precision: str = "fp16", # флаг разрешает билдеру разные тактики (напр. fp16, fp32)
    dynamic_axes: DynamicAxisSpec | None = None,
    opt_shape: Sequence[int] | None = None, # по какому параметру оптимизируем (по умолчанию скорость)
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
    timing_cache_path: str | Path | None = None, # файл кэша замеров тактик. Общий на все сборки. Ускоряет повторные сборки
    verbose: bool = False, # поднять INFO-поток TRT из DEBUG в INFO
) -> BuildResult:
    """Собирает `.engine` из `.onnx` и пишет рядом паспорт сборки"""

    trt: Any = require_tensorrt()

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if precision not in PRECISIONS:
        raise ValueError(
            f"precision={precision!r} не поддержан. Доступно: {PRECISIONS}. "
            f"int8 требует калибровки - это стадия calibrate, её ещё нет."
        )

    inputs = onnx_inputs(onnx_path)
    if len(inputs) != 1:
        raise NotImplementedError(
            f"В графе {len(inputs)} входов ({[name for name, _ in inputs]}), а профиль "
            f"здесь строится для одного. Все наши модели - один тензор изображения; "
            f"под мультивход нужно расширять и профиль, и раннер."
        )
    input_name, dims = inputs[0]
    shape_min, shape_opt, shape_max = profile_shapes(dims, dynamic_axes, opt_shape)

    logger = _make_logger(trt, verbose)
    builder = trt.Builder(logger)
    network = builder.create_network(_network_flags(trt, precision))

    _parse_onnx(trt, network, logger, onnx_path)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    precision_route = _apply_precision(trt, builder, config, network, precision)

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, shape_min, shape_opt, shape_max)
    config.add_optimization_profile(profile)

    timing_cache = None
    if timing_cache_path is not None:
        timing_cache_path = Path(timing_cache_path)
        blob = timing_cache_path.read_bytes() if timing_cache_path.is_file() else b""
        timing_cache = config.create_timing_cache(blob)
        config.set_timing_cache(timing_cache, ignore_mismatch=False)

    log.info(
        "Сборка движка: %s -> %s | precision=%s | профиль %s..%s (opt %s) | workspace %.1f ГиБ",
        onnx_path.name,
        output_path.name,
        precision,
        shape_min,
        shape_max,
        shape_opt,
        workspace_bytes / 1024**3,
    )

    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started

    if serialized is None:
        raise RuntimeError(
            "TensorRT вернул пустой движок. Причина всегда выше в логе, в потоке "
            "'TRT: ...' -- чаще всего не хватило workspace или ни одна тактика не "
            "подошла под заданный профиль."
        )

    output_path.write_bytes(bytes(serialized))

    if timing_cache is not None and timing_cache_path is not None:
        timing_cache_path.parent.mkdir(parents=True, exist_ok=True)
        timing_cache_path.write_bytes(bytes(timing_cache.serialize()))

    engine = trt.Runtime(logger).deserialize_cuda_engine(bytes(serialized))
    if engine is None:
        raise RuntimeError("Движок собран, но не десериализуется")

    meta = {
        "path": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "precision": precision,
        "precision_route": precision_route,
        "graph_precisions": sorted(graph_precisions(network)),
        "source_onnx": str(onnx_path),
        "source_onnx_sha256": sha256_file(onnx_path),
        "hardware_tag": hardware_tag(),
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "compute_capability": ".".join(
            map(str, torch.cuda.get_device_capability(torch.cuda.current_device()))
        ),
        "tensorrt_version": trt.__version__,
        "cuda_version": torch.version.cuda,
        "input_name": input_name,
        "shape_min": list(shape_min),
        "shape_opt": list(shape_opt),
        "shape_max": list(shape_max),
        "workspace_bytes": workspace_bytes,
        "timing_cache": str(timing_cache_path) if timing_cache_path else None,
        "build_seconds": round(build_seconds, 1),
        "device_memory_bytes": getattr(
            engine, "device_memory_size_v2", getattr(engine, "device_memory_size", None)
        ),
        "layer_precisions": _layer_precision_summary(trt, engine, output_path),
    }

    # Паспорт пишет сама стадия, а не оркестратор: артефакт без паспорта
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    log.info(
        "Движок собран за %.1f с: %s (%.1f МиБ), паспорт — %s",
        build_seconds,
        output_path,
        meta["size_bytes"] / 1024**2,
        meta_path.name,
    )
    return BuildResult(path=output_path, meta=meta)
