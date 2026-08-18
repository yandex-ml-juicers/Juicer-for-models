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

PRECISIONS = ("fp32", "fp16", "int8")


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
            # Ошибки копим: настоящая причина падения приходит сюда, а наружу
            # C++ отдаёт nullptr или None. Без этого списка в исключении
            # остаётся только «смотри выше в логе».
            self.errors: list[str] = []

        def log(self, severity, message) -> None:
            if severity in (trt.Logger.ERROR, trt.Logger.INTERNAL_ERROR):
                self.errors.append(str(message))
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


def _apply_int8(trt: Any, builder: Any, config: Any, *, fp16_fallback: bool = True) -> str:
    """Флаги под int8. Сами масштабы придут из калибратора, а не отсюда.

    `fp16_fallback` — про то, что делать со слоями, которым int8 не идёт.
    По умолчанию билдеру разрешено выбрать для них fp16, и это правильный
    деплойный режим: он даёт лучший движок из возможных. Но и худший
    инструмент измерения — на карте, где fp16 быстрее int8 (любая Volta),
    билдер выберет fp16 почти везде, и «замер int8» окажется замером fp16.
    Чтобы узнать цену именно int8, фолбэк надо выключить.
    """
    flag = getattr(trt.BuilderFlag, "INT8", None)
    if flag is None:
        raise RuntimeError(
            f"В TensorRT {trt.__version__} нет BuilderFlag.INT8: неявная квантизация "
            f"с калибратором из этой ветки удалена. int8 там задаётся явно, узлами "
            f"QuantizeLinear/DequantizeLinear в самом ONNX, — это работа стадии "
            f"экспорта, а не сборки."
        )

    if not getattr(builder, "platform_has_fast_int8", True):
        log.warning(
            "У карты нет быстрого int8: движок соберётся, но ускорения не будет."
        )
    config.set_flag(flag)
    route = "BuilderFlag.INT8"

    if fp16_fallback:
        fp16 = getattr(trt.BuilderFlag, "FP16", None)
        if fp16 is not None:
            config.set_flag(fp16)
            route += " + FP16"
    return route


def _apply_precision(
    trt: Any,
    builder: Any,
    config: Any,
    network: Any,
    precision: str,
    *,
    int8_fp16_fallback: bool = True,
) -> str:
    if precision == "fp32":
        return "fp32"

    if precision == "int8":
        return _apply_int8(trt, builder, config, fp16_fallback=int8_fp16_fallback)

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


DEFAULT_BUILD_FAILURE = (
    "TensorRT вернул пустой движок — чаще всего не хватило workspace или ни одна "
    "тактика не подошла под заданный профиль."
)

# Отказы билдера, у которых причина известна точно. Общая формулировка про
# workspace и тактики в этих случаях вредна: она посылает крутить настройки
# там, где настройки ни при чём. Проверено на SegNeXt — три прогона ушло на
# перебор флагов, прежде чем стало ясно, что упирается в ограничение TensorRT.
KNOWN_BUILD_FAILURES: tuple[tuple[str, str, str], ...] = (
    (
        "replaceFillNodesForMyelin",
        "int8",
        "Сборка int8 несовместима со случайной генерацией внутри графа.\n"
        "В ONNX есть узел RandomUniformLike (в наших моделях это `torch.rand` в\n"
        "NMF-разложении SegNeXt, вызываемый на каждом forward). Такие узлы\n"
        "TensorRT исполняет только внутри Myelin и при сборке заменяет, ожидая\n"
        "их там; с флагом INT8 граф режется иначе, узел уезжает из Myelin — и\n"
        "билдер падает на внутреннем ассерте.\n"
        "Настройками это не лечится: проверено, что не помогают ни\n"
        "int8_fp16_fallback (оба значения), ни расширение fp32_layers.\n"
        "Лечится в модели: детерминированные базисы в eval вместо torch.rand —\n"
        "тогда RandomUniformLike уходит из графа. fp16 при этом собирается\n"
        "(quantize=trt_fp16_seg 'quantize.build.fp32_layers=[bmm]').",
    ),
)


def diagnose_build_failure(errors: Sequence[str], precision: str) -> str:
    """Расшифровка отказа билдера, если причина среди известных."""
    haystack = "\n".join(errors)
    for marker, affected_precision, diagnosis in KNOWN_BUILD_FAILURES:
        if marker in haystack and precision == affected_precision:
            return diagnosis
    return DEFAULT_BUILD_FAILURE


def build_options(
    *,
    precision: str,
    opt_shape: Sequence[int] | None = None,
    fp32_layers: Sequence[str] | None = None,
    fp32_margin: int = 0,
    calibration_algorithm: str | None = None,
    int8_fp16_fallback: bool | None = None,
) -> dict:
    """Настройки, от которых движок зависит ПО СУЩЕСТВУ.

    Пишется в паспорт и сверяется при переиспользовании. Без этого
    `quantize.build.fp32_layers=[bmm]` поверх уже собранного движка молча
    вернул бы старый: и веса, и .onnx те же, а защиты точности, ради которой
    флаг и ставили, в нём нет. Ровно та же ловушка ждёт переключение
    int8_fp16_fallback — эксперимент выглядел бы проведённым.

    Того, что на движок не влияет (verbose, timing_cache) или уже зашито в
    путь (precision, железо), здесь нет: лишние ключи заставляли бы
    пересобирать движок на ровном месте.
    """
    options: dict = {
        "opt_shape": list(opt_shape) if opt_shape is not None else None,
        "fp32_layers": sorted(fp32_layers or []),
        "fp32_margin": int(fp32_margin),
    }
    if precision == "int8":
        options["calibration_algorithm"] = calibration_algorithm
        options["int8_fp16_fallback"] = bool(int8_fp16_fallback)
    return options


def _attach_calibrator(
    trt: Any,
    builder: Any,
    config: Any,
    *,
    calibration: Any,
    cache_path: Path | None,
    algorithm: str,
    onnx_sha256: str,
    input_name: str,
    shape_min: Sequence[int],
    shape_max: Sequence[int],
    device: str = "cuda",
) -> Any:
    """Подключает калибратор к сборке и фиксирует форму калибровочного батча."""
    if calibration is None:
        raise ValueError(
            "precision=int8 требует калибровочной выборки. Веса билдер квантует сам, "
            "а диапазоны АКТИВАЦИЙ зависят от данных, и взять их неоткуда, кроме как "
            "прогнав модель на реальных примерах.\n"
            "Запусти стадию calibrate: quantize.stages='[export,calibrate,build,validate,benchmark]'."
        )

    from src.quantization.calibration import make_calibrator

    handle = make_calibrator(
        trt,
        calibration,
        cache_path=cache_path,
        algorithm=algorithm,
        onnx_sha256=onnx_sha256,
        device=device,
    )

    shape = tuple(handle.shape)
    if len(shape) != len(shape_min) or any(
        not low <= dim <= high for dim, low, high in zip(shape, shape_min, shape_max)
    ):
        raise ValueError(
            f"Форма калибровочного батча {shape} не влезает в профиль движка "
            f"{tuple(shape_min)}..{tuple(shape_max)}. Калибровка идёт через тот же вход, "
            f"что и инференс, поэтому форма обязана быть допустимой — правь "
            f"quantize.calibrate.batch_size."
        )

    config.int8_calibrator = handle.calibrator

    # Динамическому входу нужен ОТДЕЛЬНЫЙ профиль калибровки. Без него
    # TensorRT берёт kOPT первого профиля оптимизации (у нас это максимум
    # диапазона, обычно 64) и требует от калибратора батчи ровно такого
    # размера — а он отдаёт свои. Фиксируем профиль по форме выборки.
    set_calibration_profile = getattr(config, "set_calibration_profile", None)
    if set_calibration_profile is not None:
        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, shape, shape, shape)
        set_calibration_profile(profile)

    log.info(
        "Калибровка int8: %d батчей %s, алгоритм %s%s",
        handle.meta["num_batches"], shape, algorithm,
        ", масштабы из кэша" if handle.meta["cache_reused"] else "",
    )
    return handle


def constrain_layer_precision(
    trt: Any, network: Any, patterns: Sequence[str], span: bool = True, margin: int = 0
) -> dict:
    """Заставляет TensorRT считать выбранные слои в fp32.

    Зачем. Защита точности, написанная в модели на уровне PyTorch
    (`torch.autocast(enabled=False)`, `x.float()`), до движка не доезжает: в
    ONNX она становится обычными узлами `Cast`, а билдер в слабо
    типизированном режиме вправе их игнорировать. Проверено на SegNeXt —
    NMF-разложение уехало в fp16 и выдало NaN. Требовать точность надо здесь,
    у билдера.

    `span=True` — ключевой режим. Опасны не отдельные операции, а цепочка:
    если `bmm` посчитан в fp32, а следующее за ним деление осталось в fp16,
    результат всё равно уедет в NaN, потому что знаменатель там почти ноль по
    построению. Поэтому шаблоны работают якорями: берётся диапазон от первого
    совпадения до последнего, и всё между ними уходит в fp32.

    Имена слоёв здесь — ДО слияния (то, что дал парсер ONNX), а не те, что
    видны в `<движок>.layers.json` после сборки. Совпадают только исходные
    имена узлов вроде `node_bmm_7`; myelin-идентификаторов (`myl309`) на этом
    этапе ещё не существует.
    """
    names = [network.get_layer(index).name for index in range(network.num_layers)]

    missing = [pattern for pattern in patterns
               if not any(pattern in name for name in names)]
    if missing:
        raise ValueError(
            f"Шаблоны {missing} не нашли ни одного слоя из {len(names)}. Молча оставить "
            f"это нельзя: значит защита точности не действует, а движок соберётся и будет "
            f"выдавать NaN. Проверь имена в <движок>.layers.json — они могли измениться "
            f"после переэкспорта."
        )

    matched = [index for index, name in enumerate(names)
               if any(pattern in name for pattern in patterns)]
    if span:
        # margin — запас по краям диапазона. Нужен потому, что опасны не только
        # операции между якорями, но и границы: каст результата обратно в fp16
        # стоит сразу ЗА последним якорем, и если значение вылезло за 65504,
        # получается Inf. Подобрать имя такого слоя нельзя — он безымянный
        # служебный, а вот отступить от якоря на несколько слоёв можно.
        selected = range(max(0, min(matched) - margin),
                         min(len(names), max(matched) + margin + 1))
    else:
        selected = matched

    constrained, skipped = [], []
    for index in selected:
        layer = network.get_layer(index)
        try:
            layer.precision = trt.float32
            for output in range(layer.num_outputs):
                layer.set_output_type(output, trt.float32)
            constrained.append(layer.name)
        except Exception as error:  # noqa: BLE001 — часть типов слоёв точность не принимает
            skipped.append(f"{layer.name}: {error}")

    summary = {
        "patterns": list(patterns),
        "span": span,
        "margin": margin,
        "layers_total": len(names),
        "layers_matched": len(matched),
        "layers_constrained": len(constrained),
        "index_range": [min(selected), max(selected)] if constrained else None,
        "skipped": skipped[:5],
    }
    log.info(
        "В fp32 переведено %d слоёв из %d (якорей %d, диапазон %s)%s",
        len(constrained), len(names), len(matched), summary["index_range"],
        f", пропущено {len(skipped)}" if skipped else "",
    )
    return summary


def _parse_onnx(trt: Any, network: Any, logger: Any, onnx_path: Path) -> None:
    parser = trt.OnnxParser(network, logger)
    if parser.parse_from_file(str(onnx_path)):
        return

    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    raise RuntimeError(
        "TensorRT не смог разобрать ONNX:\n  " + "\n  ".join(errors)
    )


# Отчёт инспектора называет один и тот же тип по-разному в зависимости от
# того, откуда он взят: у весов это "Half"/"Float", у тензоров — строка формата
# вида "Channel major FP16 format where channel % 8 == 0", где кроме типа
# закодирована ещё и раскладка. Сводим к одному словарю, иначе гистограмма
# точностей смешивает точность с раскладкой и не читается.
PRECISION_ALIASES = {
    "half": "FP16", "fp16": "FP16",
    "float": "FP32", "fp32": "FP32",
    "int8": "INT8", "int32": "INT32", "uint8": "UINT8",
    "bf16": "BF16", "fp8": "FP8", "bool": "BOOL",
}


# Точность, закодированная в имени тактики NVIDIA. Последняя зацепка для
# слоёв без весов на движке с ДИНАМИЧЕСКИМ входом: там инспектор пишет в
# Format/Datatype «N/A», потому что раскладка выбирается уже во время
# исполнения. Проверено на ResNet-18 (динамический батч): пять слоёв из
# двадцати шести иначе остаются без ответа вовсе.
#   *mma — ядра тензорных ядер: hmma половинные, imma целочисленные;
#   f16f16 / i8i8 — типы операндов в именах implicit-gemm ядер.
TACTIC_MARKERS = {
    "i8i8": "INT8", "igemm": "INT8", "imma": "INT8", "int8": "INT8",
    "f16f16": "FP16", "hgemm": "FP16", "hmma": "FP16",
    "f32f32": "FP32", "sgemm": "FP32",
}


def layer_precision(layer: dict) -> str:
    """В какой точности исполняется слой, по отчёту EngineInspector.

    Источники по убыванию надёжности:
      1. `Weights.Type` — у слоёв с весами это прямой ответ;
      2. формат тензоров — у Reformat, Pooling и поэлементных своих весов нет,
         а считаются они в типе того, что производят;
      3. имя тактики — когда движок собран под динамический вход и формат
         тензора ещё неизвестен.

    Если не ответил ни один — так и говорим. Прежняя формулировка «без весов»
    утверждала то, чего мы не проверяли: на деле это «точность определить не
    удалось», и путать одно с другим нельзя ровно там, где разбор слоёв и
    нужен — в вопросе «применился ли int8 вообще».
    """
    weights = layer.get("Weights")
    if isinstance(weights, dict) and weights.get("Type"):
        name = str(weights["Type"])
        return PRECISION_ALIASES.get(name.lower(), name)

    # Входы наравне с выходами: у Reformat выход бывает N/A, а вход — нет.
    for tensor in (*layer.get("Outputs", []), *layer.get("Inputs", [])):
        fmt = str(tensor.get("Format/Datatype", ""))
        if not fmt or "N/A" in fmt:
            continue
        # Формат при статическом входе выглядит как
        # "Channel major FP16 format where channel % 8 == 0" — раскладка нам
        # здесь не нужна, только тип.
        for token, precision in PRECISION_ALIASES.items():
            if token in fmt.lower():
                return precision
        return fmt

    tactic = str(layer.get("TacticName", "")).lower()
    for token, precision in TACTIC_MARKERS.items():
        if token in tactic:
            return precision

    return f"не определена ({layer.get('LayerType', '?')})"


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
    sidecar.write_text(raw, encoding="utf-8")

    layers = info.get("Layers", info) if isinstance(info, dict) else info
    if not isinstance(layers, list):
        return None

    summary: dict[str, int] = {}
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        precision = layer_precision(layer)
        summary[precision] = summary.get(precision, 0) + 1

    if not summary:
        log.warning(
            "Инспектор не отдал точности слоёв. Обычно это значит, что движок собран "
            "без detailed_layers: TensorRT по умолчанию хранит только имена слоёв."
        )
        return None

    log.info("Слои движка по точности: %s (полный разбор: %s)", summary, sidecar.name)

    unresolved = [
        str(layer.get("Name", "?"))
        for layer in layers
        if isinstance(layer, dict) and layer_precision(layer).startswith("не определена")
    ]
    if unresolved:
        log.info(
            "У %d слоёв точность определить не удалось (%s%s): ни весов, ни читаемого "
            "формата тензора, ни распознанной тактики. Обычно это служебные слои "
            "движка с динамическим входом. В вопросе «применилась ли точность» их "
            "надо считать неизвестными, а не отсутствующими — разбор в %s.",
            len(unresolved),
            ", ".join(unresolved[:3]),
            ", ..." if len(unresolved) > 3 else "",
            sidecar.name,
        )
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
    detailed_layers: bool = True, # хранить в движке разбор слоёв (иначе не узнать, что ушло в fp16)
    fp32_layers: Sequence[str] | None = None, # шаблоны имён слоёв, которые обязаны остаться в fp32
    fp32_margin: int = 0, # запас по краям диапазона: границы (касты) тоже бывают опасны
    calibration: Any | None = None, # CalibrationData: выборка для int8, обязательна при precision=int8
    calibration_cache: str | Path | None = None, # таблица масштабов; переживает пересборку
    calibration_algorithm: str = "entropy2", # entropy2 для свёрток, minmax для трансформеров
    int8_fp16_fallback: bool = True, # разрешить билдеру уводить неудобные слои в fp16
) -> BuildResult:
    """Собирает `.engine` из `.onnx` и пишет рядом паспорт сборки"""

    trt: Any = require_tensorrt()

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if precision not in PRECISIONS:
        raise ValueError(
            f"precision={precision!r} не поддержан. Доступно: {PRECISIONS}."
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
    try:
        builder = trt.Builder(logger)
    except TypeError as error:
        # trt.Builder отдаёт nullptr, а pybind переводит это в невнятное
        # "factory function returned nullptr". Настоящая причина — в логгере.
        raise RuntimeError(
            "TensorRT не смог создать билдер. Причины бывают две: сборка TensorRT не под "
            "ту CUDA, что драйвер и torch, либо архитектура карты уже не поддерживается "
            "этой веткой TensorRT. Что сказал сам TensorRT:\n  "
            + "\n  ".join(logger.errors or ["(сообщений не было)"])
        ) from error
    network = builder.create_network(_network_flags(trt, precision))

    _parse_onnx(trt, network, logger, onnx_path)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    precision_route = _apply_precision(
        trt, builder, config, network, precision, int8_fp16_fallback=int8_fp16_fallback
    )

    # Ссылку держим до конца сборки: билдер зовёт калибратор из C++ и читает
    # его буфер по указателю. Сборщик мусора об этом не знает.
    calibration_handle = None
    if precision == "int8":
        calibration_handle = _attach_calibrator(
            trt, builder, config,
            calibration=calibration,
            cache_path=Path(calibration_cache) if calibration_cache is not None else None,
            algorithm=calibration_algorithm,
            onnx_sha256=sha256_file(onnx_path),
            input_name=input_name,
            shape_min=shape_min,
            shape_max=shape_max,
        )

    precision_constraints = None
    if fp32_layers:
        if precision == "fp32":
            log.info("fp32_layers не нужны: движок и так целиком в fp32.")
        else:
            precision_constraints = constrain_layer_precision(
                trt, network, list(fp32_layers), margin=fp32_margin
            )
            # OBEY, а не PREFER: PREFER при невозможности соблюсти ограничение
            # тихо откатывается — то есть возвращает ровно тот NaN, от которого
            # мы защищаемся, и молча. Пусть лучше падает сборка.
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    if detailed_layers:
        # По умолчанию TensorRT хранит в движке только имена слоёв, и инспектор
        # отдаёт пустой разбор. А без него нельзя ответить на главный вопрос
        # fp16-стадии: какие слои реально ушли в fp16, а какие остались в fp32.
        # На выбор тактик не влияет, растёт только объём метаданных.
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

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
            diagnose_build_failure(logger.errors, precision)
            + "\n\nЧто сказал сам TensorRT:\n  "
            + "\n  ".join(logger.errors or ["(сообщений не было)"])
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
        "precision_constraints": precision_constraints,
        "build_options": build_options(
            precision=precision,
            opt_shape=opt_shape,
            fp32_layers=fp32_layers,
            fp32_margin=fp32_margin,
            calibration_algorithm=calibration_algorithm,
            int8_fp16_fallback=int8_fp16_fallback,
        ),
        "calibration": calibration_handle.meta if calibration_handle is not None else None,
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
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(
        "Движок собран за %.1f с: %s (%.1f МиБ), паспорт — %s",
        build_seconds,
        output_path,
        meta["size_bytes"] / 1024**2,
        meta_path.name,
    )
    return BuildResult(path=output_path, meta=meta)
