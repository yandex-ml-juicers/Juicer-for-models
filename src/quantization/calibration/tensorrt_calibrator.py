"""Калибратор TensorRT: отдаёт билдеру батчи и хранит таблицу масштабов"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.quantization.calibration.data import CalibrationData

log = logging.getLogger(__name__)

# Алгоритмы выбора масштаба по гистограмме активаций.
#   entropy2 — минимизирует KL между исходным и квантованным распределением;
#              отсекает хвосты ради точности в плотной части. Стандартный
#              выбор для свёрточных сетей.
#   minmax   — берёт абсолютный максимум, ничего не отсекает. Нужен там, где
#              редкий выброс несёт смысл (внимание в трансформерах): entropy
#              его срежет, и модель развалится.
CALIBRATORS = {
    "entropy2": "IInt8EntropyCalibrator2",
    "minmax": "IInt8MinMaxCalibrator",
    "entropy": "IInt8EntropyCalibrator",
    "legacy": "IInt8LegacyCalibrator",
}


@dataclass
class Calibration:
    """Готовый калибратор плюс то, что о нём нужно знать сборке и отчёту."""

    calibrator: Any
    shape: tuple[int, ...]
    meta: dict


def cache_sidecar(cache_path: Path) -> Path:
    """Паспорт кэша: сам файл кэша — это голая таблица чисел без происхождения."""
    return cache_path.with_suffix(cache_path.suffix + ".json")


def cache_is_usable(cache_path: Path | None, expected: dict) -> bool:
    """Можно ли взять готовые масштабы, не пересчитывая.

    Кэш TensorRT не помнит, из какой сети и каких данных он получен. Если
    подложить его к другому `.onnx`, сборка пройдёт, а движок будет
    квантован по чужим диапазонам — молча и неотличимо от исправного.
    Поэтому решает не наличие файла, а совпадение паспорта.
    """
    if cache_path is None or not cache_path.is_file():
        return False

    sidecar = cache_sidecar(cache_path)
    if not sidecar.is_file():
        log.warning(
            "Кэш калибровки %s есть, а паспорта рядом нет — происхождение масштабов "
            "неизвестно, пересчитываю.", cache_path.name,
        )
        return False

    stored = json.loads(sidecar.read_text(encoding="utf-8"))
    diverged = [key for key, value in expected.items() if stored.get(key) != value]
    if diverged:
        log.info(
            "Кэш калибровки не подходит (разошлись: %s) — пересчитываю масштабы.",
            ", ".join(diverged),
        )
        return False
    return True


def make_calibrator(
    trt: Any,
    data: CalibrationData,
    *,
    cache_path: str | Path | None = None,
    algorithm: str = "entropy2",
    device: torch.device | str = "cuda",
    onnx_sha256: str | None = None,
) -> Calibration:
    """Калибратор поверх материализованной выборки.

    `trt` передаётся аргументом, а не импортируется: точка импорта TensorRT в
    проекте одна (`backends.tensorrt_backend.require_tensorrt`), и здесь она
    не нужна — зато так модуль тестируется двойником, без GPU.
    """
    if algorithm not in CALIBRATORS:
        raise ValueError(
            f"algorithm={algorithm!r} неизвестен. Доступны: {sorted(CALIBRATORS)}. "
            f"Для свёрточных сетей — entropy2, для трансформеров — minmax."
        )

    base = getattr(trt, CALIBRATORS[algorithm], None)
    if base is None:
        raise RuntimeError(
            f"В TensorRT {getattr(trt, '__version__', '?')} нет {CALIBRATORS[algorithm]}: "
            f"неявная квантизация через калибратор из этой ветки удалена. int8 там "
            f"задаётся явно — узлами QuantizeLinear/DequantizeLinear в самом ONNX "
            f"(NVIDIA ModelOpt или onnxruntime.quantization). Это работа на стадии "
            f"экспорта, а не сборки; для V100 рабочая связка — tensorrt-cu12==10.3.0."
        )

    cache_path = Path(cache_path) if cache_path is not None else None
    expected = {
        "algorithm": algorithm,
        "samples_sha256": data.meta.get("sha256"),
        "num_samples": data.num_samples,
        "batch_size": data.batch_size,
        "onnx_sha256": onnx_sha256,
    }
    reuse = cache_is_usable(cache_path, expected)
    total = data.num_batches

    class _Calibrator(base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            self._batches = data.batches()
            self._served = 0
            # Буфер один на всю калибровку и живёт в атрибуте: TensorRT
            # получает голый указатель и читает по нему уже после возврата из
            # get_batch. Локальный тензор к тому моменту был бы освобождён, и
            # билдер прочитал бы чужую память — без ошибки, просто мусор.
            self._buffer = torch.empty(data.batch_shape, dtype=torch.float32, device=device)

        def get_batch_size(self) -> int:
            return data.batch_size

        def get_batch(self, names, *_) -> list[int] | None:
            try:
                chunk = next(self._batches)
            except StopIteration:
                # None означает «выборка кончилась»: билдер переходит к
                # построению гистограмм. Это штатное завершение, не ошибка.
                return None

            self._buffer.copy_(torch.from_numpy(chunk))
            self._served += 1
            if self._served == 1 or self._served % 10 == 0 or self._served == total:
                log.info("Калибровка: батч %d из %d", self._served, total)
            return [int(self._buffer.data_ptr())]

        def read_calibration_cache(self) -> bytes | None:
            if not reuse or cache_path is None:
                return None
            log.info(
                "Масштабы квантования берутся из кэша %s — выборка совпала, "
                "пересчитывать нечего.", cache_path.name,
            )
            return cache_path.read_bytes()

        def write_calibration_cache(self, cache) -> None:
            if cache_path is None:
                return
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(bytes(cache))
            cache_sidecar(cache_path).write_text(
                json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            log.info("Таблица масштабов сохранена: %s", cache_path)

    meta = {
        "algorithm": algorithm,
        "num_samples": data.num_samples,
        "num_batches": total,
        "batch_shape": list(data.batch_shape),
        "samples": str(data.path),
        "samples_sha256": data.meta.get("sha256"),
        "cache": str(cache_path) if cache_path is not None else None,
        "cache_reused": reuse,
    }
    return Calibration(calibrator=_Calibrator(), shape=data.batch_shape, meta=meta)
