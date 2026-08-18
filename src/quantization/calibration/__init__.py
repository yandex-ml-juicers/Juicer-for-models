"""Калибровка под int8: сбор статистик активаций на репрезентативной выборке.

Две части, намеренно разделённые:
  * `data` — какие примеры считать репрезентативными и как зафиксировать их
    так, чтобы сборка была воспроизводимой (чистый torch/numpy, тестируется
    без GPU);
  * `tensorrt_calibrator` — как отдать их билдеру TensorRT и не пересчитывать
    масштабы на каждой сборке.
"""

from src.quantization.calibration.data import (
    CalibrationData,
    calibration_dir,
    calibration_paths,
    collect_calibration_samples,
    load_calibration_data,
    matches_request,
)
from src.quantization.calibration.tensorrt_calibrator import (
    Calibration,
    cache_is_usable,
    cache_sidecar,
    make_calibrator,
)

__all__ = [
    "Calibration",
    "CalibrationData",
    "cache_is_usable",
    "cache_sidecar",
    "calibration_dir",
    "calibration_paths",
    "collect_calibration_samples",
    "load_calibration_data",
    "make_calibrator",
    "matches_request",
]
