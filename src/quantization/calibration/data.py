"""Калибровочная выборка для int8: фиксированный набор реальных батчей.

Почему это отдельный артефакт, а не поток прямо из DataLoader. Калибровка
задаёт масштабы квантования, то есть напрямую определяет качество движка.
Если выборка каждый раз новая, два движка из одного и того же `.onnx`
получаются разными, и разницу в метрике не с чем связать. Материализованный
`.npy` с sha256 в паспорте делает сборку воспроизводимой и снимает со стадии
build зависимость от датасета: пересобрать движок можно на машине, где данных
нет вовсе.

Что в выборке лежит: тензоры ПОСЛЕ препроцессинга, ровно в том виде, в каком
модель получает их на инференсе. Калибровка меряет диапазоны активаций, а они
зависят от нормировки — выборка с чужой нормировкой даст неверные масштабы и
тихую просадку метрики.
"""

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

CALIB_DIRNAME = "calib"
SAMPLES_NAME = "samples.npy"
MANIFEST_NAME = "samples.json"


@dataclass(frozen=True)
class CalibrationData:
    """Материализованная выборка: путь до `.npy` и её паспорт."""

    path: Path
    meta: dict

    @property
    def batch_size(self) -> int:
        return int(self.meta["batch_size"])

    @property
    def num_samples(self) -> int:
        return int(self.meta["num_samples"])

    @property
    def num_batches(self) -> int:
        return self.num_samples // self.batch_size

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return tuple(self.meta["sample_shape"])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return (self.batch_size, *self.sample_shape)

    def batches(self) -> Iterator[np.ndarray]:
        """Батчи по одному, через memmap.

        Держать выборку в памяти целиком незачем: 512 кадров сегментации в
        fp32 — это гигабайты, а калибратору за раз нужен ровно один батч.
        """
        array = np.load(self.path, mmap_mode="r")
        for start in range(0, self.num_batches * self.batch_size, self.batch_size):
            # np.array копирует срез в обычную память. Отдавать наружу сам
            # срез memmap нельзя: он только для чтения, и torch.from_numpy на
            # нём предупреждает о неопределённом поведении.
            yield np.array(array[start : start + self.batch_size], dtype=np.float32)


def calibration_dir(artifacts: str | Path) -> Path:
    """Папка калибровки внутри артефактов модели.

    Рядом с `.onnx`, а НЕ внутри `engines/<hw_tag>/`: и выборка, и таблица
    масштабов от железа не зависят — это свойства модели и данных. Движок,
    собранный на другой карте, переиспользует ту же калибровку.
    """
    return Path(artifacts) / CALIB_DIRNAME


def calibration_paths(artifacts: str | Path) -> tuple[Path, Path]:
    """(файл выборки, файл паспорта)."""
    directory = calibration_dir(artifacts)
    return directory / SAMPLES_NAME, directory / MANIFEST_NAME


def _channel_stats(samples: np.ndarray) -> dict:
    """Поканальные среднее/СКО и общий диапазон выборки.

    Дешёвая страховка от самой частой ошибки калибровки: выборка собрана с
    другой нормировкой, чем та, на которой модель обучалась. Ошибки не будет,
    масштабы просто окажутся не те — а в паспорте это видно сразу.
    """
    axes = (0, *range(2, samples.ndim))
    return {
        "channel_mean": np.mean(samples, axis=axes).round(4).tolist(),
        "channel_std": np.std(samples, axis=axes).round(4).tolist(),
        "min": float(samples.min()),
        "max": float(samples.max()),
    }


def collect_calibration_samples(
    loader: DataLoader,
    *,
    output_dir: str | Path,
    num_samples: int = 512,
    batch_size: int = 8,
    source: dict | None = None,
) -> CalibrationData:
    """Забирает из лоадера `num_samples` примеров и кладёт их в `.npy`.

    `batch_size` здесь — размер батча КАЛИБРОВКИ, он не обязан совпадать с
    батчем лоадера: TensorRT читает выборку своими порциями, и их размер
    участвует в калибровочном профиле движка.
    """
    from src.quantization.export import sha256_file

    if batch_size < 1:
        raise ValueError(f"batch_size={batch_size}: батч калибровки должен быть положительным.")
    if num_samples < batch_size:
        raise ValueError(
            f"num_samples={num_samples} меньше batch_size={batch_size}: "
            f"не наберётся даже одного батча."
        )

    requested = num_samples
    num_samples -= num_samples % batch_size
    if num_samples != requested:
        log.info(
            "Калибровка: %d примеров округлено вниз до %d — TensorRT читает выборку "
            "батчами по %d, неполный батч ему отдать нельзя.",
            requested, num_samples, batch_size,
        )

    collected: list[torch.Tensor] = []
    total = 0
    for batch in loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.detach().to("cpu", torch.float32)
        collected.append(images)
        total += images.shape[0]
        if total >= num_samples:
            break

    if not collected:
        raise ValueError("Лоадер калибровки не отдал ни одного батча.")

    samples = torch.cat(collected)[:num_samples].numpy()

    if samples.shape[0] < num_samples:
        available = samples.shape[0] - samples.shape[0] % batch_size
        if available == 0:
            raise ValueError(
                f"В выборке всего {samples.shape[0]} примеров, а батч калибровки "
                f"{batch_size}. Уменьши quantize.calibrate.batch_size."
            )
        log.warning(
            "Данные кончились раньше: набрано %d примеров вместо %d, беру %d. "
            "Масштабы квантования оценятся по меньшей выборке — это законно, но "
            "менее устойчиво.",
            samples.shape[0], num_samples, available,
        )
        samples = samples[:available]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / SAMPLES_NAME
    manifest_path = output_dir / MANIFEST_NAME

    np.save(samples_path, samples)

    meta = {
        "path": str(samples_path),
        "sha256": sha256_file(samples_path),
        "num_samples": int(samples.shape[0]),
        "batch_size": int(batch_size),
        "num_batches": int(samples.shape[0] // batch_size),
        "sample_shape": [int(dim) for dim in samples.shape[1:]],
        "size_bytes": samples_path.stat().st_size,
        "dtype": str(samples.dtype),
        "stats": _channel_stats(samples),
        "source": source or {},
    }
    manifest_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(
        "Калибровочная выборка: %d примеров %s (%.1f МиБ) -> %s",
        meta["num_samples"], tuple(meta["sample_shape"]),
        meta["size_bytes"] / 1024**2, samples_path,
    )
    log.info(
        "Поканальные статистики выборки: mean=%s std=%s, диапазон [%.3f, %.3f]. "
        "Если они не похожи на нормировку обучения — масштабы квантования будут не те.",
        meta["stats"]["channel_mean"], meta["stats"]["channel_std"],
        meta["stats"]["min"], meta["stats"]["max"],
    )
    return CalibrationData(path=samples_path, meta=meta)


def load_calibration_data(artifacts: str | Path) -> CalibrationData:
    """Читает ранее собранную выборку по паспорту рядом с ней."""
    samples_path, manifest_path = calibration_paths(artifacts)

    if not manifest_path.is_file() or not samples_path.is_file():
        raise FileNotFoundError(
            f"Калибровочной выборки нет: {samples_path}. Стадия calibrate не "
            f"выполнялась — запусти её вместе со сборкой:\n"
            f"  quantize.stages='[calibrate,build,validate,benchmark]'"
        )

    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    return CalibrationData(path=samples_path, meta=meta)


def matches_request(data: CalibrationData, *, num_samples: int, batch_size: int, source: dict) -> bool:
    """Годится ли лежащая выборка под текущий запрос.

    Сверяем и параметры, и источник: выборка, собранная с другого сплита или
    другого датасета, внешне неотличима, а масштабы даст другие.
    """
    if data.batch_size != batch_size:
        return False
    if data.num_samples != num_samples - num_samples % batch_size:
        return False
    return dict(data.meta.get("source") or {}) == dict(source)
