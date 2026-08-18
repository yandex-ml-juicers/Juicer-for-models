"""Пайплайн PTQ: fp32 -> fp16"""

import copy
import json
import logging
import os
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data import base_loader
from src.quantization.backends.base import make_runner
from src.quantization.benchmark import (
    benchmark_runner,
    check_measurement_sanity,
    log_speedup_summary,
    speedup_table,
)
from src.quantization.export import export_onnx, sha256_file
from src.quantization.numerics import compare_runners, evaluate_runner
from src.quantization.report import QuantizationReport, check_acceptance
from src.utils import resolve_device, seed_everything
from src.utils.checkpoints import load_checkpoint_into
from src.utils.seed import make_generator, seed_worker
from typing import Any

log = logging.getLogger(__name__)

STAGES = ("export", "calibrate", "build", "validate", "benchmark")


def check_single_process() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        raise RuntimeError(
            f"scripts/quantize.py запущен под распределённым лаунчером "
            f"(WORLD_SIZE={world_size}). Сборка движка и замер производительности "
            f"должны идти в одном процессе: python scripts/quantize.py ..."
        )


def get_normalize_stats(cfg: DictConfig | Any) -> tuple | None:
    for node in (cfg.get("data", {}).get("dataset", {}), cfg.get("data", {})):
        stats = node.get("normalize") if node else None
        if stats:
            return tuple(stats.mean), tuple(stats.std)
    return None


def find_training_snapshot(cfg: DictConfig) -> Path | None:
    """Снапшот конфига того запуска, которым обучены веса.

    Ищем в двух местах: явный `quantize.source_run` и — если его нет — рядом
    с чекпоинтом. Второе важнее первого: `.hydra/config.yaml` лежит в той же
    папке, что и `best.pt`, то есть доступен ВСЕГДА, даже когда пайплайн
    запущен через `experiment=`. Именно так ловится расхождение нормировки,
    которое иначе выглядит как необъяснимая просадка метрики.
    """
    if cfg.quantize.source_run:
        path = Path(to_absolute_path(cfg.quantize.source_run)) / ".hydra" / "config.yaml"
        if not path.is_file():
            raise FileNotFoundError(
                f"Снапшот конфига не найден: {path}. source_run должен указывать "
                f"на папку запуска обучения (в ней лежит .hydra/config.yaml)."
            )
        return path

    if cfg.ckpt_path is None:
        return None
    path = Path(to_absolute_path(cfg.ckpt_path)).parent / ".hydra" / "config.yaml"
    return path if path.is_file() else None


def warn_on_normalize_drift(cfg: DictConfig, snapshot: DictConfig | Any) -> None:
    """Сверяет нормировку входа с той, на которой модель обучалась.

    Расхождение бьёт по метрике на единицы процентов и при этом никак себя не
    проявляет: ошибки нет, модель считает, числа просто хуже. На сравнение
    fp32 с fp16 не влияет (вход у обоих один), но абсолютную метрику делает
    непригодной для отчёта.
    """
    current = get_normalize_stats(cfg)
    trained = get_normalize_stats(snapshot)
    if not current or not trained or current == trained:
        return

    log.warning(
        "Нормировка входа разошлась с обучением!\n"
        "  обучали:    mean=%s std=%s\n"
        "  валидируем: mean=%s std=%s\n"
        "Абсолютная метрика будет занижена, и fp16 тут ни при чём. Сравнение "
        "fp32 с fp16 при этом корректно: вход у обоих одинаковый.",
        trained[0], trained[1], current[0], current[1],
    )


def build_student(cfg: DictConfig) -> torch.nn.Module:
    """Собирает ученика и грузит в него веса

    Архитектура подтягивается из снапшота обучающего запуска
    (если  задан 'quantize.source_run) или из текущего configs/
    """
    model_cfg = cfg.model.student
    snapshot_path = find_training_snapshot(cfg)

    if snapshot_path is not None:
        snapshot = OmegaConf.load(snapshot_path)
        warn_on_normalize_drift(cfg, snapshot)
        if cfg.quantize.source_run:
            model_cfg = snapshot.model.student
            log.info("Архитектура ученика взята из снапшота %s", snapshot_path)

    try:
        student = instantiate(model_cfg)
    except Exception as error:
        raise RuntimeError(
            f"Не удалось собрать модель по конфигу {OmegaConf.to_container(model_cfg)}. "
            f"Снапшот фиксирует конфиг, но не код: если фабрика с тех пор "
            f"переименована или удалена, собирай через experiment=<пресет> "
            f"без quantize.source_run."
        ) from error

    if cfg.ckpt_path is None:
        raise ValueError("Нужны веса: ckpt_path=outputs/<name>/<run>/best.pt")

    load_checkpoint_into(student, to_absolute_path(cfg.ckpt_path), "student")
    return student.eval()


def take_sample(cfg: DictConfig, get_loader, device: torch.device) -> torch.Tensor:
    batch_size = int(cfg.quantize.export.batch_size)
    source = cfg.quantize.export.sample_source

    if source == "loader":
        images = next(iter(get_loader()))[0]
        while images.shape[0] < batch_size:
            images = torch.cat([images, images], dim=0)
        return images[:batch_size].to(device)

    if source == "random":
        size = cfg.data.dataset.image_size
        if size is None:
            raise ValueError(
                "sample_source=random требует data.dataset.image_size, а он не задан. "
                "Возьми образец из лоадера (sample_source=loader) или задай размер явно."
            )
        height, width = (size, size) if isinstance(size, int) else tuple(size)
        return torch.randn(batch_size, 3, height, width, device=device)

    raise ValueError(f"sample_source={source!r}; доступны 'loader' и 'random'.")


def get_dynamic_axes(cfg: DictConfig) -> dict[int, tuple[str, int, int]]:
    """Список из конфига -> спека осей для экспорта и профиля TensorRT."""
    return {
        int(entry.axis): (str(entry.name), int(entry.min), int(entry.max))
        for entry in cfg.quantize.export.dynamic_axes or []
    }


def check_batch_fits_profile(cfg: DictConfig) -> None:
    """Влезает ли батч валидации в профиль движка.

    Профиль — деплойное решение (под какие размеры подбирать тактики), а
    batch_size лоадера приходит из конфига обучения, где он совсем про другое.
    Разъезжаются они регулярно, поэтому сверяем по конфигу — до того, как
    поднимется датасет: иначе о несовпадении узнаёшь через полминуты загрузки.
    """
    if cfg.quantize.backend != "tensorrt":
        return

    axes = get_dynamic_axes(cfg)
    if 0 in axes:
        _, low, high = axes[0]
    else:
        # Батч не объявлен динамическим — движок примет ровно тот размер, с
        # которым экспортировали. Для сегментации в полном разрешении это
        # нормальный режим, и промахнуться тут даже легче, чем с профилем.
        low = high = int(cfg.quantize.export.batch_size)

    allowed = f"[{low}, {high}]" if low != high else f"ровно {low} (вход статический)"
    stages = set(cfg.quantize.stages)

    # Замер сам задаёт размеры батчей и в лоадер не ходит вовсе, поэтому его
    # проверяем отдельно: иначе прогон без датасета требовал бы согласовывать
    # батч валидации, который ему не нужен.
    if "benchmark" in stages:
        outside = [size for size in cfg.quantize.benchmark.batch_sizes if not low <= size <= high]
        if outside:
            raise ValueError(
                f"Размеры батчей замера {outside} не подходят движку: он принимает "
                f"{allowed}. Правь quantize.benchmark.batch_sizes."
            )

    if cfg.quantize.precision == "int8" and "build" in stages:
        calib_batch = int(cfg.quantize.calibrate.batch_size)
        if not low <= calib_batch <= high:
            raise ValueError(
                f"Батч калибровки {calib_batch} не подходит движку: он принимает {allowed}. "
                f"Калибровка идёт через тот же вход, что и инференс — правь "
                f"quantize.calibrate.batch_size."
            )

    if "validate" not in stages:
        return

    loader = cfg.data.loader
    batch = int(loader.eval_batch_size or loader.batch_size)
    if low <= batch <= high:
        return

    raise ValueError(
        f"Батч валидации {batch} не подходит движку: он принимает {allowed}. "
        f"Либо приведи батч в соответствие:\n"
        f"  data.loader.eval_batch_size={high}\n"
        f"либо расширь профиль и пересобери движок:\n"
        f"  'quantize.export.dynamic_axes=[{{axis:0,name:batch,min:1,max:{batch}}}]' "
        f"quantize.reuse=false\n"
        f"Первое обычно правильнее: профиль описывает то, как модель поедет в прод, "
        f"а батч лоадера — как удобнее считать метрику."
    )


def write_source_manifest(path: Path, cfg: DictConfig, checkpoint: Path) -> dict:
    manifest = {
        "name": cfg.name,
        "task_type": cfg.get("task_type", "classification"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_run": cfg.quantize.source_run,
        "dataset": cfg.data.dataset.get("_target_") or cfg.data.dataset.get("build", {}).get("_target_"),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def stage_export(cfg, student, get_loader, device, onnx_path, manifest, report) -> dict:
    """ONNX + встроенная L0-проверка (torch против onnxruntime)."""
    settings = cfg.quantize.export

    if cfg.quantize.reuse and onnx_path.is_file():
        previous = onnx_path.parent / "source.json"
        same = previous.is_file() and json.loads(previous.read_text(encoding="utf-8")).get(
            "checkpoint_sha256"
        ) == manifest["checkpoint_sha256"]
        if same:
            log.info("export: %s уже собран из этих же весов", onnx_path.name)
            report.stage("export", {"path": str(onnx_path), "reused": True})
            return {"path": str(onnx_path), "reused": True}
        log.info("export: артефакт есть, но веса другие")

    verify = bool(settings.get("verify", True))
    if not verify:
        log.warning(
            "L0-проверка экспорта выключена (quantize.export.verify=false). Расхождение "
            "движка с моделью теперь не с чем сопоставить: поломка экспорта и эффект "
            "точности станут неразличимы."
        )

    result = export_onnx(
        student.to("cpu"),
        sample=take_sample(cfg, get_loader, torch.device("cpu")),
        output_path=onnx_path,
        opset=int(settings.opset),
        dynamo=bool(settings.dynamo),
        dynamic_axes=get_dynamic_axes(cfg),
        external_data=settings.external_data,
        optimize=bool(settings.optimize),
        verify=verify,
        parity_atol=float(settings.parity_atol),
    )
    student.to(device)
    report.stage("export", result.meta)
    return result.meta


def calibration_source(cfg: DictConfig) -> dict:
    """Паспорт источника выборки: по нему решается, годится ли собранная ранее.

    Только JSON-совместимые типы: паспорт переживает запись на диск, и
    сравнивать его придётся уже после чтения.
    """
    normalize = get_normalize_stats(cfg)
    size = cfg.data.dataset.get("image_size")
    # image_size: [1024, 2048] приходит из Hydra как ListConfig — это не list,
    # isinstance его не ловит, а json.dumps не переваривает. Приводим к
    # примитивам по способности итерироваться, а не по типу.
    if size is not None and not isinstance(size, int):
        size = [int(dim) for dim in size]
    return {
        "split": str(cfg.quantize.calibrate.split),
        "dataset": cfg.data.dataset.build.get("_target_"),
        "transform": cfg.data.transform.eval.get("_target_"),
        "image_size": size,
        "normalize": [list(normalize[0]), list(normalize[1])] if normalize else None,
        "seed": int(cfg.seed),
    }


def calibration_loader(cfg: DictConfig) -> DataLoader:
    """Лоадер под калибровку: картинки одного сплита, препроцессинг инференса.

    Три решения, каждое из которых влияет на масштабы квантования сильнее,
    чем кажется.

    **Сплит по умолчанию train, а не eval.** Калибровка подбирает диапазоны
    под данные. Подбирать их по той же выборке, на которой потом отчитываешься
    метрикой, — подгонка: результат окажется оптимистичным ровно там, где его
    и меряют.

    **Препроцессинг всегда eval, даже для train-сплита.** Калибровка обязана
    видеть активации такими, какими они будут в проде. Случайный кроп,
    отражение и цветовые сдвиги обучающей аугментации их смещают.

    **Перемешивание обязательно.** Датасеты разложены по классам подряд:
    первые 512 картинок ImageNet-100 — это два класса из ста, и калибровка
    оценила бы диапазоны по ним вместо всей задачи. Генератор фиксирован
    seed'ом, так что выборка при этом остаётся воспроизводимой.
    """
    settings = cfg.quantize.calibrate
    split = str(settings.split)
    if split not in ("train", "eval"):
        raise ValueError(f"quantize.calibrate.split={split!r}; доступны 'train' и 'eval'.")

    if cfg.get("task_type") == "detection":
        raise NotImplementedError(
            "Калибровка для детекции не поддержана: там свой collate_fn и батч не "
            "сводится к одному тензору изображений."
        )

    if split == "eval":
        log.warning(
            "Калибровка идёт по eval-сплиту — по той же выборке, на которой считается "
            "отчётная метрика. Масштабы подстроятся под неё, и просадка int8 выйдет "
            "заниженной. Годится для отладки, не для отчёта."
        )

    transform = instantiate(cfg.data.transform.eval)
    dataset = instantiate(cfg.data.dataset.build, train=(split == "train"), transform=transform)

    return DataLoader(
        dataset,
        batch_size=int(settings.batch_size),
        shuffle=True,
        generator=make_generator(cfg.seed),
        num_workers=int(cfg.data.loader.num_workers),
        pin_memory=False,
        worker_init_fn=seed_worker,
    )


def stage_calibrate(cfg, artifacts, report) -> dict:
    """Калибровочная выборка под int8; для fp32/fp16 — осознанный пропуск."""
    if cfg.quantize.precision != "int8":
        reason = (
            f"precision={cfg.quantize.precision}: ни масштабов, ни zero-point здесь нет, "
            f"статистики активаций собирать не для чего."
        )
        log.info("calibrate: пропуск - %s", reason)
        report.stage("calibrate", {"reason": reason}, status="skipped")
        return {"skipped": True}

    from src.quantization.calibration import (
        calibration_dir,
        calibration_paths,
        collect_calibration_samples,
        load_calibration_data,
        matches_request,
    )

    settings = cfg.quantize.calibrate
    num_samples = int(settings.num_samples)
    batch_size = int(settings.batch_size)
    source = calibration_source(cfg)
    samples_path, manifest_path = calibration_paths(artifacts)

    if cfg.quantize.reuse and samples_path.is_file() and manifest_path.is_file():
        data = load_calibration_data(artifacts)
        if matches_request(data, num_samples=num_samples, batch_size=batch_size, source=source):
            log.info(
                "calibrate: выборка уже собрана (%d примеров из %s) - переиспользую",
                data.num_samples, source["split"],
            )
            report.stage("calibrate", {**data.meta, "reused": True})
            return data.meta
        log.info("calibrate: выборка есть, но собрана под другой запрос - пересобираю")

    data = collect_calibration_samples(
        calibration_loader(cfg),
        output_dir=calibration_dir(artifacts),
        num_samples=num_samples,
        batch_size=batch_size,
        source=source,
    )
    report.stage("calibrate", data.meta)
    return data.meta


def engine_path_for(cfg: DictConfig, artifacts: Path) -> Path:
    """Путь до движка: `engines/<hw_tag>/model_<precision>.engine`"""
    from src.quantization.build_engine import hardware_tag

    return artifacts / "engines" / hardware_tag() / f"model_{cfg.quantize.precision}.engine"


def stage_build(cfg, onnx_path, artifacts, report) -> dict:
    """Сборка .engine движка"""
    if cfg.quantize.backend != "tensorrt":
        reason = f"backend={cfg.quantize.backend}: движок собирать нечем, quantize.backend != tensorrt"
        log.info("build: пропуск - %s", reason)
        report.stage("build", {"reason": reason}, status="skipped")
        return {"skipped": True}

    from src.quantization.build_engine import build_engine, build_options

    settings = cfg.quantize.build
    engine_path = engine_path_for(cfg, artifacts)

    opt_batch = settings.opt_batch
    opt_shape = None
    if opt_batch is not None:
        exported = report.payload["stages"].get("export", {}).get("input_shape")
        if exported is None:
            raise ValueError(
                "quantize.build.opt_batch задан, но форма входа неизвестна: стадия "
                "export в этом запуске не выполнялась. Добавь её в quantize.stages "
                "или убери opt_batch."
            )
        opt_shape = [int(opt_batch), *exported[1:]]

    options = build_options(
        precision=cfg.quantize.precision,
        opt_shape=opt_shape,
        fp32_layers=list(settings.get("fp32_layers") or []),
        fp32_margin=int(settings.get("fp32_margin") or 0),
        calibration_algorithm=str(cfg.quantize.calibrate.algorithm),
        int8_fp16_fallback=bool(settings.get("int8_fp16_fallback", True)),
    )

    calibration, calibration_cache = None, None
    if cfg.quantize.precision == "int8":
        from src.quantization.calibration import calibration_dir, load_calibration_data

        calibration = load_calibration_data(artifacts)
        # Кэш масштабов лежит рядом с выборкой, а не в engines/<hw_tag>/:
        # таблица диапазонов активаций — свойство модели и данных, от карты и
        # версии TensorRT она не зависит и переживает пересборку под другое железо.
        calibration_cache = (
            calibration_dir(artifacts) / f"scales_{cfg.quantize.calibrate.algorithm}.cache"
        )

    if cfg.quantize.reuse and engine_path.is_file():
        meta_path = engine_path.with_suffix(".meta.json")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            same_onnx = meta.get("source_onnx_sha256") == sha256_file(onnx_path)
            # У int8-движка исходников два. Те же веса, откалиброванные по
            # другой выборке, дают другие масштабы и другое качество — сверять
            # один только .onnx здесь недостаточно.
            same_calibration = calibration is None or (
                (meta.get("calibration") or {}).get("samples_sha256")
                == calibration.meta.get("sha256")
            )
            # Настройки сборки — третий исходник наравне с .onnx и выборкой.
            # Движок, собранный без fp32_layers, внешне неотличим от собранного
            # с ними: тот же путь, тот же размер, другое поведение.
            same_options = meta.get("build_options") == options
            if same_onnx and same_calibration and same_options:
                log.info("build: %s собран из этих же исходников - пропускаем", engine_path.name)
                report.stage("build", {**meta, "reused": True})
                return meta
        log.info(
            "build: движок есть, но собран из другого .onnx, другой выборки или с "
            "другими настройками - пересобираем"
        )

    result = build_engine(
        onnx_path,
        engine_path,
        precision=cfg.quantize.precision,
        dynamic_axes=get_dynamic_axes(cfg),
        opt_shape=opt_shape,
        workspace_bytes=int(float(settings.workspace_gb) * 1024**3),
        timing_cache_path=engine_path.parent / "timing.cache" if settings.timing_cache else None,
        verbose=bool(settings.verbose),
        detailed_layers=bool(settings.get("detailed_layers", True)),
        fp32_layers=list(options["fp32_layers"]),
        fp32_margin=int(options["fp32_margin"]),
        calibration=calibration,
        calibration_cache=calibration_cache,
        calibration_algorithm=str(cfg.quantize.calibrate.algorithm),
        int8_fp16_fallback=bool(settings.get("int8_fp16_fallback", True)),
    )
    report.stage("build", result.meta)
    return result.meta


def make_candidate_runner(cfg, student, onnx_path, artifacts, device):
    """Раннер кандидата под выбранный бэкенд. Копируем student"""
    backend = cfg.quantize.backend
    precision = cfg.quantize.precision

    if backend == "torch":
        return make_runner("torch", model=copy.deepcopy(student), precision=precision,
                           device=device, channels_last=bool(cfg.quantize.get("channels_last")))

    if backend == "onnxruntime":
        providers = OmegaConf.to_container(cfg.quantize.providers) if cfg.quantize.providers else None
        return make_runner("onnxruntime", onnx_path=str(onnx_path), providers=providers, device=device)

    if backend == "tensorrt":
        return make_runner("tensorrt", engine_path=str(engine_path_for(cfg, artifacts)),
                           device=device, name=f"trt_{precision}")

    raise ValueError(f"backend={backend!r} неизвестен.")


def stage_validate(cfg, reference, candidate, loader, device, report) -> dict:
    settings = cfg.quantize.validate
    task_type = cfg.get("task_type", "classification")
    dataset = cfg.data.dataset

    batches = int(settings.compare_batches)

    # Собственный шум модели: тот же fp32-раннер сравнивается САМ С СОБОЙ на
    # тех же батчах. Для детерминированной модели это ровно ноль, и строка в
    # отчёте лишний раз это подтверждает. А вот если внутри есть случайность
    # (SegNeXt инициализирует базисы NMF через torch.rand на каждом forward),
    # без этой величины расхождение fp16 не с чем сопоставить: можно принять
    # за эффект точности то, что модель творит сама с собой.
    noise_floor = None
    if bool(settings.get("measure_noise_floor", True)):
        noise_floor = compare_runners(
            reference, reference, loader, device=device, limit_batches=batches
        )
        log.info(
            "Собственный шум модели (fp32 против себя же): max_abs=%.3g | совпадение=%.4f",
            noise_floor["max_abs"], noise_floor["argmax_agreement"],
        )

    comparison = compare_runners(
        reference, candidate, loader, device=device, limit_batches=batches,
    )

    if noise_floor is not None and comparison["max_abs"] <= noise_floor["max_abs"]:
        log.warning(
            "Расхождение %s (%.3g) не превышает собственный шум модели (%.3g). "
            "Вердикт приёмки ниже относится к сумме двух эффектов и об эффекте "
            "точности сам по себе не говорит ничего — сначала убирайте случайность "
            "из модели.",
            candidate.name, comparison["max_abs"], noise_floor["max_abs"],
        )

    limit = settings.limit_eval_batches
    limit = int(limit) if limit is not None else None
    common = {
        "task_type": task_type,
        "num_classes": dataset.get("num_classes"),
        "ignore_index": dataset.get("ignore_index", 255),
        "limit_batches": limit,
    }
    baseline_metrics = evaluate_runner(reference, loader, device, **common)
    candidate_metrics = evaluate_runner(candidate, loader, device, **common)

    verdict = check_acceptance(
        baseline_metrics, candidate_metrics, comparison,
        task_type=task_type,
        max_metric_drop=float(settings.max_metric_drop),
        min_agreement=float(settings.min_agreement),
        noise_floor=noise_floor,
    )

    payload = {
        "comparison": comparison,
        "noise_floor": noise_floor,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "verdict": verdict,
    }
    report.stage("validate", payload, status="ok" if verdict["passed"] else "failed")

    rows = [comparison, baseline_metrics, candidate_metrics]
    if noise_floor is not None:
        rows.insert(0, {**noise_floor, "candidate": f"{reference.name} (шум модели)"})
    report.table("numerics.csv", rows)
    return payload


def resolve_sample_shape(cfg: DictConfig, onnx_path: Path, get_loader) -> list[int]:
    """Форма одного примера (C, H, W) для замера.

    Порядок источников — от точного к запасному:
    1. сам `.onnx` — там записано ровно то, подо что собран движок;
    2. `data.dataset.image_size` — если экспорт в этом запуске не делался;
    3. лоадер — последний вариант, потому что он тянет за собой датасет,
       а замеру данные не нужны вовсе: он считает на случайных тензорах.
    """
    if onnx_path.is_file():
        from src.quantization.build_engine import onnx_inputs

        (_, dims), = onnx_inputs(onnx_path)
        spatial = dims[1:]
        if all(dim is not None for dim in spatial):
            return [int(dim) for dim in spatial if dim is not None]

    size = cfg.data.dataset.get("image_size")
    if size is not None:
        height, width = (size, size) if isinstance(size, int) else tuple(size)
        return [3, int(height), int(width)]

    log.info("Форма входа неизвестна из конфига — беру её из лоадера.")
    return list(next(iter(get_loader()))[0].shape[1:])


def stage_benchmark(cfg, reference, candidate, sample_shape, device, report) -> list[dict]:
    """Замер обоих раннеров по сетке батчей + ускорение относительно базы."""
    settings = cfg.quantize.benchmark
    common = {
        "sample_shape": sample_shape,
        "batch_sizes": list(settings.batch_sizes),
        "device": device,
        "warmup": int(settings.warmup),
        "iters": int(settings.iters),
        "modes": list(settings.modes),
    }

    rows = benchmark_runner(reference, **common) + benchmark_runner(candidate, **common)
    log_speedup_summary(rows, baseline=str(settings.baseline))
    # После сводки, а не до: предупреждение должно оказаться последним на
    # экране, иначе его унесёт таблицей, а оно как раз про то, можно ли этой
    # таблице верить.
    anomalies = check_measurement_sanity(rows)
    rows = speedup_table(rows, baseline=str(settings.baseline))

    report.table("benchmark.csv", rows)
    report.stage(
        "benchmark",
        {"rows": len(rows), "anomalies": anomalies},
        status="ok" if not anomalies else "unreliable",
    )
    return rows


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    if cfg.get("quantize") is None:
        raise ValueError(
            "Не выбран тип квантизации в config. Например quantize=trt_fp16"
        )

    unknown = set(cfg.quantize.stages) - set(STAGES)
    if unknown:
        raise ValueError(f"Неизвестные стадии {sorted(unknown)}; доступны {STAGES}.")

    check_single_process()
    if {"validate", "benchmark"} & set(cfg.quantize.stages):
        check_batch_fits_profile(cfg)
    seed_everything(cfg.seed, deterministic=cfg.deterministic, warn_only=cfg.deterministic_warn_only)
    device = resolve_device(cfg.device)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    report = QuantizationReport(output_dir)

    artifacts = Path(to_absolute_path(cfg.quantize.artifacts_dir)) / str(cfg.quantize.model_id)
    artifacts.mkdir(parents=True, exist_ok=True)
    onnx_path = artifacts / "model_fp32.onnx"

    log.info("Конфиг запуска:\n%s", OmegaConf.to_yaml(cfg))
    log.info("Отчёты: %s | артефакты: %s", output_dir, artifacts)

    student = build_student(cfg).to(device)

    loaders: dict[str, DataLoader] = {}

    def get_loader() -> DataLoader:
        if "eval" not in loaders:
            _, loaders["eval"] = base_loader(
                cfg.data, cfg.get("task_type", "classification"), seed=cfg.seed
            )
        return loaders["eval"]

    manifest = write_source_manifest(
        artifacts / "source.json", cfg, Path(to_absolute_path(cfg.ckpt_path))
    )
    report.context(
        name=cfg.name,
        backend=cfg.quantize.backend,
        precision=cfg.quantize.precision,
        device=str(device),
        # В отчёт: без этого два прогона с разной раскладкой различаются только
        # временами, и через неделю не вспомнить, какой из них какой.
        channels_last=bool(cfg.quantize.get("channels_last")),
        artifacts_dir=str(artifacts),
        source=manifest,
    )

    stages = list(cfg.quantize.stages)
    if "export" in stages:
        stage_export(cfg, student, get_loader, device, onnx_path, manifest, report)
    if "calibrate" in stages:
        stage_calibrate(cfg, artifacts, report)
    if "build" in stages:
        stage_build(cfg, onnx_path, artifacts, report)

    if not ({"validate", "benchmark"} & set(stages)):
        log.info("Отчёт: %s", report.path)
        return 0.0

    # Раскладка эталона обязана совпадать с кандидатом, иначе сравнение
    # смешает эффект точности с эффектом раскладки.
    channels_last = bool(cfg.quantize.get("channels_last"))
    if channels_last and cfg.quantize.backend != "torch":
        log.warning(
            "channels_last=true при backend=%s: раскладка применится только к "
            "torch-эталону, у движка она своя. Ускорение к такой базе будет "
            "означать другое — сверяйтесь с этим при чтении таблицы.",
            cfg.quantize.backend,
        )

    reference = make_runner("torch", model=student, precision="fp32", device=device,
                            channels_last=channels_last)
    candidate = make_candidate_runner(cfg, student, onnx_path, artifacts, device)

    verdict = None
    try:
        if "validate" in stages:
            payload = stage_validate(cfg, reference, candidate, get_loader(), device, report)
            verdict = payload["verdict"]
        if "benchmark" in stages:
            sample_shape = resolve_sample_shape(cfg, onnx_path, get_loader)
            stage_benchmark(cfg, reference, candidate, sample_shape, device, report)
    finally:
        candidate.close()
        reference.close()

    log.info("Отчёт: %s", report.path)

    if verdict is not None and not verdict["passed"] and cfg.quantize.validate.strict:
        raise RuntimeError(
            "Квантизованная модель не прошла валидацию stage_validate: "
            + "; ".join(verdict["violations"])
            + f". Подробности в {report.path}."
        )

    if verdict is None:
        return 0.0
    return float(verdict["candidate"])


if __name__ == "__main__":
    main()
