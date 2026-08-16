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
from src.quantization.benchmark import benchmark_runner, speedup_table
from src.quantization.export import export_onnx, sha256_file
from src.quantization.numerics import compare_runners, evaluate_runner
from src.quantization.report import QuantizationReport, check_acceptance
from src.utils import resolve_device, seed_everything
from src.utils.checkpoints import load_checkpoint_into
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

    result = export_onnx(
        student.to("cpu"),
        sample=take_sample(cfg, get_loader, torch.device("cpu")),
        output_path=onnx_path,
        opset=int(settings.opset),
        dynamo=bool(settings.dynamo),
        dynamic_axes=get_dynamic_axes(cfg),
        external_data=settings.external_data,
        optimize=bool(settings.optimize),
        parity_atol=float(settings.parity_atol),
    )
    student.to(device)
    report.stage("export", result.meta)
    return result.meta


def stage_calibrate(cfg, report) -> dict:
    """Заглушка под int8"""
    reason = (
        f"precision={cfg.quantize.precision}: ни масштабов, ни zero-point здесь нет, "
        f"статистики активаций собирать не для чего."
    )
    log.info("calibrate: пропуск - %s", reason)
    report.stage("calibrate", {"reason": reason}, status="skipped")
    return {"skipped": True}


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

    from src.quantization.build_engine import build_engine

    engine_path = engine_path_for(cfg, artifacts)

    if cfg.quantize.reuse and engine_path.is_file():
        meta_path = engine_path.with_suffix(".meta.json")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("source_onnx_sha256") == sha256_file(onnx_path):
                log.info("build: %s собран из этого же .onnx - пропускаем", engine_path.name)
                report.stage("build", {**meta, "reused": True})
                return meta
        log.info("build: движок есть, но собран из другого .onnx - пересобираем")

    settings = cfg.quantize.build
    opt_batch = settings.opt_batch
    opt_shape = None
    if opt_batch is not None:
        sample_shape = report.payload["stages"]["export"]["input_shape"]
        opt_shape = [int(opt_batch), *sample_shape[1:]]

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
    )
    report.stage("build", result.meta)
    return result.meta


def make_candidate_runner(cfg, student, onnx_path, artifacts, device):
    """Раннер кандидата под выбранный бэкенд. Копируем student"""
    backend = cfg.quantize.backend
    precision = cfg.quantize.precision

    if backend == "torch":
        return make_runner("torch", model=copy.deepcopy(student), precision=precision, device=device)

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
    rows = speedup_table(rows, baseline=str(settings.baseline))

    report.table("benchmark.csv", rows)
    report.stage("benchmark", {"rows": len(rows)})
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
        artifacts_dir=str(artifacts),
        source=manifest,
    )

    stages = list(cfg.quantize.stages)
    if "export" in stages:
        stage_export(cfg, student, get_loader, device, onnx_path, manifest, report)
    if "calibrate" in stages:
        stage_calibrate(cfg, report)
    if "build" in stages:
        stage_build(cfg, onnx_path, artifacts, report)

    if not ({"validate", "benchmark"} & set(stages)):
        log.info("Отчёт: %s", report.path)
        return 0.0

    reference = make_runner("torch", model=student, precision="fp32", device=device)
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
