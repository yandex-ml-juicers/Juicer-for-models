"""Замер производительности раннера: задержка и пропускная способность"""

import logging
import time
from collections.abc import Sequence

import torch
from torch import Tensor

from src.quantization.backends.base import Runner

log = logging.getLogger(__name__)

DEFAULT_WARMUP = 50
DEFAULT_ITERS = 200


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Перцентиль по ближайшему рангу"""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))
    return ordered[index]


def _summarize(durations_ms: Sequence[float], batch_size: int) -> dict:
    mean = sum(durations_ms) / len(durations_ms)
    p50 = _percentile(durations_ms, 0.50)
    return {
        "iters": len(durations_ms),
        "mean_ms": mean,
        "p50_ms": p50,
        "p90_ms": _percentile(durations_ms, 0.90),
        "p99_ms": _percentile(durations_ms, 0.99),
        "min_ms": min(durations_ms),
        # Пропускная способность считается по медиане, а не по среднему:
        "throughput_sps": batch_size * 1000.0 / p50,
    }


def _time_cuda(runner: Runner, batch: Tensor, iters: int, device: torch.device) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for index in range(iters):
        starts[index].record()
        runner.infer(batch)
        ends[index].record()

    torch.cuda.synchronize(device)
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def _time_host(runner: Runner, batch: Tensor, iters: int) -> list[float]:
    """Замер на CPU-бэкендах: там всё синхронно и события неприменимы"""
    durations = []
    for _ in range(iters):
        started = time.perf_counter()
        runner.infer(batch)
        durations.append((time.perf_counter() - started) * 1000.0)
    return durations


def benchmark_runner(
    runner: Runner,
    *,
    sample_shape: Sequence[int],
    batch_sizes: Sequence[int] = (1, 8, 32),
    device: torch.device | str = "cuda",
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
    modes: Sequence[str] = ("compute", "e2e"),
) -> list[dict]:
    """Прогоняет раннер по сетке батчей и возвращает строки для benchmark.csv.

    sample_shape - один пример без батча, например (3, 224, 224).
    """
    device = torch.device(device)
    rows: list[dict] = []

    for mode in modes:
        if mode not in ("compute", "e2e"):
            raise ValueError(f"Режим {mode!r} неизвестен; доступны 'compute' и 'e2e'.")

        for batch_size in batch_sizes:
            shape = (batch_size, *sample_shape)
            if mode == "compute":
                # Вход заранее в памяти карты: меряем только вычисление.
                batch = torch.randn(shape, device=device)
            else:
                # pin_memory — то, как это делает DataLoader; без него
                # копия с хоста медленнее
                batch = torch.randn(shape).pin_memory() if device.type == "cuda" else torch.randn(shape)

            try:
                for _ in range(warmup):
                    runner.infer(batch)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    durations = _time_cuda(runner, batch, iters, device)
                else:
                    durations = _time_host(runner, batch, iters)
            except (RuntimeError, ValueError) as error:
                log.warning("Замер %s batch=%d пропущен: %s", runner.name, batch_size, error)
                continue

            row = {
                "runner": runner.name,
                "mode": mode,
                "batch_size": batch_size,
                **_summarize(durations, batch_size),
            }
            if device.type == "cuda":
                row["peak_torch_mem_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2

            rows.append(row)
            log.info(
                "%-12s %-7s batch=%-4d p50=%7.3f мс | p99=%7.3f мс | %8.1f сэмплов/с",
                runner.name,
                mode,
                batch_size,
                row["p50_ms"],
                row["p99_ms"],
                row["throughput_sps"],
            )

    return rows


def speedup_table(rows: Sequence[dict], baseline: str) -> list[dict]:
    """Добавляет к строкам ускорение относительно базового раннера"""
    reference = {
        (row["mode"], row["batch_size"]): row["p50_ms"]
        for row in rows
        if row["runner"] == baseline
    }
    if not reference:
        log.warning("Базовый раннер %r в замерах не найден — ускорения не будет.", baseline)

    enriched = []
    for row in rows:
        base = reference.get((row["mode"], row["batch_size"]))
        enriched.append({**row, "speedup_vs_" + baseline: base / row["p50_ms"] if base else None})
    return enriched
