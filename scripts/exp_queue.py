"""Очередь экспериментов с поддержкой многокарточных (DDP) задач.

    python scripts/exp_queue.py                          # queue_jobs.yaml рядом со скриптом
    python scripts/exp_queue.py queue_jobs/my_jobs.yaml    # именной сохранённый батч
"""

import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.utils.distributed import find_free_port

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"
DEFAULT_QUEUE_CONFIG = Path(__file__).parent / "queue_jobs.yaml"


class GpuPool:
    """Пул карт с атомарной выдачей группами."""

    def __init__(self, gpu_ids: list[int]) -> None:
        self._free = list(gpu_ids)
        self._total = len(gpu_ids)
        self._condition = threading.Condition()

    def acquire(self, count: int) -> list[int]:
        if count > self._total:
            raise ValueError(
                f"Задача просит {count} GPU, а всего доступно {self._total}. "
                f"Такая задача не запустится никогда — правь available_gpus."
            )
        with self._condition:
            self._condition.wait_for(lambda: len(self._free) >= count)
            return [self._free.pop(0) for _ in range(count)]

    def release(self, gpu_ids: list[int]) -> None:
        with self._condition:
            self._free.extend(gpu_ids)
            self._condition.notify_all()


def normalize(job, default_gpus: int) -> tuple[str, int]:
    if isinstance(job, str):
        return job, default_gpus
    return job["args"], int(job.get("gpus", default_gpus))


def build_command(args: str, num_gpus: int, port: int) -> list[str]:
    """Одна карта — обычный python, несколько — torchrun."""
    if num_gpus == 1:
        return [sys.executable, str(TRAIN_SCRIPT), *args.split()]
    return [
        # sys.executable -m torch.distributed.run, а не голое "torchrun":
        # это та же точка входа, но гарантированно из текущего окружения.
        # На общей машине в PATH легко оказывается torchrun из чужого venv.
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--nnodes=1",
        f"--master_port={port}",
        str(TRAIN_SCRIPT),
        *args.split(),
    ]


def resolve_threads_per_process(num_slots: int, configured: int | None) -> int:
    if configured is not None:
        return max(1, int(configured))
    return max(1, (os.cpu_count() or num_slots) // max(1, num_slots))


def run_job(job, index: int, pool: GpuPool, default_gpus: int,
            batch_stamp: str, threads_per_process: int) -> None:
    args, num_gpus = normalize(job, default_gpus)
    gpu_ids = pool.acquire(num_gpus)
    try:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads_per_process)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)

        port = find_free_port()
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = str(port)

        env["JUICER_RUN_ID"] = f"{batch_stamp}_j{index:02d}"

        command = build_command(args, num_gpus, port)
        print(f"[START]   GPU {gpu_ids} | {args}", flush=True)
        try:
            subprocess.run(command, env=env, cwd=REPO_ROOT, check=True)
            print(f"[SUCCESS] GPU {gpu_ids} | {args}", flush=True)
        except subprocess.CalledProcessError as error:
            print(f"[FAILED]  GPU {gpu_ids} | {args} | код {error.returncode}", flush=True)
    finally:
        pool.release(gpu_ids)


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_QUEUE_CONFIG
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(
            f"{config_path}: ожидался yaml-словарь с ключами "
            f"available_gpus/experiments, получен {type(cfg).__name__}"
        )

    gpu_ids = list(cfg.available_gpus)
    default_gpus = int(cfg.get("default_gpus_per_job", 1))
    experiments = list(cfg.experiments)

    pool = GpuPool(gpu_ids)
    batch_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    threads_per_process = resolve_threads_per_process(
        len(gpu_ids), cfg.get("threads_per_process", None)
    )

    print(f"Очередь: {len(experiments)} задач на картах {gpu_ids} "
          f"(по умолчанию {default_gpus} на задачу, "
          f"{threads_per_process} потоков CPU на процесс)\n")

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [
            executor.submit(run_job, job, index, pool, default_gpus,
                            batch_stamp, threads_per_process)
            for index, job in enumerate(experiments)
        ]
        for future in futures:
            future.result()

    print("\nВсе эксперименты завершены.")


if __name__ == "__main__":
    main()
