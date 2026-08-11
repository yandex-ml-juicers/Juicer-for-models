"""Очередь экспериментов с поддержкой многокарточных (DDP) задач.

    python scripts/exp_queue.py
    python scripts/exp_queue.py queue_jobs/my_jobs.yaml
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
        self._all = list(gpu_ids)
        self._total = len(gpu_ids)
        self._condition = threading.Condition()

    def acquire(self, count: int, devices: list[int] | None = None) -> list[int]:
        """Забирает карты: любые `count` штук либо строго перечисленные `devices`."""
        if devices is not None:
            unknown = [gpu for gpu in devices if gpu not in self._all]
            if unknown:
                raise ValueError(
                    f"Задача закреплена за картами {devices}, но {unknown} нет "
                    f"в available_gpus={self._all}. Такая задача не запустится никогда."
                )
            with self._condition:
                self._condition.wait_for(lambda: all(gpu in self._free for gpu in devices))
                for gpu in devices:
                    self._free.remove(gpu)
                return list(devices)

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


def parse_job_args(job, default_gpus: int) -> tuple[str, int, list[int] | None]:
    if isinstance(job, str):
        return job, default_gpus, None

    devices = job.get("devices")
    if devices is None:
        return job["args"], int(job.get("gpus", default_gpus)), None

    devices = [int(gpu) for gpu in devices]
    declared = job.get("gpus")
    if declared is not None and int(declared) != len(devices):
        raise ValueError(
            f"У задачи gpus={declared}, но devices={devices} — это {len(devices)} карт. "
            f"Убери gpus или приведи в соответствие."
        )
    return job["args"], len(devices), devices


def build_command(args: str, num_gpus: int, port: int) -> list[str]:
    """Одна карта — обычный python, несколько — torchrun."""
    if num_gpus == 1:
        return [sys.executable, str(TRAIN_SCRIPT), *args.split()]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--nnodes=1",
        f"--master_port={port}",
        str(TRAIN_SCRIPT),
        *args.split(),
    ]


def run_job(job, index: int, pool: GpuPool, default_gpus: int,
            batch_stamp: str) -> None:
    args, num_gpus, pinned = parse_job_args(job, default_gpus)
    gpu_ids = pool.acquire(num_gpus, devices=pinned)
    try:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
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

    print(f"Очередь: {len(experiments)} задач на картах {gpu_ids} "
          f"(по умолчанию {default_gpus} на задачу\n")

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [
            executor.submit(run_job, job, index, pool, default_gpus, batch_stamp)
            for index, job in enumerate(experiments)
        ]
        for future in futures:
            future.result()

    print("\nВсе эксперименты завершены.")


if __name__ == "__main__":
    main()
