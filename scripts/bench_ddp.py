"""Бенчмарк масштабирования: один и тот же эксперимент на 1, 2, 4 картах.

    python scripts/bench_ddp.py --experiment=baseline/b2_feature_kd --gpus 1 2 4

Что меряет и почему именно так:

- Время берётся из history.csv (колонка time_epoch), а не измеряется снаружи.
  Так из замера выпадает всё, что к обучению не относится: импорт torch,
  загрузка весов учителя, инициализация CUDA-контекста.
- ПЕРВАЯ эпоха всегда отбрасывается. В неё попадают прогрев cuDNN, старт
  воркеров DataLoader и первый обмен через NCCL — на короткой эпохе это
  запросто половина времени.
- Батч глобальный, поэтому число шагов оптимизации на всех конфигурациях
  одинаково. Ускорение считается честно: та же работа за меньшее время.

Проверяется одновременно и корректность: eval_acc на разном числе карт
обязана совпадать с точностью до шума. Систематическое расхождение означает
проблему в шардировании или синхронизации, а не в скорости.
"""

import argparse
import csv
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"

sys.path.insert(0, str(REPO_ROOT))
from src.utils.distributed import find_free_port  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="например baseline/b2_feature_kd")
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 4],
                        help="сколько карт использовать в каждом прогоне")
    parser.add_argument("--devices", type=str, default=None,
                        help="физические карты через запятую, например 1,2,3. "
                             "По умолчанию берутся первые N видимых")
    parser.add_argument("--epochs", type=int, default=4,
                        help="эпох на прогон; первая отбрасывается, так что нужно >= 2")
    parser.add_argument("--limit-train-batches", type=int, default=50)
    parser.add_argument("--limit-eval-batches", type=int, default=20)
    parser.add_argument("--threads", type=int, default=None,
                        help="OMP_NUM_THREADS на процесс; по умолчанию ядра/карты")
    parser.add_argument("--per-gpu-batch", type=int, default=None,
                        help="фиксировать батч НА КАРТУ (weak scaling): глобальный "
                             "батч станет per_gpu_batch * число карт. По умолчанию "
                             "глобальный батч фиксирован (strong scaling). "
                             "Не сочетать с data.loader.batch_size в --extra")
    parser.add_argument("--extra", type=str, default="",
                        help="дополнительные overrides Hydra одной строкой")
    parser.add_argument("--clearml", action="store_true",
                        help="включить ClearML: кривые всех конфигураций лягут "
                             "в UI и их можно будет сравнить наложением")
    parser.add_argument("--run-prefix", type=str, default="ddp_bench",
                        help="префикс имени запуска; попадает в outputs/ и в ClearML")
    parser.add_argument("--concurrent", type=int, default=None,
                        help="вместо масштабирования запустить N ОДНОКАРТОЧНЫХ задач "
                             "одновременно, по одной на карту. Меряет пропускную "
                             "способность машины под нагрузкой — то есть режим, в "
                             "котором сервер обычно и живёт")
    return parser.parse_args()


def build_command(args_line: str, num_gpus: int, port: int) -> list[str]:
    if num_gpus == 1:
        return [sys.executable, str(TRAIN_SCRIPT), *args_line.split()]
    return [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}", "--nnodes=1", f"--master_port={port}",
        str(TRAIN_SCRIPT), *args_line.split(),
    ]


def read_history(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def run_one(options: argparse.Namespace, num_gpus: int, stamp: str,
            devices: list[str] | None = None, label: str | None = None) -> dict | None:
    run_name = label or f"{options.run_prefix}_ws{num_gpus}"
    if devices is not None:
        gpu_ids: list[str] = list(devices)
    elif options.devices:
        gpu_ids = str(options.devices).split(",")[:num_gpus]
    else:
        gpu_ids = [str(i) for i in range(num_gpus)]

    parts = [
        f"experiment={options.experiment}",
        # Чекпоинты выключены всегда: при clearml.output_uri=true каждый из них
        # уезжает на сервер сотней мегабайт, что на общей машине невежливо и
        # растягивает прогон. На сам замер это не влияет — time_epoch
        # считается ДО сохранения, — но время ожидания результата растёт.
        f"clearml.enabled={'true' if options.clearml else 'false'}",
        f"name={run_name}",
        f"trainer.epochs={options.epochs}",
        f"trainer.limit_train_batches={options.limit_train_batches}",
        f"trainer.limit_eval_batches={options.limit_eval_batches}",
        "trainer.progress_bar=false",
        "trainer.save_best=false trainer.save_last=false",
        options.extra,
    ]
    if options.per_gpu_batch is not None:
        # Weak scaling: с ростом числа карт растёт и объём работы, зато
        # нагрузка на каждую карту постоянна. Так меряется потолок
        # пропускной способности железа, а не ускорение конкретного
        # эксперимента.
        parts.append(f"data.loader.batch_size={options.per_gpu_batch * num_gpus}")
    args_line = " ".join(parts).strip()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    env["JUICER_RUN_ID"] = stamp
    env["MASTER_ADDR"] = "127.0.0.1"
    port = find_free_port()
    env["MASTER_PORT"] = str(port)
    # Явно, чтобы torchrun не поставил 1 поток на процесс: тогда CPU-часть
    # (аугментации, сборка батчей) схлопнется, и замер будет мерить не DDP.
    threads = options.threads or max(1, (os.cpu_count() or num_gpus) // num_gpus)
    env["OMP_NUM_THREADS"] = str(threads)

    print(f"\n=== {run_name}: {num_gpus} GPU (карты {gpu_ids}, "
          f"{threads} потоков CPU на процесс) ===", flush=True)
    result = subprocess.run(build_command(args_line, num_gpus, port), env=env, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"[FAILED] прогон на {num_gpus} GPU упал с кодом {result.returncode}")
        return None

    history_path = REPO_ROOT / "outputs" / run_name / stamp / "history.csv"
    if not history_path.exists():
        print(f"[FAILED] нет {history_path}")
        return None

    rows = read_history(history_path)
    if len(rows) < 2:
        print("[FAILED] нужно минимум 2 эпохи: первая отбрасывается как прогревочная")
        return None

    # Первая эпоха выброшена: прогрев cuDNN, старт воркеров, первый обмен NCCL.
    epoch_times = [float(row["time_epoch"]) for row in rows[1:]]
    global_batch = int(float(rows[-1]["global_batch_size"]))

    return {
        "world_size": num_gpus,
        "time_epoch": statistics.median(epoch_times),
        "time_spread": max(epoch_times) - min(epoch_times),
        "eval_acc": float(rows[-1]["eval_acc"]),
        "global_batch": global_batch,
        # Пропускная способность приблизительная: time_epoch включает и eval,
        # поэтому limit_eval_batches стоит держать маленьким.
        "images_per_sec": options.limit_train_batches * global_batch
                          / statistics.median(epoch_times),
    }


def run_concurrent(options: argparse.Namespace, stamp: str) -> list[dict]:
    """N однокарточных задач одновременно, по одной на карту.

    Меряет не ускорение одного эксперимента, а пропускную способность машины
    целиком — то есть режим, в котором сервер обычно и работает: все карты
    заняты разными задачами. Задачи делят между собой CPU, диск и шину, и
    именно эта деградация здесь и интересна.
    """
    count = options.concurrent
    devices = (str(options.devices).split(",") if options.devices
               else [str(i) for i in range(count)])[:count]
    if len(devices) < count:
        raise SystemExit(f"нужно {count} карт, а в --devices указано {len(devices)}")

    # Потоки делим на всех: задач ровно count, и суммарно они не должны
    # запросить больше ядер, чем есть.
    threads = options.threads or max(1, (os.cpu_count() or count) // count)
    per_job = argparse.Namespace(**{**vars(options), "threads": threads})

    with ThreadPoolExecutor(max_workers=count) as executor:
        futures = [
            executor.submit(run_one, per_job, 1, stamp, [device],
                            f"{options.run_prefix}_conc{index}")
            for index, device in enumerate(devices)
        ]
        return [row for row in (future.result() for future in futures) if row]


def report_concurrent(options: argparse.Namespace, results: list[dict]) -> None:
    print(f"\n{'=' * 78}")
    print(f"Эксперимент: {options.experiment} | {len(results)} однокарточных задач "
          f"одновременно | батч: {results[0]['global_batch']}")
    print(f"{'задача':>8} {'с/эпоху':>10} {'разброс':>9} {'img/s':>10} {'eval_acc':>9}")
    print("-" * 78)
    for index, row in enumerate(results):
        print(f"{index:>8} {row['time_epoch']:>10.2f} {row['time_spread']:>9.2f} "
              f"{row['images_per_sec']:>10.0f} {row['eval_acc']:>9.4f}")
    print("-" * 78)
    total = sum(row["images_per_sec"] for row in results)
    slowest = max(row["time_epoch"] for row in results)
    print(f"{'ИТОГО':>8} {slowest:>10.2f} {'':>9} {total:>10.0f}")
    print("=" * 78)
    print("\nСравни ИТОГО с img/s одиночного запуска той же задачи: разница "
          "показывает, сколько машина теряет на конкуренции за CPU, диск и шину.")


def main() -> None:
    options = parse_args()
    if options.epochs < 2:
        raise SystemExit("нужно минимум 2 эпохи: первая отбрасывается как прогревочная")

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if options.concurrent:
        results = run_concurrent(options, stamp)
        if not results:
            raise SystemExit("ни одна задача не завершилась успешно")
        report_concurrent(options, results)
        return

    results = [row for row in (run_one(options, n, stamp) for n in options.gpus) if row]
    if not results:
        raise SystemExit("ни один прогон не завершился успешно")

    baseline = results[0]["time_epoch"]
    baseline_ws = results[0]["world_size"]

    print(f"\n{'=' * 78}")
    print(f"Эксперимент: {options.experiment} | глобальный батч: {results[0]['global_batch']}")
    print(f"{'GPU':>4} {'с/эпоху':>10} {'разброс':>9} {'ускорение':>10} "
          f"{'эффект.':>9} {'img/s':>10} {'eval_acc':>9}")
    print("-" * 78)
    for row in results:
        speedup = baseline / row["time_epoch"]
        efficiency = speedup / (row["world_size"] / baseline_ws)
        print(f"{row['world_size']:>4} {row['time_epoch']:>10.2f} {row['time_spread']:>9.2f} "
              f"{speedup:>9.2f}x {efficiency:>8.0%} {row['images_per_sec']:>10.0f} "
              f"{row['eval_acc']:>9.4f}")
    print("=" * 78)

    accuracies = [row["eval_acc"] for row in results]
    spread = max(accuracies) - min(accuracies)
    print(f"\nРазброс eval_acc между конфигурациями: {spread:.4f}")
    if spread > 0.02:
        print("ВНИМАНИЕ: расхождение больше 2 п.п. При глобальном батче и одинаковом "
              "числе шагов такого быть не должно — проверь шардирование и синхронизацию.")


if __name__ == "__main__":
    main()
