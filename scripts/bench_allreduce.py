"""Сколько стоит синхронизация градиентов сама по себе.

    torchrun --nproc_per_node=2 scripts/bench_allreduce.py
    CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 scripts/bench_allreduce.py

Зачем. У нас две DDP-обёртки: на ученике и на лоссе. У MGD параметров в
лоссе больше, чем в самом ученике (13.6M против 11.7M у ResNet-18), и весь
их градиент уходит в сеть на каждом шаге. Вопрос — успевает ли этот обмен
спрятаться за вычислением backward или становится узким местом.

Скрипт меряет чистое время all_reduce для размеров, соответствующих нашим
моделям, и переводит его в пропускную способность шины. Дальше это число
сравнивается с временем шага обучения из bench_ddp.py: если обмен занимает
проценты от шага — перекрытие работает, если десятки — упирается в сеть.

Формула busbw для ring-allreduce: каждый ранк отправляет и принимает
2*(N-1)/N от размера тензора, поэтому эффективный объём трафика больше
самого тензора почти вдвое.
"""

import os
import time

import torch
import torch.distributed as dist

# Размеры, привязанные к нашим моделям (число параметров в fp32).
PAYLOADS = [
    ("адаптеры FeatureKD (1.4M)", 1_392_640),
    ("ResNet-18 целиком (11.7M)", 11_700_000),
    ("адаптер MGD (13.6M)", 13_635_584),
    ("ResNet-50 целиком (23.5M)", 23_500_000),
]

WARMUP_ITERS = 10
MEASURE_ITERS = 50


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend)
    is_main = rank == 0

    if is_main:
        name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        print(f"world_size={world_size} | backend={backend} | {name}\n")
        print(f"{'что синхронизируем':<28} {'МиБ':>8} {'мс':>9} {'ГБ/с':>9}")
        print("-" * 58)

    for label, numel in PAYLOADS:
        tensor = torch.ones(numel, dtype=torch.float32, device=device)
        size_bytes = tensor.numel() * tensor.element_size()

        # Прогрев: первый обмен включает установку соединений NCCL и
        # аллокацию внутренних буферов — он в разы дороже последующих.
        for _ in range(WARMUP_ITERS):
            dist.all_reduce(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Барьер, чтобы все ранки начали замер одновременно: иначе в время
        # попадёт ожидание отставшего.
        dist.barrier()
        start = time.perf_counter()
        for _ in range(MEASURE_ITERS):
            dist.all_reduce(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / MEASURE_ITERS

        # Алгоритмическая полоса ring-allreduce.
        bus_bytes = 2 * (world_size - 1) / world_size * size_bytes
        if is_main:
            print(f"{label:<28} {size_bytes / 2**20:>8.1f} {elapsed * 1000:>9.2f} "
                  f"{bus_bytes / elapsed / 1e9:>9.1f}")

        del tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if is_main:
        print("\nСравни колонку 'мс' со временем одного шага обучения "
              "(bench_ddp.py: с/эпоху делить на limit_train_batches).")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
