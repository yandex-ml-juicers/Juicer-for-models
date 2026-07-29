import subprocess
import queue
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from omegaconf import OmegaConf

DEFAULT_QUEUE_CONFIG_PATH = Path(__file__).parent / "queue_jobs.yaml"

# Без аргумента - queue_jobs.yaml рядом со скриптом (быстрый разовый прогон).
# С аргументом - можно указать сохранённый именной батч, например:
#   python scripts/exp_queue.py queue_jobs/imagenet_sweep.yaml
queue_config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_QUEUE_CONFIG_PATH
queue_cfg = OmegaConf.load(queue_config_path)

AVAILABLE_GPUS = list(queue_cfg.available_gpus)
EXPERIMENTS = list(queue_cfg.experiments)

# Создаем потокобезопасную очередь и кладем туда свободные GPU
gpu_queue = queue.Queue()
for gpu_id in AVAILABLE_GPUS:
    gpu_queue.put(gpu_id)

def run_task(exp_args):
    # Забираем свободную видеокарту (если все заняты - поток уснет и будет ждать)
    gpu_id = gpu_queue.get()
    
    print(f"[START] GPU: {gpu_id} | Task: {exp_args}")
    
    # Изолируем процесс только на выданной видеокарте
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Запускаем train.py
    cmd = f"python scripts/train.py {exp_args}"
    try:
        # subprocess заблокирует текущий поток до окончания обучения
        subprocess.run(cmd, shell=True, env=env, check=True)
        print(f"[SUCCESS] GPU: {gpu_id} | Task: {exp_args}")
    except subprocess.CalledProcessError:
        print(f"[ERROR] GPU: {gpu_id} | Task: {exp_args} failed!")
    finally:
        # возвращаем GPU обратно в очередь для следующих задач
        gpu_queue.put(gpu_id)


if __name__ == "__main__":
    print(f"Запуск очереди из {len(EXPERIMENTS)} задач на {len(AVAILABLE_GPUS)} GPU...")
    
    # Создаем пул потоков размером с количество видеокарт
    with ThreadPoolExecutor(max_workers=len(AVAILABLE_GPUS)) as executor:
        list(executor.map(run_task, EXPERIMENTS))

    print("Все эксперименты завершены!")
