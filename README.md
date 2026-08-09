# Juicer-for-models


## Установка

Требования: Python >= 3.10; для обучения на GPU — PyTorch со сборкой CUDA,
поддерживающей архитектуру карт

```bash
pip install -r requirements.txt
pip install -e . --no-deps        # пакет src/ становится импортируемым
# для тестов: pip install -e ".[dev]" --no-build-isolation

# Новые cu128 (A100, RTX 30xx/40xx, H100)
pip install --force-reinstall 'torch>=2.7' 'torchvision>=0.22' --index-url https://download.pytorch.org/whl/cu128
# Старые cu126 (Tesla V100, compute capability 7.0)
pip install --force-reinstall 'torch>=2.7' 'torchvision>=0.22' --index-url https://download.pytorch.org/whl/cu126
```

Проверки

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list())"
python -c "import torch; x = torch.randn(1000, 1000).cuda(); print((x @ x).sum().item())"
```

Для V100 в списке обязан быть `sm_70`. 


## Быстрый старт

```bash
# Бейзлайн 1: ванильная дистилляция Хинтона, ResNet-56 -> ResNet-20
python scripts/train.py experiment=baseline/b1_vanilla_kd

# Контроль к нему: тот же ученик с нуля, без учителя
python scripts/train.py experiment=b1_scratch

# Бейзлайн 2: feature-based KD, ResNet-50 -> ResNet-18 (логиты + карты признаков)
python scripts/train.py experiment=b2_feature_kd
python scripts/train.py experiment=b2_scratch

# Любой гиперпараметр меняется из CLI, без правки кода:
python scripts/train.py experiment=b1_vanilla_kd loss.temperature=8 trainer.epochs=30 seed=1

# Смоук-тест пайплайна: синтетические данные, CPU, без сети, ~10 секунд
python scripts/train.py data/dataset=fake_cifar10 '~model/teacher' loss=ce \
    trainer.epochs=1 trainer.limit_train_batches=3 trainer.limit_eval_batches=2

# Оценка сохранённого чекпоинта
python scripts/eval.py experiment=b2_feature_kd ckpt_path=outputs/<name>/<run>/best.pt
```

Артефакты каждого запуска — в `outputs/<name>/<дата_время>/`:
`.hydra/config.yaml` (полный снапшот конфига), `train.log`, `history.csv`
(метрики и все компоненты лосса по эпохам), `best.pt` / `last.pt`.

## Очередь экспериментов

`scripts/exp_queue.py` запускает пачку экспериментов сразу, раскидывая их
по нескольким GPU: как только видеокарта освобождается, она забирает
следующую задачу из очереди.

Список задач и доступные GPU задаются в yaml-файле (по умолчанию —
`scripts/queue_jobs.yaml`):

```yaml
# какие GPU можно занимать (значения для CUDA_VISIBLE_DEVICES)
available_gpus: [0, 1]

# каждая строка - это overrides, которые подставятся в
# `python scripts/train.py <строка>` отдельным процессом на своей GPU
experiments:
  - "experiment=b1_vanilla_kd loss.temperature=3 seed=1"
  - "experiment=b2_feature_kd model.student.lr=0.005"
```

Запуск:

```bash
# быстрый разовый прогон — правишь scripts/queue_jobs.yaml и запускаешь без аргументов
python scripts/exp_queue.py

# именной сохранённый батч (например, крупный sweep), который стоит
# сохранить и, может, повторить позже — путь передаётся аргументом
python scripts/exp_queue.py scripts/queue_jobs/imagenet_sweep.yaml
```

Каждая задача — отдельный процесс `python scripts/train.py ...`, поэтому
у неё свои артефакты в `outputs/` и свой `train.log`, как при обычном запуске.

## Тесты

```bash
python -m pytest tests/ -q
```