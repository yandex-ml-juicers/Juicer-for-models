# Juicer-for-models

R&D-фреймворк дистилляции CV-моделей (учитель → ученик) на PyTorch + Hydra.
Бейзлайны из `notebooks/` распилены на модули: каждый эксперимент — это YAML-конфиг,
а не копия тренировочного цикла.

Не работал с Hydra — начни с мини-гайда [docs/hydra.md](docs/hydra.md).

## Установка

```bash
pip install -r requirements.txt
pip install -e . --no-deps        # пакет src/ становится импортируемым
# для тестов: pip install -e ".[dev]" --no-build-isolation
```

## Быстрый старт

```bash
# Бейзлайн 1: ванильная дистилляция Хинтона, ResNet-56 -> ResNet-20
python scripts/train.py experiment=b1_vanilla_kd

# Контроль к нему: тот же ученик с нуля, без учителя
python scripts/train.py experiment=b1_scratch

# Бейзлайн 2: feature-based KD, ResNet-50 -> ResNet-18 (логиты + карты признаков)
python scripts/train.py experiment=b2_feature_kd
python scripts/train.py experiment=b2_scratch

# Любой гиперпараметр меняется из CLI, без правки кода:
python scripts/train.py experiment=b1_vanilla_kd loss.temperature=8 trainer.epochs=30 seed=1

# Смоук-тест пайплайна: синтетические данные, CPU, без сети, ~10 секунд
python scripts/train.py data=fake '~model/teacher' loss=ce \
    trainer.epochs=1 trainer.limit_train_batches=3 trainer.limit_eval_batches=2

# Оценка сохранённого чекпоинта
python scripts/eval.py experiment=b2_feature_kd ckpt_path=outputs/<name>/<run>/best.pt
```

Артефакты каждого запуска — в `outputs/<name>/<дата_время>/`:
`.hydra/config.yaml` (полный снапшот конфига), `train.log`, `history.csv`
(метрики и все компоненты лосса по эпохам), `best.pt` / `last.pt`.

## Структура

```
configs/                # композиция Hydra: оси эксперимента
  config.yaml           #   корень: defaults + seed/device/deterministic
  data/                 #   cifar10, fake (синтетика для смоуков/CI)
  model/teacher/        #   предобученные учителя (torch.hub, detectors)
  model/student/        #   ученики (hub, адаптированный torchvision)
  loss/                 #   ce | hinton_kd | feature_kd
  optimizer/            #   adam | adamw (_partial_: params подставляет train.py)
  scheduler/            #   cosine (T_max = ${trainer.epochs})
  trainer/              #   epochs, amp, limit_batches, чекпоинты
  experiment/           #   1 бейзлайн = 1 файл: готовые сочетания осей
src/
  data/                 # transforms, датасеты, DataLoader'ы с сидированием воркеров
  models/               # фабрики моделей, FeatureExtractor (хуки), 1x1-адаптеры
  losses/               # DistillationLoss-интерфейс и реализации
  training/             # Trainer: единый цикл для всех режимов
  utils/                # seed_everything, метрики, CSV-история
scripts/                # точки входа: train.py, eval.py
tests/                  # лоссы, детерминизм, композиция конфигов
```

## Архитектура в трёх предложениях

**Лосс — это модуль со своими требованиями.** Каждый лосс наследует
`DistillationLoss` и декларирует `requires_teacher` и `required_features`;
`Trainer` по этим декларациям решает, грузить ли учителя и вешать ли
forward-хуки на промежуточные слои. У `FeatureKD` есть обучаемые
1×1-адаптеры каналов — они живут внутри лосса, и optimizer собирается из
`student.parameters() + criterion.parameters()` (Trainer это проверяет).

**Учитель — всегда заморожен** (eval + requires_grad=False), этим владеет
Trainer, а не конфиг модели.

**Эксперимент — это YAML.** `configs/experiment/*.yaml` переопределяет оси
(`data`, `model/teacher`, `model/student`, `loss`, `optimizer`) и значения
поверх них. Обучение без дистилляции — не отдельный скрипт, а конфиг:
`loss=ce` + удалённый учитель (`- override /model/teacher: null`).

## Воспроизводимость

`seed_everything` (см. [src/utils/seed.py](src/utils/seed.py)) фиксирует Python/NumPy/torch/CUDA,
включает детерминированные CUDA-ядра (`use_deterministic_algorithms`,
`CUBLAS_WORKSPACE_CONFIG`, cudnn.deterministic) и сидирует воркеры
DataLoader (`worker_init_fn` + отдельный `torch.Generator` для shuffle).

- Два запуска с одним конфигом дают **бит-в-бит** одинаковые веса и метрики
  (проверено тестами и двойным смоук-прогоном).
- Гарантия действует на одном железе и версиях torch/CUDA; между разными GPU
  результаты могут отличаться — это свойство CUDA, а не кода.
- Детерминизм стоит ~10–20% скорости. Для черновых прогонов можно отключить:
  `deterministic=false`.

## Как добавить

- **Лосс**: класс-наследник `DistillationLoss` в `src/losses/` + YAML в
  `configs/loss/` с `_target_`. Trainer менять не нужно.
- **Модель**: фабричная функция в `src/models/factory.py` (или готовый
  `_target_` из torch.hub/timm) + YAML в `configs/model/{teacher,student}/`.
- **Эксперимент**: YAML в `configs/experiment/` с `# @package _global_`
  и списком `override`-ов (см. существующие как образец).

## Соответствие ноутбукам

| Ноутбук | Эксперименты |
|---|---|
| `baseline_1.0_11_07` | `b1_vanilla_kd` + контроль `b1_scratch` |
| `baseline_2.0_12_07` | `b2_feature_kd` + контроль `b2_scratch` |
| `baseline_3.0_12_07` | отложен: ансамбль учителей ляжет как `TeacherEnsemble(nn.Module)` + конфиг в `model/teacher/`, файнтюн учителей — как experiment с `loss=ce` |

## ClearML

Трекинг выключен по умолчанию (смоуки и CI работают оффлайн). Для запуска
с логированием в ClearML нужен настроенный `clearml.conf` или переменные
`CLEARML_*` (см. base_docs.md), дальше:

```bash
python scripts/train.py experiment=b1_vanilla_kd clearml.enabled=true
```

В задачу уезжают: полный Hydra-конфиг, консоль, git-коммит и diff,
чекпоинты (`output_uri=true`), скаляры по эпохам (loss train/eval,
accuracy, lr, компоненты лосса) и итоговый `best_eval_acc`. Теги задаются
в experiment-конфиге (`clearml.tags`), имя задачи = `name` запуска.

## Тесты

```bash
python -m pytest tests/ -q
```

Покрывают формулы лоссов (граничные случаи: `alpha=0`, идентичные логиты,
identity-адаптеры), детерминизм (init моделей, порядок данных), композицию
всех experiment-конфигов и согласованность «лосс ↔ наличие учителя».
