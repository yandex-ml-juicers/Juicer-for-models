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

# если хотим запускать квантизацию:
pip install -e ".[deploy]"          # onnx + onnxscript + onnxruntime

# TensorRT — ТОЛЬКО с суффиксом CUDA и не старше 10.x, иначе не заработает:
#   без суффикса ставится сборка под CUDA 13 (cudaError 35 на драйвере под 12);
#   в 10.16+ и 11.x удалена поддержка Volta, то есть V100.
python -c "import torch; print(torch.version.cuda)"   # -> 12.6
pip install 'tensorrt-cu12==10.3.0'
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

## Аугментации и регуляризации

По умолчанию включён только базовый набор (crop + flip + jitter), чтобы
бейзлайны оставались сравнимыми. Всё остальное подключается конфигом:

```bash
# Mixup + CutMix (классификация) / CutMix (сегментация) — группа augment
python scripts/train.py experiment=<...> augment=mixup_cutmix
python scripts/train.py experiment=<...> augment=cutmix_segmentation

# усиленный набор аугментаций одного примера
python scripts/train.py experiment=<...> data/transform/train=cityscapes_seg_strong_transform
python scripts/train.py experiment=<...> data/transform/train=imagenet_strong_transform

# регуляризация внутри модели
python scripts/train.py experiment=<...> model.student.drop_path_rate=0.2   # SegFormer / timm / ResNet
python scripts/train.py experiment=<...> model.student.dropout=0.1          # U-Net
```

Что каждый рычаг делает, когда его включать и почему для сегментации нужен
CutMix, а не Mixup — в [docs/augmentations.md](docs/augmentations.md).

Отдельный случай — дистилляция: учитель предобучен на чистых кадрах и на
сильной фотометрии проседает, то есть отдаёт ученику испорченные таргеты.
`teacher_skips` в train-трансформе даёт ученику сильный кадр, а учителю —
слабый, при общей геометрии. Что показывать учителю, а что нет — меряется:

```bash
python scripts/probe_teacher_augmentations.py experiment=<...> +probe.samples=50
```

## Группы параметров оптимизатора

Предобученный энкодер и случайно инициализированный декодер не должны
учиться одним и тем же шагом, а weight decay не должен применяться к
BatchNorm и bias. По умолчанию выключено (одна группа на всё):

```yaml
optimizer:
  lr: 1.0e-3
  weight_decay: 3.0e-2

param_groups:
  encoder_lr_mult: 0.1              # энкодеру 1.0e-4 вместо 1.0e-3
  no_decay_on_norm_and_bias: true
```

Таблица групп печатается в `train.log`, а в ClearML на графике `lr`
появляются отдельные серии `encoder` / `decoder`. Подробности и подводные
камни (в первую очередь `eta_min`) — в [docs/param_groups.md](docs/param_groups.md).

## Лоссы

```bash
# лоссы сегментации: CE + Dice / трудные пиксели / прямая оптимизация IoU
python scripts/train.py experiment=<...> loss=dice
python scripts/train.py experiment=<...> loss=ohem      # loss=focal
python scripts/train.py experiment=<...> loss=lovasz

# дистилляция: граница отдельно от тела, разнородные архитектуры
python scripts/train.py experiment=<...> loss=bpkd
python scripts/train.py experiment=<...> loss=heteroakd

# несколько лоссов сразу (CE + FitNets + Dice)
python scripts/train.py experiment=<...> loss=composite_fitnets_dice
```

Веса слагаемых можно менять по ходу обучения — например гасить дистилляцию
к концу, когда учитель начинает тянуть ученика к своим ошибкам:

```yaml
loss_schedule:
  hint_weight: {schedule: cosine, start: 0.7, end: 0.0, start_epoch: 120}
  ce_weight:   {schedule: cosine, start: 0.3, end: 1.0, start_epoch: 120}
```

И отдельно — мультимасштабные таргеты учителя (несколько прогонов вместо
одного, зато чище):

```yaml
model:
  teacher_inference: {scales: [0.75, 1.0, 1.25], flip: true}
```

Подробности — в [docs/losses.md](docs/losses.md).

## SegNeXt

Свёрточный сегментатор, который на Cityscapes догоняет трансформеры своего
размера (T: 4.3M / 79.8 mIoU, S: 13.9M / 81.3, B: 27.6M / 82.6, L: 48.9M / 83.2).
Годится и учеником, и учителем:

```bash
# ученик: энкодер MSCAN с ImageNet качается автоматически
python scripts/train.py experiment=scratch/cityscapes_scratch_segnext_t
python scripts/train.py experiment=<...> model/student=segnext model.student.variant=s

# учитель: веса на Cityscapes нужно скачать руками
python scripts/train.py experiment=<...> model/teacher=segnext \
    model.teacher.checkpoint_path=data/weights/segnext/segnext_base_1024x1024_city.pth
```

Автоматически скачиваемых cityscapes-весов у SegNeXt нет: OpenMMLab
опубликовал только ADE20K, а чекпоинты авторов лежат на TsingHua Cloud
(таблица Cityscapes в README
[Visual-Attention-Network/SegNeXt](https://github.com/Visual-Attention-Network/SegNeXt)).
Файл кладётся в `data/weights/segnext/` и указывается в `checkpoint_path` —
ключи переименовывать не нужно, конвертер понимает и mmsegmentation, и
оригинальный репозиторий. Второй путь, без чужих файлов, — обучить SegNeXt
самим (`scratch/cityscapes_scratch_segnext_t`) и подставить `best.pt`.

Артефакты каждого запуска — в `outputs/<name>/<дата_время>/`:
`.hydra/config.yaml` (полный снапшот конфига), `train.log`, `history.csv`
(метрики и все компоненты лосса по эпохам), `best.pt` / `last.pt`.

## ESPNetv2

Энкодер EESP (arXiv:1811.11431) — только ученик. В отличие от большинства
lightweight-сеток этого класса, у EESP есть собственные ImageNet-веса
классификатора (а не только чужая сборка целиком на Cityscapes) — декодер
всё равно случайный, ученик проходит весь KD-рецепт проекта с нуля. Два
варианта декодера поверх ОДНОГО и того же энкодера:

```bash
# наш UNetDoubleConv — декодер держится постоянным между всеми студентами
# TimmUNet/ESPNetv2 проекта (чистая абляция "что даёт энкодер"); s050 — 0.92M
python scripts/train.py experiment=<...> model/student=espnetv2 model.student.variant=s050

# декодер из оригинальной статьи (EESPNet_Seg) — работает в num_classes-мерном
# пространстве, поэтому на порядок легче: s200 — 1.25M, тот же бюджет, что
# и espnetv2/s050 сверху, но БЕЗ страйда 32 (taps.stage4 не существует —
# FitNets/HeteroAKD с ним только на stage1-3)
python scripts/train.py experiment=<...> model/student=espnetv2_native model.student.variant=s200
```

Доступные варианты — `s050`/`s100`/`s125`/`s150`/`s200` (масштаб `s` из
статьи, общий для обоих декодеров; веса качаются автоматически, см.
`src/models/espnetv2.py`).

## Квантизация (PTQ fp32 -> fp16)

Обученный чекпоинт превращается в движок под конкретное железо одной командой:

```bash
pip install -e ".[deploy]"   # onnx + onnxscript + onnxruntime

CUDA_VISIBLE_DEVICES=2 python scripts/quantize.py experiment=<...> \
    ckpt_path=outputs/<name>/<run>/best.pt quantize=trt_fp16
```

Карта выбирается через `CUDA_VISIBLE_DEVICES`: TensorRT собирает движок на
текущем устройстве процесса, и прятать чужие карты надёжнее, чем указывать
индекс.

Стадии (`export -> calibrate -> build -> validate -> benchmark`) перечислены в
`quantize.stages`; отчёты остаются в папке запуска, а `.onnx` и `.engine` -
в `data/deploy/<model_id>/`. Подробности, три уровня валидации и как читать
замеры — в [docs/quantization.md](docs/quantization.md).

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