# Hydra: как запускать эксперименты в этом репо

Мини-гайд для погружения в Hydra. Цель — чтобы через 10 минут ты мог
запустить любой бейзлайн, поменять любой гиперпараметр и завести свой 
эксперимент, не трогая Python-код.

## Зачем нам это

В ноутбуках каждый эксперимент — копия тренировочного цикла с другими числами
внутри. Через месяц невозможно ответить, чем именно запуск А отличался от Б.

С Hydra код один, а всё, что можно крутить (модели, лосс, батч, темпер­атура,
seed), вынесено в YAML-файлы в `configs/`. Эксперимент — это файл конфига.
Запуск любого параметра с любым значением — одна команда, и каждая команда
оставляет после себя папку с точным снапшотом конфига (.hydra), по которому запуск
можно повторить бит-в-бит.

## Что такое Hydra за 30 секунд

[Hydra](https://hydra.cc) — библиотека управления конфигурацией от Meta.
Делает три вещи:

1. **Композиция.** Перед запуском `main()` собирает один большой конфиг из
   маленьких YAML-файлов (по файлу на модель, лосс, датасет...) и отдаёт его
   в функцию объектом `cfg`. В коде это выглядит как `cfg.trainer.epochs`.
2. **Переопределения из CLI.** Любое значение можно поменять прямо в команде
   запуска: `trainer.epochs=30`. Никаких правок файлов ради разового прогона.
3. **Папка на запуск.** Каждый запуск получает свою директорию
   (`outputs/<имя>/<дата_время>/`) с логами и копией итогового конфига.

## Запуск бейзлайнов

```bash
# Бейзлайн 1: дистилляция Хинтона ResNet-56 -> ResNet-20
python scripts/train.py experiment=b1_vanilla_kd

# Его контроль: тот же ученик, но с нуля, без учителя
python scripts/train.py experiment=b1_scratch

# Бейзлайн 2: feature-based KD ResNet-50 -> ResNet-18, и его контроль
python scripts/train.py experiment=b2_feature_kd
python scripts/train.py experiment=b2_scratch

# Смоук-тест: синтетические данные, CPU, без сети, ~10 секунд
python scripts/train.py data/dataset=fake_cifar10 '~model/teacher' loss=ce \
    trainer.epochs=1 trainer.limit_train_batches=3 trainer.limit_eval_batches=2
```

`experiment=...` — это выбор готового пресета из `configs/experiment/`.
Всё остальное в команде — точечные переопределения поверх него. Кроме
CIFAR-бейзлайнов выше в этой папке лежит растущий набор ImageNet-экспериментов
(scratch/vanilla-KD/ESKD для resnet18/50/152) — актуальный список смотри
командой `ls configs/experiment/` или `python scripts/train.py --help`,
здесь их специально не перечисляем: список меняется чаще, чем этот файл.

## Как устроена папка configs/

Каждая подпапка — **группа конфигов**, она же «ось» эксперимента. Файл внутри
папки — вариант значения этой оси. Выбираешь по одному файлу на ось — получаешь
эксперимент.

```
configs/
├── config.yaml         # корень: какие оси есть и что выбрано по умолчанию
├── data/
│   ├── dataset/        # cifar10.yaml, fake_cifar10.yaml, imagenet1k.yaml, imagenet100.yaml
│   ├── loader/         # base_loader.yaml (batch_size, num_workers, ...)
│   └── transform/
│       ├── train/      # base_transform.yaml — трансформ для train_loader
│       └── eval/       # base_transform.yaml — трансформ для eval_loader
├── model/
│   ├── teacher/        # resnet56_cifar.yaml, resnet50_cifar.yaml, resnet50/152_imagenet_pretrained.yaml
│   └── student/        # resnet20_cifar.yaml, resnet18_cifar32.yaml, resnet18/50/152_imagenet.yaml
├── loss/                # ce.yaml, hinton_kd.yaml, feature_kd.yaml
├── optimizer/           # adam.yaml, adamw.yaml, SGD.yaml
├── scheduler/           # cosine.yaml, warmup_cosine.yaml
├── trainer/             # default.yaml (эпохи, AMP, чекпоинты)
└── experiment/          # готовые сочетания осей: b1_vanilla_kd.yaml, imagenet1k_scratch_resnet18.yaml, ...
```

`data/` — не одна ось, а четыре независимые: **какой датасет**, **какими
значениями грузить его в DataLoader** (`batch_size`, `num_workers`, ...) и
**как строить трансформы** — отдельно для train и для eval
(`_target_: src.data.transforms.base_transform`, а `mean`/`std`/`image_size` —
интерполяции на выбранный `data/dataset`, не свои числа).

Один `loader`-файл и одна пара `train`/`eval` transform-файлов
обслуживают любой датасет — переключаешь только `data/dataset`, остальные три
оси обычно не трогаешь.

Дефолтный выбор записан в начале `configs/config.yaml`:

```yaml
defaults:
  - _self_
  - data/dataset: cifar10       # <группа>: <имя файла без .yaml>
  - data/loader: base_loader
  - data/transform/train: base_transform
  - data/transform/eval: base_transform
  - model/teacher: resnet56_cifar
  - model/student: resnet20_cifar
  - loss: hinton_kd
  - optimizer: adam
  - scheduler: cosine
  - trainer: default
  - experiment: null            # пресет по умолчанию не выбран
```

Итоговый конфиг = склейка выбранных файлов, где содержимое `loss/hinton_kd.yaml`
оказывается в `cfg.loss`, `data/dataset/cifar10.yaml` — в `cfg.data.dataset`,
и т.д. Важно: `cifar10`/`imagenet1k` — это имя *выбранного файла*, а не ключ
в дереве — оно нигде не остаётся после сборки (подробнее — раздел
[«Пакеты» в config_yaml.md](config_yaml.md)).

**Посмотреть итоговый конфиг, не запуская обучение:**

```bash
python scripts/train.py experiment=b2_feature_kd --cfg job
```

**Посмотреть все группы и их варианты:** `python scripts/train.py --help`.

## Переопределения из CLI — шпаргалка

| Что сделать | Синтаксис | Пример |
|---|---|---|
| Поменять значение | `путь.к.ключу=значение` | `trainer.epochs=30 seed=1` |
| Выбрать другой файл группы | `группа=файл` | `loss=ce optimizer=adamw` |
| И то и другое сразу | сначала группа, потом значения | `loss=hinton_kd loss.temperature=8` |
| Убрать группу совсем | `'~группа'` (в кавычках!) | `'~model/teacher'` — обучение без учителя |
| Добавить ключ, которого нет | `+ключ=значение` | `+trainer.new_flag=true` |
| Запустить серию (multirun) | `-m ключ=v1,v2,v3` | `-m seed=1,2,3` → три запуска подряд |

Два правила, которые сначала путают:

- `loss=ce` и `loss.temperature=8` — разные операции: первая **заменяет файл
  целиком**, вторая меняет **одно значение** внутри уже выбранного файла.
- Опечатка в имени ключа — это **ошибка запуска**, а не молча созданный новый
  ключ. Это защита: `trainer.epohs=30` не потеряется втихую. Новые ключи
  добавляются только явно, через `+`.

## Файлы экспериментов (пресеты)

`configs/experiment/*.yaml` — «сохранённые» сочетания осей + значений.
Разберём `b1_vanilla_kd.yaml`:

```yaml
# @package _global_          # служебная строка: «пиши мои ключи в корень
                             # конфига, а не в cfg.experiment» — просто
                             # всегда оставляй её первой строкой
defaults:                    # какие файлы групп выбрать
  - override /data/dataset: cifar10
  - override /data/loader: base_loader
  - override /data/transform/train: base_transform
  - override /data/transform/eval: base_transform
  - override /model/teacher: resnet56_cifar
  - override /model/student: resnet20_cifar
  - override /loss: hinton_kd
  - override /optimizer: adam
  - override /scheduler: cosine

name: b1_resnet56_to_resnet20_kd   # имя запуска -> подпапка в outputs/

trainer:                     # точечные значения поверх выбранных файлов
  epochs: 20
loss:
  temperature: 4.0
  alpha: 0.9
data:
  dataset:
    normalize:
      std: [0.2023, 0.1994, 0.2010]   # путь: cfg.data.dataset.normalize.std
  loader:
    batch_size: 128
```

Первые четыре строки `defaults` не меняют выбор (в корне и так `cifar10`/
`base_loader`/`base_transform` для train и eval) — переобъявлены явно, ради
самодокументируемости эксперимента. Раз группа уже выбрана в корне, повторное
упоминание всегда через `override`, даже если значение то же самое — иначе
Hydra не поймёт, что это изменение существующего выбора, а не второе
объявление той же оси (`ConfigCompositionException: Could not override ...`).

`data/transform/train` и `data/transform/eval` — это **две отдельные
группы**, не одна группа с двумя значениями: переопределять (или выбирать)
их нужно **обеими** строками по отдельности, даже если в обоих случаях
выбирается один и тот же файл `base_transform.yaml`. Группы `data/transform`
(без `/train` или `/eval`) не существует — обращение к ней целиком, одной
строкой, упадёт `Could not override 'data/transform'. No match in the
defaults list.`

**Завести свой эксперимент** = скопировать ближайший по смыслу файл, поменять
`name`, оси и значения. Всё. Код не трогаем.

Приоритет, если одно и то же задано в нескольких местах (каждый следующий
перебивает предыдущего):

```
дефолты групп  <  файл эксперимента  <  переопределения из CLI
```

Нюанс синтаксиса: «выключить группу» в CLI пишется `'~model/teacher'`,
а внутри yaml-файла эксперимента — `- override /model/teacher: null`
(см. `b1_scratch.yaml`). Действие одно, синтаксис разный.

## `_target_`: объекты прямо из YAML

Во многих конфигах есть ключ `_target_` — это путь к Python-классу или функции.
Такой конфиг — «рецепт объекта»: `train.py` вызывает `instantiate(cfg.loss)`,
и Hydra строит объект, передав остальные ключи как аргументы конструктора.

```yaml
# configs/loss/hinton_kd.yaml
_target_: src.losses.HintonKD
temperature: 4.0
alpha: 0.9
```

эквивалентно `HintonKD(temperature=4.0, alpha=0.9)`.

Поэтому добавление нового лосса или модели не требует правок `train.py`:
пишешь класс, кладёшь yaml с `_target_` — и он уже доступен как `loss=мой_лосс`.

У оптимизаторов и шедулеров в конфиге есть `_partial_: true` — им нужен
аргумент, который существует только в рантайме (`params` модели, `optimizer`).
Hydra в этом случае возвращает заготовку, а `train.py` довызывает её с нужным
аргументом. Знать это нужно только если будешь добавлять свой оптимизатор.

## `${...}`: ссылки внутри конфига

Значение можно не дублировать, а сослаться на другое место конфига:

```yaml
# configs/scheduler/cosine.yaml
T_max: ${trainer.epochs}     # длина косинуса всегда равна числу эпох
```

Поменял `trainer.epochs=30` из CLI — `T_max` подтянулся сам. Так же
`num_classes` ученика ссылается на `${data.dataset.num_classes}` — размер
последнего слоя классификатора всегда соответствует выбранному `data/dataset`,
переключил `cifar10` на `imagenet1k` — число классов подтянулось само, без
правки конфига модели.

## Что остаётся после запуска

```
outputs/<name>/<дата_время>/
├── .hydra/
│   ├── config.yaml      # ПОЛНЫЙ итоговый конфиг запуска (снапшот)
│   └── overrides.yaml   # что было переопределено из CLI
├── train.log            # всё, что печаталось в консоль
├── history.csv          # метрики и компоненты лосса по эпохам
├── best.pt              # чекпоинт лучшей эпохи (по eval accuracy)
└── last.pt              # чекпоинт последней эпохи
```

Хочешь понять, чем был запуск двухнедельной давности — открой его
`.hydra/config.yaml`. Хочешь его повторить — примени те же overrides
(благодаря `seed` и `deterministic: true` результат совпадёт бит-в-бит).

Оценить сохранённый чекпоинт:

```bash
python scripts/eval.py experiment=b2_feature_kd \
    ckpt_path=outputs/b2_resnet50_to_resnet18_feature_kd/<дата_время>/best.pt
```


Как воспроизвести результат?

Вариант А (опасно): Повторить ту же команду (взять список из overrides.yaml (outputs/default/папка запуска/.hydra/overrides.yaml) и подставить обратно, например experiment=b1_scratch ...). Однако гарантия не полная (конфиг эксперимента может быть уже изменён)

Вариант Б (рекомендуется): загрузить сам снапшот напрямую в обход текущего состояния configs/:
```bash
python scripts/train.py \
    --config-path /абсолютный/путь/outputs/<name>/<дата_время>/.hydra \
    --config-name config \
    hydra.run.dir=outputs/replay
```


## Частые грабли

- **`~` без кавычек.** Bash может истолковать `~model/teacher` как домашнюю
  директорию. Пиши `'~model/teacher'`.
- **Меняешь конфиг, а эффекта нет.** Проверь, не перебивает ли твоё значение
  файл эксперимента или CLI (см. приоритеты выше). Быстрая диагностика —
  `--cfg job`: печатает итоговый конфиг без запуска обучения.
- **`Could not override 'x'. Key not found`** — опечатка в имени ключа или
  ты пытаешься добавить новый ключ без `+`.
- **Ищешь, какие есть варианты у оси** — просто загляни в папку группы:
  `ls configs/loss/`.

## Куда дальше

Официальный туториал короткий и хороший: https://hydra.cc/docs/intro/ —
для работы с этим репо достаточно разделов Basic Tutorial → Config groups,
Defaults, Instantiating objects.
