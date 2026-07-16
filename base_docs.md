# Основное


Используем четыре типа веток:

| Тип | От какой ветки | Вливается в | Когда использовать |
|---|---|---|---|
| `main` | — | — | Всегда рабочий, протестированный код. Только через Pull Request, прямые пуши запрещены. |
| `dev` | `main` | `main` (через PR) | Ветка интеграции. Все фичи и эксперименты сначала попадают сюда. |
| ветви функциональностей | `dev` | `dev` | Разработка фичи или проведение эксперимента (см. префиксы ниже). |
| `hotfix/*` | `main` | `main` и `dev` | Быстрое исправление критичной ошибки в `main` в обход `dev`. |

## main

- **`main` (или `master`):** Всегда содержит рабочий, протестированный код. Сюда нельзя пушить напрямую — только через Pull Requests (PR).

## dev 

- **`develop` (опционально):** Ветка для интеграции новых фич.

## feature и hotfix branches

Могут порождаться от: dev
Должны вливаться в: dev

Правила наименования веток:

Используйте префиксы, чтобы сразу было понятно, над чем идет работа:

| Префикс | Значение | Пример |
|---|---|---|
| `feat/` | Новая функциональность | `feat/kd-loss-implementation` |
| `exp/` | Эксперимент | `exp/resnet50-to-resnet18-baseline` |
| `fix/` | Исправление бага | `fix/dataloader-memory-leak` |
| `docs/` | Только документация / README / `docs/` | `docs/update-configs-guide` |
| `chore/` | Инфраструктура: CI, зависимости, конфиг линтеров и т.п. | `chore/add-precommit-hooks` |
| `hotfix/` | Срочное исправление в `main` | `hotfix/broken-checkpoint-path` |

Под фичей мы тут будем понимать как эксперимент, так и написание кода/документации.
Ветви функциональностей (feature branches), используются для разработки новых функций. Смысл существования ветви функциональности (feature branch) состоит в том, что она живёт так долго, сколько продолжается разработка данной функциональности (фичи). Когда работа в ветви завершена, последняя вливается обратно в главную ветвь разработки или же удаляется (в случае неудачного эксперимента).  
  
Ветви функциональностей (feature branches) в нашем случаи стоит отправлять на github тоже, чтобы другие видели, кто какую взял фичу.

Ветви исправлений (hotfix branches) создаются из главной (main) ветви. Допускается создавать ветки от main для быстрых и мелких исправлений. Но, вероятно нам это не нужно, так как нам не нужно поддерживать постоянную работу какого-то сервиса.

# Commits

Используем Conventional Commits: `<тип>: <описание>`.

- **feat** — новая функциональность.
- **fix** — исправление ошибки.
- **docs** — изменения в документации.
- **style** — форматирование (отступы, запятые), без изменения логики.
- **refactor** — изменение кода без новых фич и без исправления багов.
- **test** — добавление/изменение тестов.
- **chore** — рутинные задачи (сборка, зависимости, конфиги CI).
- **exp** - новый эксперимент

Пример: `feat: add temperature scaling to KD loss`.

# Структура

```markdown
├── .github/                # GitHub Actions для CI/CD, тестов и линтеров
├── docker/
│   ├── Dockerfile
├── configs/                # Конфигурационные файлы (YAML или JSON)
│   ├── data.yaml           # Пути к датасетам, параметры аугментации
│   ├── model.yaml          # Настройки учителя и ученика
│   └── train_kd.yaml       # Гиперпараметры обучения (температура, альфа, эпохи)
├── data/                   # Папка для данных
│   ├── weights/     
│   ├── raw/                # Исходные данные (например, скачанные датасеты)
│   └── processed/          # Предобработанные данные
├── notebooks/              # Jupyter ноутбуки для разведочного анализа (EDA) и baseline
│   └── baseline_1.0_MM_DD.ipynb
├── scripts/                # Точки входа для запуска кода
│   ├── train.py            # Основной скрипт запуска обучения
│   ├── eval.py         # Скрипт для валидации и подсчета метрик
├── src/                    # Основной исходный код (модуль)
│   ├── __init__.py
│   ├── data/               # Всё для работы с данными
│   │   ├── generate.py     # генерация подвыборок, выбор нудного датасета...
│   │   └── transforms.py   # Аугментации, нормировка ... (преобразование данных)
│   ├── models/             # модели
│   │   ├── teachers/ # Тяжелые модели (учителя)
│   │   │     ├── teacher1.py
│   │   │     ├── teacher2.py 
│   │   ├── students/ # Легкие модели (ученики)
│   │   │     ├── student1.py     
│   │   │     ├── student2.py
│   ├── test/               # тесты не для моделей
│   │   ├── test_loss_functions.py
│   │   └── test_transforms.py
│   ├── utils/              # Вспомогательные функции
│   │   ├── metrics.py      # Подсчет метрик (IoU, MAE, Accuracy и т.д.)
│   │   └── logger.py       # Логирование экспериментов
│   ├── docs/               # мини документация 
│   │   ├── docs_configs.md          # Описание структуры конфигов 
│   │   └── docs_models_zoo.md       # Зоопарк моделей
├── .gitignore              # Исключения для Git
├── .dvcignore              # Исключения для DVC
├── .pre-commit-config.yaml # Настройки pre-commit хуков (для форматирования кода)
├── README.md
├── HISTORY.md              # история экспериментов, что использовали, какой конфиг, результаты
└── requirements.txt        # Зависимости проекта
```

# DVC

# ClearML

Три роли:

- **SDK (`clearml`)** — библиотека внутри твоего кода. Логирует в сервер.
- **Server** — у нас это облако `app.clear.ml`. Хранит всё и рисует UI.
- **Agent (`clearml-agent`)** — отдельный процесс на GPU-сервере, он прекрасно работает с облачным сервером.

## Регистрация

1. Зайти на **https://app.clear.ml** и зарегистрироваться (можно через Google/GitHub).
3. **Проекты.** Всё живёт в проектах. Проекты бывают **вложенными** через `/`:

## Установка
```
pip install clearml
```
1. В веб-UI: `Settings → Workspace → App Credentials → Create new credentials`.
2. Появится блок вида:
   ```
   api {
       web_server: https://app.clear.ml
       api_server: https://api.clear.ml
       files_server: https://files.clear.ml
       credentials {
           access_key: "XXXXXXXX"
           secret_key: "YYYYYYYY"
       }
   }
   ```
  Это Не коммитить, не пересылать в открытых чатах.

```bash
clearml-init
```

Альтернатива — переменные окружения (для Docker/CI/серверов)

Вместо файла можно задать env-переменные (удобно в контейнерах, где нет `~/clearml.conf`):

```bash
export CLEARML_WEB_HOST=https://app.clear.ml
export CLEARML_API_HOST=https://api.clear.ml
export CLEARML_FILES_HOST=https://files.clear.ml
export CLEARML_API_ACCESS_KEY=XXXX
export CLEARML_API_SECRET_KEY=YYYY
```

### Проверка

```python
# check_clearml.py
from clearml import Task
task = Task.init(project_name="my project", task_name="my task")
task.get_logger().report_single_value("test_val", 1.0)
print("Task URL:", task.get_output_log_web_page())
task.close()
```

## Базовые понятия

- **Task (эксперимент)** — центральная единица. Один запуск скрипта = одна Task. Внутри Task:
  - **Hyperparameters** — параметры (числа/строки/словари), в т.ч. конфиг Hydra.
  - **Configuration objects** — большие конфиги/файлы целиком.
  - **Artifacts** — произвольные файлы/объекты (предсказания, csv, чекпоинты).
  - **Models** — модели в реестре.
  - **Scalars** — числовые метрики во времени (графики).
  - **Plots / Debug samples** — графики, изображения, матрицы ошибок, аудио.
  - **Console** — перехваченный stdout/stderr.
  - **Info** — git-коммит, ветка, diff, список пакетов, машина, время.
- **Тип задачи** (`task_type`): `Training`, `Testing`, `Inference`, `Data Processing`, `Application`, `Monitor`,`controller`, `optimizer`, `Service`, `qc`, `custom`.
- **Статусы:** `Draft` (создана, но код не бежал) → `Running` → `Completed` / `Failed`
`Published` — «замороженная», доступна только для чтения
  (для эталонных запусков/моделей).
- **Теги** — произвольные метки (`baseline`, `release-candidate`).

## Первый эксперимент

```python
from clearml import Task

task = Task.init(
    project_name="Juicer-for-models/Name",         # проект
    task_name="resnet50_to_resnet18_vanila_KD_training",    # имя запуска
    task_type=Task.TaskTypes.training,             # тип
    tags=["baseline", "kd"],                       # теги
    output_uri=True,                               # куда складывать артефакты/модели
    auto_connect_frameworks=True,                  # авто-перехват фреймворков
)
```

Важные параметры `Task.init`:

| Параметр | Смысл |
|----------|-------|
| `project_name`, `task_name` | Куда и под каким именем |
| `task_type` | Тип |
| `tags` | Список меток |
| `output_uri` | `True` → файловый сервер ClearML; путь к хранилищу. Куда уезжают загруженные артефакты и автоматически сохранённые модели |
| `auto_connect_frameworks` | `True`/`False`/словарь — включить/выключить авто-перехват |
| `auto_connect_arg_parser` | Перехват `argparse` |
| `auto_resource_monitoring` | Мониторинг GPU/CPU/RAM (по умолчанию `True`). |
| `reuse_last_task_id` | Переиспользовать ли «черновую» задачу вместо создания новой. |
| `continue_last_task` | Продолжить предыдущую задачу. |

> Вызвать как можно **раньше** в программе, до создания моделей/тренера.
> Для Hydra — первой строкой внутри функции с @hydra.main.

Закрывать задачу в конце скрипта: `task.close()`.

## Что логируется автоматически

При `Task.init(..., auto_connect_frameworks=True)` ClearML **без единой дополнительной строки**
перехватывает [такие библиотеки](https://github.com/clearml/clearml/tree/master/examples/frameworks):

- **Git:** URL репозитория, ветку, коммит, **и diff незакоммиченных изменений**.
- **Окружение:** список установленных Python-пакетов с версиями.
- **Метрики фреймворков:** и также перехватываются графики **matplotlib/seaborn** (`plt.show()`).
- **Модели:** вызовы `torch.save(...)` - модель попадает в реестр и, если задан `output_uri`, её веса загружаются в хранилище.
- **Аргументы:** `argparse`, `конфиг Hydra`.
- **Консоль:** stdout/stderr.
- **Ресурсы:** утилизация GPU/CPU/RAM/диска во времени.

```python
task = Task.init(
    project_name="Juicer-for-models/Name", task_name=f"{cfg.model.teacher.name}_{cfg.model.teacher.name}_{cfg.exp.name_exp}_{cfg.exp.task_type}",
    auto_connect_frameworks={
        "pytorch": True,
        "matplotlib": True,
        "tensorboard": True,
        "detect_repository": True,   # можно False, чтобы не тянуть git
    },
    auto_connect_arg_parser=False,   # т.к. используем Hydra
)
```

## Явное логирование

[Про явное логирование](https://clear.ml/docs/latest/docs/references/sdk/logger/
)
Самое частое — залогировать словарь параметров.

```python
params = {"lr": 1e-3, "batch_size": 128, "epochs": 60, "optimizer": "adamw"}
params = task.connect(params)   # ВАЖНО: возвращает подключённую версию
```

Секции параметров:

```python
task.connect(model_cfg, name="model")
task.connect(optim_cfg, name="optimizer")
```

### Метрики (scalars) — графики во времени

```python
logger = task.get_logger()

for epoch in range(epochs):
    train_loss = ...
    val_acc1 = ...
    # title = группа/график, series = линия внутри графика
    logger.report_scalar(title="loss", series="train", value=train_loss, iteration=epoch)
    logger.report_scalar(title="loss", series="val",   value=val_loss,   iteration=epoch)
    logger.report_scalar(title="accuracy", series="val_acc1", value=val_acc1, iteration=epoch)
```

Схема именования: **`title`** = карточка-график в UI, **`series`** = линия на нём. Т.е. `train`
и `val` с одинаковым `title="loss"` лягут на **один** график — удобно сравнивать.

Финальные (одиночные) значения без оси времени:

```python
logger.report_single_value(name="best_val_acc1", value=0.712)
```

### Графики, изображения, таблицы, медиа

```python
# Матрица ошибок
logger.report_confusion_matrix(title="val", series="cm", matrix=cm, iteration=epoch)
# Гистограмма
logger.report_histogram(title="weights", series="layer1", values=w, iteration=epoch)
# Готовая фигура matplotlib
logger.report_matplotlib_figure(title="roc", series="val", figure=fig, iteration=epoch)
# Изображение (например, примеры предсказаний)
logger.report_image(title="samples", series="val", image=img_np, iteration=epoch)
# Таблица (pandas DataFrame или список списков)
logger.report_table(title="per_class", series="acc", table_plot=df, iteration=epoch)
# Произвольный текст
logger.report_text("Distillation temperature = 4.0")
```

### Артефакты — произвольные файлы и объекты

```python
# Объекты Python: dict, pandas.DataFrame, numpy.ndarray, PIL.Image
task.upload_artifact(name="val_predictions", artifact_object=predictions_df)
task.upload_artifact(name="class_mapping", artifact_object={"cat": 0, "dog": 1})

# Файлы и папки
task.upload_artifact(name="config_snapshot", artifact_object="outputs/config.yaml")
task.upload_artifact(name="checkpoints", artifact_object="outputs/checkpoints/")  # папка целиком
```

Получить артефакт в другом скрипте:

```python
from clearml import Task
src = Task.get_task(project_name="Juicer-for-models", task_name=f"{cfg.model.teacher.name}_{cfg.model.teacher.name}_{cfg.exp.name_exp}_{cfg.exp.task_type}")
df = src.artifacts["val_predictions"].get()               # объект в память
local_path = src.artifacts["checkpoints"].get_local_copy() # скачать файлы, вернуть путь
```

### Модели и реестр моделей

Модель в ClearML — **отдельная сущность** от артефакта (у неё своя карточка, версии, стадии).

- **Авто-режим:** если `output_uri` задан, `torch.save(model.state_dict(), "student.pt")`
  автоматически регистрирует модель и грузит веса в хранилище.
- **Явный режим:**
  ```python
  from clearml import OutputModel
  output_model = OutputModel(task=task, name="student-resnet18", framework="PyTorch")
  torch.save(student.state_dict(), "student.pt")
  output_model.update_weights(weights_filename="student.pt")
  output_model.publish()   # заморозить как релизную версию
  ```
- **Загрузить модель** в инференсе:
  ```python
  from clearml import InputModel
  model = InputModel(model_id="<id>")   # или по имени/проекту
  weights = model.get_weights()         # локальный путь к весам
  ```

---

## Hydra + ClearML

ClearML **автоматически распознаёт Hydra**. При запуске скрипта с `@hydra.main` и вызовом
`Task.init`:

**Полный собранный конфиг** (весь `OmegaConf` после композиции defaults/overrides) сохраняется
  как **Configuration object** с именем **`OmegaConf`** — виден в разделе *Configuration* задачи.

То есть даже без `task.connect(cfg)` весь конфиг уже будет в задаче. `task.connect` при желании
можно добавить, чтобы отдельные параметры лежали плоско и участвовали в сравнении/HPO.

### Рекомендуемый шаблон `train.py`

```python
# scripts/train.py
import hydra
from omegaconf import DictConfig, OmegaConf
from clearml import Task


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # 1) Task.init — ПЕРВОЙ строкой внутри main (после того как Hydra собрала cfg)
    task = Task.init(
        project_name="Juicer-for-models",
        task_name=f"{cfg.model.teacher.name}_{cfg.model.teacher.name}_{cfg.exp.name_exp}_{cfg.exp.task_type}",
        task_type=cfg.exp.task_type,
        output_uri=True,
        auto_connect_arg_parser=False,  # у нас Hydra, не argparse
    )

    # 2) (опционально) продублировать конфиг плоско — удобно для сравнения
    task.connect(OmegaConf.to_container(cfg, resolve=True))

    logger = task.get_logger()

    # 3) обычный цикл обучения на голом PyTorch
    model = build_model(cfg.model)
    train_loader, val_loader = build_data(cfg.data)
    optimizer = build_optimizer(model, cfg.optim)

    for epoch in range(cfg.trainer.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_acc1 = validate(model, val_loader)
        logger.report_scalar("loss", "train", train_loss, iteration=epoch)
        logger.report_scalar("accuracy", "val_acc1", val_acc1, iteration=epoch)

    task.close()


if __name__ == "__main__":
    main()
```

> **Почему `Task.init` внутри `main`, а не на уровне модуля:** Hydra при входе в `main` меняет
> рабочую директорию и только там даёт собранный `cfg`. Вызов внутри `main` гарантирует, что
> ClearML захватит уже готовый конфиг и корректную рабочую папку.

## Датасеты: ClearML Data

# CI/CD + ruff + Pyrefly

# Docker 

# Hydra 

# Lightning 

# Документация
README.md — входная точка: что за проект, как запустить (Docker в первую очередь), как быстро прогнать обучение, ссылки на остальную документацию.
HISTORY.md — журнал экспериментов. Заполняется при каждом мало-мальски значимом эксперименте.
docs/ — документация по конкретным подсистемам (конфиги, зоопарк моделей, DVC, ClearML), чтобы README не разрастался.