
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

       project_name="Juicer-for-models",
        task_name=f"{cfg.name}",
        task_type=cfg.task_type,
        reuse_last_task_id = cfg.reuse_last_task_id,
        output_uri= cfg.output_uri,

task = Task.init(
    project_name=cfg.clearml.project,                      # проект
    task_name=f"{cfg.name}",                               # имя таски
    task_type=Task.TaskTypes.training,                       # тип таски
    tags=cfg.clearml.tags,                                 # теги
    reuse_last_task_id = cfg.clearml.reuse_last_task_id,   # перезаписывать ли таску с таким же именем
    continue_last_task=cfg.clearml.continue_last_task,     # Подхватит предыдущий ID и продолжит логирование
    output_uri=cfg.clearml.output_uri,                     # складывать ли артефакты/модели и если куда-то базово, то url
    auto_connect_frameworks=cfg.clearml.auto_connect_frameworks, # авто-перехват фреймворков
    auto_connect_arg_parser=cfg.clearml.auto_connect_arg_parser, # авто-перехват аргументов из argparse
    
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
    project_name="Juicer-for-models/Name",
    task_name=f"{cfg.name}",
    auto_connect_frameworks={
        "pytorch": True,
        "matplotlib": True,
        "tensorboard": True,
        "detect_repository": True,   # можно False, чтобы не тянуть git
    },
    auto_connect_arg_parser=cfg.clearml.auto_connect_arg_parser,   # т.к. используем Hydra
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
src = Task.get_task(project_name="Juicer-for-models", task_name=f"{cfg.name}")
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
    project_name=cfg.clearml.project,                      # проект
    task_name=f"{cfg.name}",                               # имя таски
    task_type=Task.TaskTypes.training,                     # тип таски
    tags=cfg.clearml.tags,                                 # теги
    reuse_last_task_id = cfg.clearml.reuse_last_task_id,   # перезаписывать ли таску с таким же именем
    continue_last_task=cfg.clearml.continue_last_task,     # Подхватит предыдущий ID и продолжит логирование
    output_uri=cfg.clearml.output_uri,                     # складывать ли артефакты/модели и если куда-то базово, то url
    auto_connect_frameworks=cfg.clearml.auto_connect_frameworks, # авто-перехват фреймворков
    auto_connect_arg_parser=cfg.clearml.auto_connect_arg_parser, # авто-перехват аргументов из argparse
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