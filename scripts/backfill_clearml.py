"""Дозаливка в ClearML прогона, который уже закончился локально, но не
залогировался (например, quota metrics_storage прервала events.add_batch —
сам прогон при этом не падает, история пишется в history.csv независимо от
ClearML, см. scripts/train.py::MetricsHistory).

Читает РОВНО ТЕ ЖЕ артефакты, что оставляет обучение в outputs/<name>/<run>/:
  .hydra/config.yaml  — разрешённый конфиг (тот же, что connect_configuration
                        кладёт в Configuration objects таски)
  history.csv         — построчно эпоха -> все метрики; колонки этого файла
                        и есть тот самый all_values, который MetricsHistory
                        и clearml_reporter получают ОДНИМ И ТЕМ ЖЕ вызовом
                        (см. SegmentationTrainer.fit()) — поэтому скрипт не
                        переизобретает роутинг метрик по графикам, а зовёт
                        ТУ ЖЕ clearml_reporter() из scripts/train.py.

Чего в history.csv нет и чего этот скрипт поэтому НЕ восстановит: раздел
Plots (per-class IoU, confusion matrix, debug-картинки, гистограмма KL) —
это сырые тензоры/картинки, они уходили в ClearML напрямую и на диск не
писались. Если нужны и они — единственный путь — заново прогнать eval с
сохранённого чекпоинта (best.pt/last.pt лежат рядом) и это отдельная задача.

Запуск на каждом сервере отдельно — скрипт работает только с локальным
outputs/, ничего никуда по сети кроме ClearML не ходит.

    python scripts/backfill_clearml.py outputs/<name>/<дата_время>
    python scripts/backfill_clearml.py <name>              # если run один
    python scripts/backfill_clearml.py <name> --dry-run    # без ClearML,
                                                             # только проверить
"""

import argparse
import csv
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

# При запуске как файла (python scripts/backfill_clearml.py ...) Python кладёт
# в sys.path[0] каталог САМОГО СКРИПТА (scripts/), а не корень репозитория —
# "from scripts.train import ..." ниже иначе не находит пакет scripts.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_run_dir(arg: str) -> Path:
    """Путь до run-каталога (.../<name>/<дата_время>) — принимает и его
    напрямую, и голое имя эксперимента (тогда каталог ищется в outputs/)."""
    direct = Path(arg)
    if (direct / ".hydra" / "config.yaml").is_file():
        return direct

    by_name = OUTPUTS / arg
    if not by_name.is_dir():
        raise FileNotFoundError(
            f"Не нашёл ни {direct} (нужен .hydra/config.yaml внутри), ни {by_name}"
        )

    candidates = sorted(
        p for p in by_name.iterdir() if (p / ".hydra" / "config.yaml").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"В {by_name} нет ни одного run-каталога с .hydra/config.yaml")
    if len(candidates) > 1:
        listing = "\n".join(f"  {c}" for c in candidates)
        raise ValueError(
            f"В {by_name} несколько прогонов — укажите нужный путь целиком:\n{listing}"
        )
    return candidates[0]


def load_history(run_dir: Path) -> list[dict]:
    history_path = run_dir / "history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"Нет {history_path} — нечего заливать")
    with history_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{history_path} пустой")
    return rows


def best_metric_from_history(rows: list[dict], metric_key: str) -> tuple[float, int]:
    """Тот же критерий, что у Trainer: is_best = eval_metric > best (строго),
    первое совпадение при равенстве побеждает — см. SegmentationTrainer.fit()."""
    best_value, best_epoch = 0.0, 0
    for row in rows:
        raw = row.get(metric_key)
        if raw in (None, ""):
            continue
        value = float(raw)
        if value > best_value:
            best_value, best_epoch = value, int(float(row["epoch"]))
    return best_value, best_epoch


def build_param_table_safely(cfg):
    """Таблица student/teacher/criterion — веса не важны (нужны только
    размеры/число параметров), поэтому качать реальные чекпоинты незачем:
    student.pretrained=False и checkpoint_path=None у обоих слотов дают ту
    же архитектуру без сети. Лучшая попытка — при любой ошибке просто
    пропускаем таблицу, это не повод срывать заливку скаляров."""
    from hydra.utils import instantiate

    from src.utils.metrics import build_param_table

    try:
        student_cfg = OmegaConf.merge(cfg.model.student, {"pretrained": False, "checkpoint_path": None})
    except Exception:
        student_cfg = cfg.model.student
    try:
        student = instantiate(student_cfg)
    except Exception as error:
        print(f"  [param_table] студент не собрался ({error}) — пропускаю таблицу", file=sys.stderr)
        return None

    teacher = None
    if cfg.model.get("teacher") is not None:
        try:
            teacher = instantiate(cfg.model.teacher)
        except Exception as error:
            print(f"  [param_table] учитель не собрался ({error}) — таблица без него", file=sys.stderr)

    criterion = None
    try:
        criterion = instantiate(cfg.loss)
    except Exception as error:
        print(f"  [param_table] лосс не собрался ({error}) — таблица без него", file=sys.stderr)

    return build_param_table(student, teacher, criterion)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="Путь до outputs/<name>/<run> или голое имя эксперимента")
    parser.add_argument(
        "--fresh-task", action="store_true",
        help="Не переиспользовать существующую таску с тем же именем (по умолчанию — переиспользуется, "
             "reuse_last_task_id=True: у прогона, упавшего на квоте, таска в ClearML уже создана, пустая)",
    )
    parser.add_argument("--skip-param-table", action="store_true", help="Не собирать модели ради таблицы параметров")
    parser.add_argument("--dry-run", action="store_true", help="Ничего не отправлять, только проверить и показать, что ушло бы")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    rows = load_history(run_dir)

    if cfg.task_type not in ("classification", "detection", "segmentation"):
        raise ValueError(f"Неизвестный task_type={cfg.task_type!r}")
    best_metric_key = {"classification": "best_acc", "detection": "best_map", "segmentation": "best_miou"}[cfg.task_type]
    eval_metric_column = {"classification": "eval_acc", "detection": "eval_map", "segmentation": "eval_miou"}[cfg.task_type]

    best_value, best_epoch = best_metric_from_history(rows, eval_metric_column)
    world_size = int(float(rows[-1].get("world_size", 1) or 1))

    print(f"run:          {run_dir}")
    print(f"name:         {cfg.name}")
    print(f"project:      {cfg.clearml.project}")
    print(f"tags:         {list(cfg.clearml.tags)}")
    print(f"эпох в CSV:   {len(rows)} (последняя: {rows[-1]['epoch']})")
    print(f"{best_metric_key}:    {best_value:.4f} (эпоха {best_epoch})")
    print(f"world_size:   {world_size}")

    if args.dry_run:
        print("\n--dry-run: в ClearML ничего не отправлено.")
        return

    if not cfg.clearml.enabled:
        raise ValueError(
            f"В сохранённом конфиге clearml.enabled=false — похоже, ClearML в этом прогоне "
            f"и не должен был использоваться. Если это ошибка, поправьте {run_dir}/.hydra/config.yaml вручную."
        )

    from clearml import Task

    from scripts.train import BEST_METRIC_KEY, _plain, clearml_reporter

    assert BEST_METRIC_KEY[cfg.task_type] == best_metric_key  # обе таблицы должны совпадать

    task = Task.init(
        project_name=cfg.clearml.project,
        task_name=cfg.name,
        task_type=Task.TaskTypes.training,
        tags=_plain(cfg.clearml.tags),
        reuse_last_task_id=not args.fresh_task,
        continue_last_task=False,
        output_uri=cfg.clearml.output_uri,
        auto_connect_frameworks=_plain(cfg.clearml.auto_connect_frameworks),
        auto_connect_arg_parser=False,
        # Нет живого GPU/CPU, который стоило бы мониторить — это разовая
        # заливка истории, а не обучение. Жёстко False (а не из cfg): у
        # прогонов до этого ключа его в сохранённом .hydra/config.yaml нет.
        auto_resource_monitoring=False,
    )
    task.connect_configuration(OmegaConf.to_container(cfg, resolve=True), name="hydra_config")
    print(f"\nТаска: {task.id} -> {task.get_output_log_web_page()}")

    report_scalar, report_single, report_table, _report_plots = clearml_reporter(task)

    for row in rows:
        report_scalar(row)
    print(f"Залито {len(rows)} эпох скаляров.")

    if not args.skip_param_table:
        table = build_param_table_safely(cfg)
        if table is not None:
            report_table(table)
            print("Таблица параметров залита.")

    report_single({best_metric_key: best_value, "best_epoch": best_epoch, "world_size": world_size})
    task.close()
    print("Готово. Раздел Plots (per-class IoU/confusion matrix/debug-картинки/KL) не восстановлен — "
          "его сырых данных нет на диске, см. докстринг скрипта.")


if __name__ == "__main__":
    main()
