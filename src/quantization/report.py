"""Отчёт PTQ-запуска: report.json и таблицы рядом с ним"""

import json
import logging
from pathlib import Path
from typing import Any

from src.utils.logger import MetricsHistory

log = logging.getLogger(__name__)

TARGET_METRIC = {"classification": "accuracy", "segmentation": "miou"}

class QuantizationReport:
    def __init__(self, output_dir: str | Path, name: str = "report.json") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / name
        self.payload: dict[str, Any] = {"stages": {}}

    def context(self, **fields) -> None:
        self.payload.update(fields)
        self.flush()

    def stage(self, name: str, payload: dict | None = None, *, status: str = "ok") -> None:
        self.payload["stages"][name] = {"status": status, **(payload or {})}
        self.flush()

    def table(self, filename: str, rows: list[dict]) -> Path:
        """Таблица в CSV рядом с отчётом; в JSON остаётся только ссылка."""
        path = self.output_dir / filename
        history = MetricsHistory(path)
        for row in rows:
            history.append(row)
        self.payload.setdefault("tables", {})[filename] = len(rows)
        self.flush()
        return path

    def flush(self) -> None:
        self.path.write_text(json.dumps(self.payload, indent=2, ensure_ascii=False, default=str))


# годен ли сквантованный движок к использованию
def check_acceptance(
    baseline: dict,
    candidate: dict,
    comparison: dict,
    *,
    task_type: str = "classification",
    max_metric_drop: float = 0.005,
    min_agreement: float = 0.99,
) -> dict:
    metric_key = TARGET_METRIC.get(task_type)
    if metric_key is None:
        raise ValueError(f"Неизвестная целевая метрика для task_type={task_type!r}.")

    drop = baseline[metric_key] - candidate[metric_key]
    agreement = comparison.get("argmax_agreement", 0.0)

    violations = []
    if drop > max_metric_drop:
        violations.append(
            f"{metric_key} просела на {drop:.4f} при допуске {max_metric_drop:.4f} "
            f"({baseline[metric_key]:.4f} -> {candidate[metric_key]:.4f})"
        )
    if agreement < min_agreement:
        violations.append(
            f"совпадение предсказаний с fp32 {agreement:.4f} ниже порога {min_agreement:.4f}"
        )

    verdict = {
        "passed": not violations,
        "metric": metric_key,
        "baseline": baseline[metric_key],
        "candidate": candidate[metric_key],
        "drop": drop,
        "max_metric_drop": max_metric_drop,
        "argmax_agreement": agreement,
        "min_agreement": min_agreement,
        "violations": violations,
    }

    if violations:
        log.error("Квантизация не прошла приёмку: %s", "; ".join(violations))
    else:
        log.info(
            "Приёмка пройдена: %s %.4f -> %.4f (потеря %.4f), совпадение %.4f",
            metric_key,
            baseline[metric_key],
            candidate[metric_key],
            drop,
            agreement,
        )
    return verdict
