"""Примитивное логирование: stdout + history.csv в директорию запуска.

Намеренно без внешних трекеров (W&B/ClearML/MLflow): интеграция трекера —
отдельная зона ответственности. Точка стыковки — MetricsHistory.append():
трекер надо будет позвать ровно в этом месте.
"""

import csv
import logging
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """Логгер модуля. Под Hydra хендлеры уже настроены (консоль + файл в run-dir)."""
    return logging.getLogger(name)


class MetricsHistory:
    """Накапливает по строке метрик на эпоху и переписывает CSV после каждой.

    Перезапись целиком (а не append) делает файл валидным даже при падении
    посреди обучения и позволяет столбцам появляться в любой момент.
    """

    def __init__(self, path: Path, resume: bool = False) -> None:
        self.path = Path(path)
        self.rows: list[dict] = []
        # При resume в тот же output_dir файл уже содержит эпохи до крэша —
        # без подгрузки первый же _flush() переписал бы CSV только новыми
        # строками и стёр историю.
        if resume and self.path.exists():
            with self.path.open(newline="") as f:
                self.rows = list(csv.DictReader(f))

    def append(self, row: dict) -> None:
        self.rows.append(dict(row))
        self._flush()

    def _flush(self) -> None:
        fieldnames: list[str] = []
        for row in self.rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
