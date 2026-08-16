"""Описания графиков (не скаляров) для внешнего трекера.

Зачем отдельный слой: тренер по контракту ничего не знает про ClearML —
он отдаёт наружу ДАННЫЕ графика, а перевод в вызов `Logger.report_*` живёт
в scripts/train.py. Заодно это позволяет строить те же графики в ноутбуках
и тестировать их без сети.

Все построители возвращают None, если строить нечего (нет данных, пустая
выборка) — вызывающему коду не нужно проверять это самому.
"""

from typing import NamedTuple, Sequence

import numpy as np


class Plot(NamedTuple):
    """Один график.

    kind определяет, каким вызовом трекера он рисуется:
    - "bar"     — столбики по категориям (per-class IoU и т.п.);
    - "matrix"  — тепловая карта (матрица ошибок);
    - "image"   — готовая картинка uint8 [H, W, 3] (Debug Samples).
    В ClearML это report_histogram / report_confusion_matrix / report_image.
    """

    kind: str
    title: str
    series: str
    values: np.ndarray
    xlabels: list[str] | None = None
    ylabels: list[str] | None = None
    xaxis: str | None = None
    yaxis: str | None = None


def bar_plot(
    title: str,
    series: str,
    values: Sequence[float] | np.ndarray,
    names: Sequence[str] | None = None,
    xaxis: str | None = None,
    yaxis: str | None = None,
) -> Plot | None:
    """Столбчатая диаграмма по категориям. NaN-категории выбрасываются.

    NaN здесь означает "класс не встретился" (нет ни пикселей разметки, ни
    предсказаний) — рисовать для него нулевой столбик было бы враньём:
    ноль читается как "модель полностью провалила класс".
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None

    if names is None:
        names = [str(i) for i in range(values.size)]
    names = list(names)[: values.size]

    finite = np.isfinite(values)
    if not finite.any():
        return None

    return Plot(
        kind="bar",
        title=title,
        series=series,
        values=values[finite],
        xlabels=[names[i] for i in np.flatnonzero(finite)],
        xaxis=xaxis,
        yaxis=yaxis,
    )


def distribution_plot(
    title: str,
    series: str,
    samples: Sequence[float] | np.ndarray,
    bins: int = 40,
    xaxis: str | None = None,
    yaxis: str = "count",
) -> Plot | None:
    """Гистограмма распределения: считает частоты сама, трекер получает готовые.

    ClearML свою гистограмму не строит (report_histogram рисует ровно то, что
    ему передали), поэтому биннинг — здесь. Ось X подписывается серединами
    корзин: так график читается без отдельной легенды.
    """
    samples = np.asarray(samples, dtype=np.float64).ravel()
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return None

    # Корзин не больше, чем самих значений: иначе короткая эпоха (или смоук-тест
    # на десяток шагов) даёт частокол из пустых столбиков.
    counts, edges = np.histogram(samples, bins=max(1, min(bins, samples.size)))
    centers = (edges[:-1] + edges[1:]) / 2.0

    return Plot(
        kind="bar",
        title=title,
        series=series,
        values=counts.astype(np.float64),
        xlabels=[f"{center:.3g}" for center in centers],
        xaxis=xaxis,
        yaxis=yaxis,
    )


def image_plot(title: str, series: str, image: np.ndarray) -> Plot | None:
    """Готовая картинка в Debug Samples. Ожидается uint8 [H, W, 3]."""
    if image is None or image.size == 0:
        return None
    return Plot(kind="image", title=title, series=series, values=image)


def matrix_plot(
    title: str,
    series: str,
    matrix: np.ndarray,
    names: Sequence[str] | None = None,
    xaxis: str | None = None,
    yaxis: str | None = None,
) -> Plot | None:
    """Тепловая карта (матрица ошибок)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 0:
        return None

    labels = None if names is None else list(names)[: matrix.shape[0]]

    return Plot(
        kind="matrix",
        title=title,
        series=series,
        values=matrix,
        xlabels=labels,
        ylabels=labels,
        xaxis=xaxis,
        yaxis=yaxis,
    )
