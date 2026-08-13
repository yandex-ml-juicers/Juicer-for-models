"""Группы параметров оптимизатора: свой learning rate и weight decay
энкодеру, декодеру и обучаемым частям лосса.

Зачем это нужно. У сегментатора с предобученным энкодером (TimmUNet,
SegNeXt, SegFormer) две части учатся из принципиально разных начальных
условий: энкодер пришёл с ImageNet и уже умеет извлекать признаки, а
декодер и голова инициализированы случайно. Один общий lr либо слишком
мал для декодера, либо слишком велик для энкодера — во втором случае
предобучение просто затирается первыми же эпохами. Канонический приём —
дать энкодеру долю общего lr (обычно 0.1).

Второе, что здесь решается, — weight decay на параметрах, которым он
противопоказан. AdamW с wd=3e-2 (как в конфигах проекта) применяется в том
числе к весам и сдвигам BatchNorm, то есть систематически подавляет
масштабирующий множитель нормализации. Стандарт везде — исключать из
decay всё одномерное: gamma/beta норм-слоёв и bias свёрток.

Как это соотносится с остальным кодом:

- группы строятся ДО обёртки в DDP, поэтому имена параметров ещё без
  префикса "module." (см. scripts/train.py);
- в первой группе всегда лежит "основной" lr (декодер), потому что
  тренер логирует param_groups[0]["lr"] как lr запуска — порядок групп
  сохраняет смысл уже накопленных графиков;
- параметры лосса (адаптеры FitNets, проекторы HeteroAKD) попадают в
  отдельную группу, но остаются в optimizer: тренер проверяет это в
  конструкторе и падает, если хоть один параметр criterion потерялся.
"""

import logging
from collections.abc import Iterable, Sequence

from torch import nn

log = logging.getLogger(__name__)


# Какие параметры считать энкодером — по префиксу имени в state_dict.
# Ключ — имя класса модели (проходим по всему MRO, поэтому наследники
# работают без отдельной записи).
#
# U-Net: боттлнек относится к энкодеру, а не к декодеру. Он завершает
# нисходящий путь и в timm-версии физически является последней стадией
# бэкбона — логичнее регулировать его вместе с ней.
ENCODER_PREFIXES: dict[str, tuple[str, ...]] = {
    "TimmUNet": ("encoder.",),
    "UNet": ("encoders.", "bottleneck."),
    "SegNeXt": ("encoder.",),
    # transformers прячет энкодер MiT внутри SegformerForSemanticSegmentation:
    # model.segformer.* — энкодер, model.decode_head.* — голова.
    "SegFormer": ("model.segformer.",),
}


def resolve_encoder_prefixes(model: nn.Module) -> tuple[str, ...] | None:
    """Префиксы имён параметров энкодера или None, если модель неизвестна."""
    for klass in type(model).__mro__:
        prefixes = ENCODER_PREFIXES.get(klass.__name__)
        if prefixes is not None:
            return prefixes
    return None


def _is_no_decay(parameter: nn.Parameter) -> bool:
    """Одномерные тензоры — это bias и gamma/beta нормализаций.

    Проверка по форме, а не по имени модуля: она не зависит ни от того,
    как названы слои, ни от того, заменил ли SyncBatchNorm обычный BN.
    """
    return parameter.ndim <= 1


def _collect(
    module: nn.Module,
    prefixes: Sequence[str] | None,
    seen: set[int],
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    """Разбор модуля на (энкодер decay/no-decay, остальное decay/no-decay).

    prefixes=None означает "энкодер не выделяем", и всё уходит в "остальное".
    seen защищает от повторов: у моделей со связанными весами один и тот же
    тензор виден под несколькими именами, а в optimizer он должен попасть
    ровно один раз.
    """
    encoder_decay: list[nn.Parameter] = []
    encoder_no_decay: list[nn.Parameter] = []
    other_decay: list[nn.Parameter] = []
    other_no_decay: list[nn.Parameter] = []

    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))

        is_encoder = prefixes is not None and name.startswith(tuple(prefixes))
        if is_encoder:
            target = encoder_no_decay if _is_no_decay(parameter) else encoder_decay
        else:
            target = other_no_decay if _is_no_decay(parameter) else other_decay
        target.append(parameter)

    return encoder_decay, encoder_no_decay, other_decay, other_no_decay


def build_param_groups(
    student: nn.Module,
    criterion: nn.Module | None = None,
    *,
    lr: float,
    weight_decay: float = 0.0,
    encoder_lr_mult: float = 1.0,
    encoder_weight_decay: float | None = None,
    criterion_lr_mult: float = 1.0,
    criterion_weight_decay: float | None = None,
    no_decay_on_norm_and_bias: bool = True,
    encoder_prefixes: Iterable[str] | None = None,
) -> list[dict]:
    """Группы параметров для optimizer.

    Args:
        lr, weight_decay: базовые значения из конфига оптимизатора. Всё
            остальное задаётся МНОЖИТЕЛЯМИ от них, чтобы при переборе lr
            (свипы, CLI-override) соотношение частей не разъезжалось.
        encoder_lr_mult: доля базового lr для энкодера. 1.0 — как раньше,
            0.1 — канонический вариант для предобученного бэкбона.
        encoder_weight_decay: свой wd энкодеру; None — как у остальных.
        criterion_lr_mult, criterion_weight_decay: то же для обучаемых
            параметров лосса. По умолчанию у них выключен decay: адаптер
            FitNets и проекторы HeteroAKD — это часть целевой функции,
            а не модели, и штрафовать их норму незачем.
        no_decay_on_norm_and_bias: выносить одномерные параметры в группы
            с weight_decay=0.
        encoder_prefixes: явные префиксы имён вместо таблицы
            ENCODER_PREFIXES — нужны для моделей, которых в ней нет.

    Returns:
        Список групп; первая — декодер с базовым lr, чтобы
        param_groups[0]["lr"] по-прежнему означал "основной lr запуска".

    Raises:
        ValueError: запрошен отдельный режим для энкодера, а выделить его
            в этой модели не удалось. Молча применить общий lr нельзя:
            прогон выглядел бы настроенным, а был бы обычным.
    """
    if encoder_prefixes is not None:
        prefixes: tuple[str, ...] | None = tuple(encoder_prefixes)
    else:
        prefixes = resolve_encoder_prefixes(student)

    encoder_is_separate = encoder_lr_mult != 1.0 or encoder_weight_decay is not None
    if encoder_is_separate and not prefixes:
        raise ValueError(
            f"Для {type(student).__name__} не задано, какие параметры считать энкодером, "
            f"а конфиг просит отдельный режим (encoder_lr_mult={encoder_lr_mult}). "
            f"Добавьте класс в src.training.param_groups.ENCODER_PREFIXES "
            f"или задайте param_groups.encoder_prefixes явно. "
            f"Модули верхнего уровня: {sorted(name for name, _ in student.named_children())}."
        )
    if not encoder_is_separate:
        # Энкодер не выделяем вовсе: лишняя группа с теми же значениями
        # только засоряет логи.
        prefixes = None

    seen: set[int] = set()
    encoder_decay, encoder_no_decay, other_decay, other_no_decay = _collect(
        student, prefixes, seen
    )

    if not no_decay_on_norm_and_bias:
        # Сливаем обратно: пусть решение "не выносить" будет явным здесь,
        # а не размазанным по вызывающему коду.
        encoder_decay, encoder_no_decay = encoder_decay + encoder_no_decay, []
        other_decay, other_no_decay = other_decay + other_no_decay, []

    encoder_lr = lr * encoder_lr_mult
    encoder_wd = weight_decay if encoder_weight_decay is None else encoder_weight_decay

    # Когда энкодер не выделен, в "остальном" лежит вся модель целиком, и
    # звать эту группу "decoder" было бы враньём — особенно в логах и в
    # сериях графика lr, где имя увидит человек.
    main = "decoder" if prefixes else "model"

    groups: list[dict] = [
        # Порядок важен: основная группа идёт первой (см. докстринг модуля).
        {"name": main, "params": other_decay, "lr": lr, "weight_decay": weight_decay},
        {"name": f"{main}_no_decay", "params": other_no_decay, "lr": lr, "weight_decay": 0.0},
        {"name": "encoder", "params": encoder_decay, "lr": encoder_lr, "weight_decay": encoder_wd},
        {"name": "encoder_no_decay", "params": encoder_no_decay, "lr": encoder_lr, "weight_decay": 0.0},
    ]

    if criterion is not None:
        criterion_decay, criterion_no_decay = [], []
        for parameter in criterion.parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if no_decay_on_norm_and_bias and _is_no_decay(parameter):
                criterion_no_decay.append(parameter)
            else:
                criterion_decay.append(parameter)

        criterion_wd = 0.0 if criterion_weight_decay is None else criterion_weight_decay
        groups += [
            {
                "name": "criterion",
                "params": criterion_decay,
                "lr": lr * criterion_lr_mult,
                "weight_decay": criterion_wd,
            },
            {
                "name": "criterion_no_decay",
                "params": criterion_no_decay,
                "lr": lr * criterion_lr_mult,
                "weight_decay": 0.0,
            },
        ]

    return [group for group in groups if group["params"]]


def describe_param_groups(groups: Sequence[dict]) -> str:
    """Таблица групп для лога: по ней видно, что рычаг реально сработал."""
    lines = [f"{'группа':<20}{'тензоров':>10}{'параметров':>14}{'lr':>12}{'weight_decay':>14}"]
    for group in groups:
        count = sum(parameter.numel() for parameter in group["params"])
        lines.append(
            f"{group.get('name', '?'):<20}{len(group['params']):>10}{count:>14,}"
            f"{group['lr']:>12.2e}{group['weight_decay']:>14.4g}"
        )
    return "\n".join(lines)
