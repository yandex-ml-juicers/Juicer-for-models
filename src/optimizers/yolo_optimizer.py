import torch
from torch import nn


def yolo_sgd(
    params,
    lr: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 5.0e-4,
    nesterov: bool = True,
):
    """SGD по рецепту ultralytics: weight decay только на многомерные веса.

    В YOLOv8 одномерные тензоры — это bias свёрток и gamma/beta BatchNorm.
    Декей на них смещает статистику нормализации и на моделях размера n стоит
    заметной части качества, поэтому ultralytics держит их отдельной группой
    с weight_decay=0.

    Обе группы идут с одним lr: тренер логирует param_groups[0]["lr"], и это
    должно быть осмысленное значение, а не lr случайной группы.

    На вход принимается либо плоский набор параметров, либо уже собранные
    группы-словари: scripts/train.py передаёт именно группы, когда у лосса
    есть feature_adapter — ему weight_decay отключают отдельно. Каждая
    входная группа режется на многомерную и одномерную половины, и заданный
    в группе weight_decay имеет приоритет над общим, иначе развязка адаптера
    потерялась бы при переходе на этот оптимизатор.
    """
    incoming = list(params)
    if incoming and isinstance(incoming[0], dict):
        groups = [dict(group) for group in incoming]
    else:
        groups = [{"params": incoming}]

    param_groups: list[dict] = []
    for group in groups:
        group_weight_decay = group.get("weight_decay", weight_decay)
        extra = {
            key: value for key, value in group.items()
            if key not in ("params", "weight_decay")
        }

        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []

        for param in group["params"]:
            if not param.requires_grad:
                continue

            (no_decay if param.ndim <= 1 else decay).append(param)

        if decay:
            param_groups.append(
                {"params": decay, "weight_decay": group_weight_decay, **extra}
            )
        if no_decay:
            param_groups.append({"params": no_decay, "weight_decay": 0.0, **extra})

    return torch.optim.SGD(
        param_groups,
        lr=lr,
        momentum=momentum,
        nesterov=nesterov,
    )
