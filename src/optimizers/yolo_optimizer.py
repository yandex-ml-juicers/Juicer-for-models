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
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []

    for param in params:
        if not param.requires_grad:
            continue

        (no_decay if param.ndim <= 1 else decay).append(param)

    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    return torch.optim.SGD(
        param_groups,
        lr=lr,
        momentum=momentum,
        nesterov=nesterov,
    )
