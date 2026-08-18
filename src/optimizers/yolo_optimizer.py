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

    params принимает и плоский список Parameter (обычный finetune), и список
    уже готовых param-group словарей — например, {"params": adapter_params,
    "weight_decay": 0.0} из scripts/train.py для feature_adapter в CLoCKDistill/
    DCKD (там weight_decay намеренно обнулён отдельно, чтобы декей не убивал
    веса адаптера). Группы с явным weight_decay сохраняются как есть, группы
    без него (и голые Parameter) идут в обычный decay/no-decay сплит.

    requires_grad НЕ фильтруется: trainer.py сверяет, что все параметры
    criterion.parameters() попали в какую-то группу оптимизатора (например,
    у CLoCKDistillLoss content_embed — намеренно замороженный, requires_grad=
    False, но зарегистрированный Parameter). SGD.step() сам безопасно
    пропускает параметры без градиента, так что включать их сюда не вредно.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    extra_groups: list[dict] = []

    for item in params:
        if isinstance(item, dict):
            if "weight_decay" in item:
                if item["params"]:
                    extra_groups.append(item)
                continue
            iterable = item["params"]
        else:
            iterable = [item]

        for param in iterable:
            (no_decay if param.ndim <= 1 else decay).append(param)

    param_groups = [
        group
        for group in (
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        )
        if group["params"]
    ] + extra_groups

    return torch.optim.SGD(
        param_groups,
        lr=lr,
        momentum=momentum,
        nesterov=nesterov,
    )
