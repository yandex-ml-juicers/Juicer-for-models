from collections import defaultdict

import torch
from torch import nn

# Параметры, которые не регуляризуются весовым декеем: всё одномерное
# (bias, LayerNorm, LayerScale gamma) и таблицы эмбеддингов.
# Стандарт рецептов ViT/DETR: декей на них ухудшает дообучение.
NO_DECAY_SUFFIXES = (
    "position_embeddings",
    "query_feat.weight",
    "reference_point_embed.weight",
)


def _vit_layer_id(name: str, num_vit_layers: int) -> int:
    """Номер блока ViT для послойного затухания LR.

    0 — патч-эмбеддинги, 1..num_vit_layers — блоки трансформера,
    num_vit_layers + 1 — всё остальное внутри бэкбона (нормы после последнего
    блока), оно учится с полным lr_encoder.

    Имена приходят из HF-реализации LwDetr:
        model.backbone.backbone.embeddings.projection.weight
        model.backbone.backbone.encoder.layer.7.attention.q_proj.weight
    Блоки называются ".layer.", а не ".layers." — прежний код искал только
    ".layers." и поэтому не находил ни одного слоя.
    """
    if ".embeddings." in name:
        return 0

    for marker in (".layer.", ".layers."):
        if marker in name:
            return int(name.split(marker)[1].split(".")[0]) + 1

    return num_vit_layers + 1


def lwdetr_adamw(
    params,
    lr: float = 1e-4,
    lr_encoder: float = 1.5e-4,
    lr_vit_layer_decay: float = 0.8,
    lr_component_decay: float = 0.7,
    weight_decay: float = 1e-4,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    fused: bool = False,
    num_vit_layers: int = 10,
):
    # Параметры с одинаковой парой (lr, weight_decay) собираются в ОДНУ группу.
    # С группой на каждый параметр (было 449 групп) fused-ядро AdamW
    # запускается на каждый тензор отдельно и весь смысл fused теряется.
    buckets: dict[tuple[float, float], list[nn.Parameter]] = defaultdict(list)

    for name, param in params:
        if not param.requires_grad:
            continue

        # ViT-бэкбон: свой lr и затухание вглубь — нижние блоки предобучены
        # лучше всего, и трогать их надо осторожнее верхних.
        if "backbone" in name and (".encoder." in name or ".embeddings." in name):
            layer_id = _vit_layer_id(name, num_vit_layers)
            param_lr = lr_encoder * lr_vit_layer_decay ** (num_vit_layers + 1 - layer_id)
        elif ".decoder." in name:
            param_lr = lr * lr_component_decay
        else:
            param_lr = lr

        param_wd = 0.0 if param.ndim <= 1 or name.endswith(NO_DECAY_SUFFIXES) else weight_decay
        buckets[(param_lr, param_wd)].append(param)

    # Сортировка по убыванию lr: тренер логирует param_groups[0]["lr"],
    # и это должен быть осмысленный максимум, а не случайная группа.
    param_groups = [
        {"params": group_params, "lr": group_lr, "weight_decay": group_wd}
        for (group_lr, group_wd), group_params in sorted(buckets.items(), key=lambda item: -item[0][0])
    ]

    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        fused=fused,
    )
