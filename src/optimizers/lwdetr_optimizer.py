from collections.abc import Iterable

import torch
from torch import nn


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
    param_groups = []

    for name, param in params:
        if not param.requires_grad:
            continue

        # ViT encoder
        if ".encoder." in name:
            layer_id = num_vit_layers + 1

            if "patch_embeddings" in name:
                layer_id = 0

            elif ".layers." in name:
                parts = name.split(".layers.")
                layer_id = int(parts[1].split(".")[0]) + 1

            lr_scale = lr_vit_layer_decay ** (num_vit_layers + 1 - layer_id)
            param_lr = lr_encoder * lr_scale

        # Decoder
        elif ".decoder." in name:
            param_lr = lr * lr_component_decay
        # Остальные компоненты
        else:
            param_lr = lr

        param_groups.append({"params": [param], "lr": param_lr})

    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        fused=fused,
    )