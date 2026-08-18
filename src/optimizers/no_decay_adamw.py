import torch


def no_decay_adamw(
    params,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    fused: bool = False,
):
    """AdamW с исключением weight_decay для одномерных параметров (bias,
    LayerNorm/BatchNorm scale) — общепринятая практика для трансформеров и
    CNN с нормализацией, decay на них не регуляризует, а портит оптимизацию.

    В отличие от lwdetr_adamw (src/optimizers/lwdetr_optimizer.py) здесь нет
    ни привязки к ViT-именам слоёв бэкбона, ни послойного затухания lr —
    architecture-agnostic, единый lr на все параметры. Годится для любой
    модели, где нужен только no-decay-сплит без остальных ViT-рецептов
    LW-DETR (изначально добавлен для RT-DETR — ResNet-бэкбон, где
    lr_vit_layer_decay/num_vit_layers из lwdetr_adamw были бы мёртвым кодом).
    """
    decay_params = []
    no_decay_params = []

    for name, param in params:
        if not param.requires_grad:
            continue
        if param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        fused=fused,
    )
