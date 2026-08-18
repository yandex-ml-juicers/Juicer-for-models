from collections import defaultdict

import torch
from torch import nn

# Головы классификации — единственное, что не переносится с COCO-претрейна
# (ckpt: 80 классов, модель: 8), они инициализированы случайно. Держать их на
# том же lr, что и предобученные части, нет причин — отсюда отдельный lr_head.
HEAD_MARKERS = ("class_embed", "enc_score_head", "denoising_class_embed")


def _backbone_stage_id(name: str, num_stages: int) -> int:
    """Номер стадии ResNet-бэкбона для послойного затухания lr.

    0 — стем (embedder), 1..num_stages — стадии encoder.stages.N,
    num_stages + 1 — всё остальное внутри бэкбона.

    Имена приходят из HF-реализации RTDetr:
        model.backbone.model.embedder.embedder.0.convolution.weight
        model.backbone.model.encoder.stages.2.layers.0.conv1.convolution.weight
    """
    if ".embedder." in name:
        return 0
    if ".stages." in name:
        return int(name.split(".stages.")[1].split(".")[0]) + 1
    return num_stages + 1


def rtdetr_adamw(
    params,
    lr: float = 1e-4,
    lr_backbone: float = 1e-5,
    lr_encoder: float = 1e-4,
    lr_head: float = 1e-4,
    lr_backbone_stage_decay: float = 1.0,
    lr_component_decay: float = 1.0,
    weight_decay: float = 1e-4,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    fused: bool = False,
    num_backbone_stages: int = 4,
):
    """AdamW с раздельным lr по частям RT-DETR и без weight_decay на одномерных.

    Прежняя версия (no_decay_adamw) давала единый lr на всю модель, включая
    COCO-предобученный ResNet50-vd — 23.47M из 42.74M параметров. На дообучении
    Cityscapes (2975 картинок) это ломало предобученные признаки: лучший eval
    mAP приходился на 4-ю эпоху, дальше метрика падала при растущем train mAP.
    Официальный рецепт RT-DETR учит бэкбон на 1e-5 при базовом 1e-4, здесь тот
    же принцип, что и в lwdetr_adamw: каждая часть со своим lr.

    Разбиение по именам параметров HF-реализации RTDetrForObjectDetection:
        model.backbone.*        23.47M  бэкбон, lr_backbone (+ затухание по стадиям)
        model.encoder*          11.94M  гибридный энкодер и его входные проекции
        model.decoder*           6.98M  декодер, bbox-головы, входные проекции
        остальное                0.35M  enc_output и прочее, базовый lr
    Головы из HEAD_MARKERS проверяются первыми: class_embed лежит внутри
    model.decoder, и без этой проверки случайно инициализированная голова
    уехала бы в группу декодера.

    Затухания по умолчанию выключены (1.0) — это ручки под эксперименты, а не
    скрытое отклонение от официального рецепта.
    """
    # Параметры с одинаковой парой (lr, weight_decay) собираются в ОДНУ группу:
    # группа на каждый параметр обесценивает fused-ядро AdamW (см. lwdetr_adamw).
    buckets: dict[tuple[float, float], list[nn.Parameter]] = defaultdict(list)

    for name, param in params:
        if not param.requires_grad:
            continue

        if any(marker in name for marker in HEAD_MARKERS):
            param_lr = lr_head
        elif name.startswith("model.backbone"):
            stage_id = _backbone_stage_id(name, num_backbone_stages)
            param_lr = lr_backbone * lr_backbone_stage_decay ** (num_backbone_stages + 1 - stage_id)
        elif name.startswith("model.encoder"):
            param_lr = lr_encoder
        elif name.startswith("model.decoder"):
            param_lr = lr * lr_component_decay
        else:
            param_lr = lr

        # Как и раньше: декей на bias/нормализациях не регуляризует, а мешает
        # оптимизации — общепринятая практика для трансформеров и CNN с нормами.
        param_wd = 0.0 if param.ndim <= 1 else weight_decay
        buckets[(param_lr, param_wd)].append(param)

    # Сортировка по убыванию lr: тренер логирует param_groups[0]["lr"], там
    # должен оказаться осмысленный максимум, а не случайная группа.
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
