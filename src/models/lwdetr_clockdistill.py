"""LW-DETR + доступ к decoder'у и его геометрии для CLoCKDistill
(Ge et al., 2025, https://arxiv.org/abs/2502.10683) — используется только
учителем в src/losses/clockdistill_loss.py, собирается через
src.models.factory.lwdetr_small_for_detection(clockdistill=True).

Что делает статья
------------------
Два компонента: LCMD (Location-and-Context-aware Memory Distillation) —
дистилляция encoder memory с масками по GT, и TCLD (Target-aware
Consistent Logit Distillation) — decoder прогоняется ВТОРОЙ раз по
"target-aware queries", построенным из GT (content query = Embed_c(class),
positional query = MLP(box)), результат матчится между учителем и
студентом с весом по уверенности учителя. Обе части описаны в
src/losses/clockdistill_loss.py — этот файл только даёт лоссу доступ к
внутренностям учителя.

LW-DETR (в отличие от ванильного Deformable-DETR) не имеет отдельного
self-attention encoder: то, что статья называет "memory", — это прямо
flatten-выход backbone-проектора (source_flatten), тот же тензор, что уже
снимается хуком model.backbone.projector для DCKD (LCMD использует его же
через FeatureExtractor, отдельный код тут не нужен). А позиционная часть
target-aware query (q_pos = MLP(box)) уже считается самим decoder'ом из
reference_points через его собственный обученный LwDetrDecoder.ref_point_head
(см. LwDetrDecoder.get_reference) — поэтому этому классу достаточно
пробросить наружу сам decoder и геометрию его входа (spatial_shapes,
valid_ratios, level_start_index, mask), а не переизобретать MLP.

Дублирует backbone- и flatten-часть LwDetrModel.forward, как и
LwDetrKDDETRProbes (src/models/lwdetr_kd_detr.py) — тот же приём и та же
цена (лишний backbone-проход на каждый teacher forward), только вместо
готовых probe_logits/probe_boxes наружу отдаются "сырые материалы" для
decoder-запроса, потому что GT (нужный для target-aware query) до модели
на этом шаге ещё не доходит — Trainer вызывает self.teacher(images) без
меток (см. src/training/trainer.py). Второй decoder-проход поэтому
выполняется не здесь, а в CLoCKDistillLoss.forward, у которого labels уже
есть.
"""

import torch
from torch import nn


class LwDetrCLoCKDistillMemory(nn.Module):
    """LW-DETR, наружу дополнительно отдающий decoder и геометрию его входа.

    Обычный forward LW-DETR (детекции, logits, pred_boxes) не меняется.
    Дополнительные поля на outputs (все — под именем clockdistill_*):
        clockdistill_memory:              [B, N, D]  flatten backbone-фичи
        clockdistill_memory_mask:         [B, N]     padding-маска
        clockdistill_spatial_shapes:      [L, 2]
        clockdistill_spatial_shapes_list: list[(H, W)] длины L
        clockdistill_level_start_index:   [L]
        clockdistill_valid_ratios:        [B, L, 2]
        clockdistill_decoder:             nn.Module (base.decoder)
        clockdistill_class_embed:         nn.Module (self.model.class_embed)
        clockdistill_bbox_embed:          nn.Module (self.model.bbox_embed)

    decoder/class_embed/bbox_embed — живые ссылки на подмодули учителя, не
    тензоры: их параметры заморожены Trainer'ом (self.teacher.requires_grad_
    (False)), но сам модуль остаётся вызываемым. CLoCKDistillLoss.forward
    прогоняет его второй раз под torch.no_grad() (target-aware query не
    обучается — см. докстринг класса выше), поэтому эта заморозка ничему не
    мешает.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: torch.Tensor | None = None,
        **kwargs: any,
    ):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, **kwargs)

        base = self.model.model  # LwDetrModel
        batch_size, _, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), dtype=torch.long, device=device)

        with torch.no_grad():
            features = base.backbone(pixel_values, pixel_mask)

            source_flatten_parts, mask_flatten_parts, spatial_shapes_list = [], [], []
            for source, mask in features:
                if mask is None:
                    raise ValueError("LW-DETR backbone не вернул attention mask")
                spatial_shapes_list.append((source.shape[-2], source.shape[-1]))
                source_flatten_parts.append(source.flatten(2).transpose(1, 2))
                mask_flatten_parts.append(mask.flatten(1))

            source_flatten = torch.cat(source_flatten_parts, 1)
            mask_flatten = torch.cat(mask_flatten_parts, 1)
            spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=device)
            level_start_index = torch.cat(
                (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
            )
            valid_ratios = torch.stack(
                [base.get_valid_ratio(mask, dtype=source_flatten.dtype) for _, mask in features], 1
            )

        outputs.clockdistill_memory = source_flatten
        outputs.clockdistill_memory_mask = mask_flatten
        outputs.clockdistill_spatial_shapes = spatial_shapes
        outputs.clockdistill_spatial_shapes_list = spatial_shapes_list
        outputs.clockdistill_level_start_index = level_start_index
        outputs.clockdistill_valid_ratios = valid_ratios
        outputs.clockdistill_decoder = base.decoder
        outputs.clockdistill_class_embed = self.model.class_embed
        outputs.clockdistill_bbox_embed = self.model.bbox_embed

        return outputs
