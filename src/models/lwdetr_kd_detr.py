"""Обёртка LW-DETR, добавляющая "general distillation points" из KD-DETR
(Wang et al., CVPR 2024, https://arxiv.org/abs/2211.08071) — используется
только учителем в src/losses/kd_detr_loss.py, собирается через
src.models.factory.lwdetr_small_for_detection(num_probe_points=...).
"""

import torch
from torch import nn

from transformers.models.lw_detr.modeling_lw_detr import refine_bboxes


class LwDetrKDDETRProbes(nn.Module):
    """LW-DETR + "general distillation points" из KD-DETR.

    В обычном forward LW-DETR reference_points decoder'а — top-k предложения
    encoder'а под РЕАЛЬНЫЕ detection queries, у KD-DETR нет над ними
    контроля. Статья вводит отдельный, не участвующий в детекции набор
    "specialized object queries": координатно заданные probe-точки, общие по
    конструкции для учителя и студента, — потому что real queries учителя и
    студента "egocentric" и напрямую не сопоставимы. Здесь эта идея
    реализована так: probe-точки сэмплируются из anchor-сетки encoder'а
    (LwDetrModel.gen_encoder_output_proposals — тот же grid, из которого
    LW-DETR берёт top-k под реальные queries), decoder прогоняется по ним
    ВТОРОЙ раз, и результат (probe_logits/probe_boxes) кладётся на выход
    обычными полями. Из этой же сетки берётся (w,h) — по построению
    осмысленный масштаб для уровня FPN, а не произвольная константа.
    Дальнейшее сопоставление с dense-предсказаниями YOLO — в
    src/losses/kd_detr_loss.py.

    Не переиспользует LwDetrModel.forward целиком: дублирует только
    backbone- и flatten-часть (см. modeling_lw_detr.LwDetrModel.forward,
    пришпилено к transformers==5.15.0 в venv), а сам decode — через реальные
    submodules учителя (self.model.model.decoder, self.model.class_embed,
    self.model.bbox_embed, refine_bboxes библиотеки). Если их сигнатуры
    разъедутся при обновлении transformers, упадёт здесь с понятной
    ошибкой, а не молча даст неверные числа.

    Контент probe-запросов (inputs_embeds decoder'а) — нули: в отличие от
    реальных queries, у probe-точек нет обученного query_feat, а позиционный
    сигнал в decoder всё равно приходит отдельно, через query_pos
    (LwDetrDecoder.get_reference), на каждом слое.
    """

    def __init__(self, model: nn.Module, num_probe_points: int) -> None:
        super().__init__()
        if num_probe_points <= 0:
            raise ValueError("num_probe_points должен быть > 0")
        self.model = model
        self.num_probe_points = num_probe_points

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

            _, output_proposals, invalid_mask = base.gen_encoder_output_proposals(
                source_flatten, ~mask_flatten, spatial_shapes_list
            )
            valid = ~invalid_mask.squeeze(-1)

            probe_points = self._sample_probe_points(
                output_proposals,
                valid,
                num_points=self.num_probe_points,
                batch_size=batch_size,
                device=device,
            )

            probe_content = torch.zeros(
                batch_size, probe_points.shape[1], base.config.d_model,
                device=device, dtype=source_flatten.dtype,
            )

            decoder_outputs = base.decoder(
                inputs_embeds=probe_content,
                reference_points=probe_points,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                encoder_hidden_states=source_flatten,
                encoder_attention_mask=mask_flatten,
            )

            probe_logits = self.model.class_embed(decoder_outputs.last_hidden_state)
            probe_boxes_delta = self.model.bbox_embed(decoder_outputs.last_hidden_state)
            probe_boxes = refine_bboxes(
                decoder_outputs.intermediate_reference_points[-1], probe_boxes_delta
            )

        # Обычные поля на выходе (ModelOutput.__setattr__ это поддерживает,
        # см. transformers.utils.generic.ModelOutput) — тот же приём, что
        # используется для teacher_outputs.projected_feature в DCKD.
        outputs.probe_logits = probe_logits
        outputs.probe_boxes = probe_boxes

        return outputs

    @staticmethod
    def _sample_probe_points(
        output_proposals: torch.Tensor,
        valid: torch.Tensor,
        *,
        num_points: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        total = output_proposals.shape[1]
        selected: list[torch.Tensor] = []

        for b in range(batch_size):
            valid_idx = torch.nonzero(valid[b], as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                # Не должно происходить при нормальном input_size, но лучше
                # деградировать до полной (невалидированной) сетки, чем упасть.
                valid_idx = torch.arange(total, device=device)

            if valid_idx.numel() >= num_points:
                perm = torch.randperm(valid_idx.numel(), device=device)[:num_points]
                idx = valid_idx[perm]
            else:
                # Валидных ячеек меньше, чем нужно точек — берём с повторами.
                idx = valid_idx[torch.randint(0, valid_idx.numel(), (num_points,), device=device)]

            selected.append(idx)

        index = torch.stack(selected, dim=0)  # [B, num_points]
        return torch.gather(output_proposals, 1, index.unsqueeze(-1).expand(-1, -1, 4))
