from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.lw_detr.modeling_lw_detr import refine_bboxes
from src.losses.base import DistillationLoss
from src.losses.yolo_loss import YOLO


class CLoCKDistillLoss(DistillationLoss):
    """CLoCKDistill (Ge et al., 2025, https://arxiv.org/abs/2502.10683) для
    LW-DETR (учитель) -> YOLOv8/v9 (студент).

    Что делает статья
    ------------------
    Два компонента:

    1. LCMD (Location-and-Context-aware Memory Distillation) — дистилляция
       encoder memory (не backbone-фич!) с двумя масками по GT:
           M_p = 1[p внутри GT-бокса]                       location mask
           S_p = 1/(H_k*W_k) внутри бокса k, иначе 1/N_bg     scale mask
           L_lcmd = alpha*sum(M*S*(A_T-A_S)^2) + beta*sum((1-M)*S*(A_T-A_S)^2)

    2. TCLD (Target-aware Consistent Logit Distillation) — GT-based query
       (q = Embed_c(class) + MLP(box)) прогоняется через decoder ещё раз,
       результат матчится с весом по уверенности учителя:
           L_tcld = sum(w * [lambda_cls*KL + lambda_l1*L1 + lambda_giou*GIoU])
           w = max_c sigmoid(teacher_cls)

    Что адаптировано под YOLOv8/v9
    --------------------------------
    - Memory: у LW-DETR нет отдельного self-attention encoder — "memory" это
      прямо flatten backbone-проектор, тот же тензор, что DCKD уже снимает
      хуком model.model.backbone.projector (см. teacher_feature_key ниже;
      лишний "model." в начале — из-за обёртки LwDetrCLoCKDistillMemory,
      см. её докстринг). Из списка уровней (LwDetrMultiScaleProjector
      отдаёт list по числу FPN-уровней) берётся последний — тот же,
      которому соответствует student_feature_key (P5/stride32), как в DCKD.
    - Positional query TCLD: не переизобретается — LwDetrDecoder сам
      считает q_pos = ref_point_head(sine(reference_points)) внутри
      forward'а (LwDetrDecoder.get_reference), поэтому достаточно передать
      GT-боксы как reference_points; MLP тут — родной, уже обученный слой
      учителя, а не новый.
    - Content query: Embed_c — nn.Embedding(num_classes, d_model),
      requires_grad_(False) сразу после создания: статья прямо говорит
      "queries remain unlearnable throughout distillation". Инициализация —
      случайная (в статье её никто заново не обучает на этом этапе, а
      совместно с учителем с нуля мы его не тренируем), поэтому весь
      TCLD-проход через decoder идёт под torch.no_grad() — как probe-точки
      в KD-DETR (src/losses/kd_detr_loss.py), тем же способом.
    - Матчинг с dense-предсказанием YOLO: тот же приём, что в KD-DETR —
      ближайший anchor того уровня FPN, чей stride ближе всего к масштабу
      объекта (см. _point_indices, дословно оттуда).
    - lambda_cls/l1/giou = 1/5/2, как в статье и как уже в KD-DETR-конфиге
      этого проекта (общий источник — DETR-KD литература).
    - alpha/beta статьи (5e-5/1e-7) посчитаны для полноразмерного
      multi-scale encoder memory (~десятки тысяч токенов). Здесь memory —
      один уровень (та же урезка, что и в LCMD выше), поэтому эти веса,
      скорее всего, потребуют переподбора под конкретный масштаб фич —
      начальная точка, не гарантированно верное значение.
    - Аукс-лоссы по промежуточным decoder-слоям (Σ_e в формуле статьи) не
      реализованы: TCLD считается только по последнему слою decoder'а —
      та же степень упрощения, что уже принята в DCKD/KD-DETR (оба тоже
      используют только финальные query учителя, не auxiliary-выходы).

    Требует учителя, собранного с clockdistill=True (см.
    src.models.factory.lwdetr_small_for_detection).
    """

    requires_teacher = True
    required_features: tuple[str, ...] = ()

    def __init__(
        self,
        num_classes: int,
        student_feature_key: str,
        teacher_feature_key: str,
        student_channels: int,
        teacher_channels: int,
        d_model: int,
        *,
        lambda_det: float = 1.0,
        lambda_lcmd: float = 1.0,
        lambda_tcld: float = 1.0,
        alpha: float = 5e-5,
        beta: float = 1e-7,
        lambda_cls: float = 1.0,
        lambda_l1: float = 5.0,
        lambda_giou: float = 2.0,
        temperature: float = 2.0,
        teacher_has_no_object: bool = False,
        max_target_queries: int = 300,
        det_strides: tuple[int, ...] = (8, 16, 32),
        det_reg_max: int = 16,
        det_box_gain: float = 7.5,
        det_cls_gain: float = 0.5,
        det_dfl_gain: float = 1.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes должен быть > 0")
        if temperature <= 0:
            raise ValueError("temperature должен быть > 0")
        if max_target_queries <= 0:
            raise ValueError("max_target_queries должен быть > 0")

        self.num_classes = num_classes
        self.student_feature_key = student_feature_key
        self.teacher_feature_key = teacher_feature_key
        self.student_required_features = (student_feature_key,)
        self.teacher_required_features = (teacher_feature_key,)
        self.lambda_det = lambda_det
        self.lambda_lcmd = lambda_lcmd
        self.lambda_tcld = lambda_tcld
        self.alpha = alpha
        self.beta = beta
        self.lambda_cls = lambda_cls
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.temperature = temperature
        self.teacher_has_no_object = teacher_has_no_object
        self.max_target_queries = max_target_queries
        self.det_strides = det_strides
        self.eps = eps

        self.task_loss = YOLO(
            num_classes=num_classes,
            strides=det_strides,
            reg_max=det_reg_max,
            box_gain=det_box_gain,
            cls_gain=det_cls_gain,
            dfl_gain=det_dfl_gain,
        )
        self.feature_adapter = (
            nn.Identity() if student_channels == teacher_channels
            else nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False)
        )

        # "queries remain unlearnable throughout distillation" (статья) —
        # см. докстринг класса.
        self.content_embed = nn.Embedding(num_classes, d_model)
        self.content_embed.weight.requires_grad_(False)

    @property
    def gradient_probe_weights(self) -> dict[str, float]:
        # total = lambda_det*det + lambda_lcmd*lcmd + lambda_tcld*tcld
        return {"det": self.lambda_det, "lcmd": self.lambda_lcmd, "tcld": self.lambda_tcld}

    def forward(
        self,
        student_outputs: Any,
        teacher_outputs: Any | None,
        labels: list[dict[str, torch.Tensor]],
        student_features: dict[str, torch.Tensor] | None = None,
        teacher_features: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        s_logits, s_boxes, feature_shapes = self._student_predictions(student_outputs)
        raw_labels = labels
        labels = self._prepare_labels(labels, batch_size=s_logits.shape[0])

        if isinstance(student_outputs, dict):
            student_outputs["kd_logits"] = s_logits
            student_outputs["kd_boxes"] = s_boxes

        det = self.task_loss(student_outputs, teacher_outputs=None, labels=raw_labels)["total"]

        if teacher_outputs is None:
            zero = det.new_zeros(())
            return {"total": self.lambda_det * det, "det": det, "lcmd": zero, "tcld": zero}

        gt_boxes = [self._target_kd_boxes(target).to(device=s_boxes.device, dtype=s_boxes.dtype) for target in labels]
        gt_labels = [target["labels"].to(device=s_boxes.device) for target in labels]

        lcmd = (
            self._lcmd(
                teacher_features=teacher_features,
                student_features=student_features,
                student_outputs=student_outputs,
                gt_boxes=gt_boxes,
            )
            if self.lambda_lcmd != 0.0 else det.new_zeros(())
        )

        tcld = (
            self._tcld(
                teacher_outputs=teacher_outputs,
                s_logits=s_logits,
                s_boxes=s_boxes,
                feature_shapes=feature_shapes,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
            )
            if self.lambda_tcld != 0.0 else det.new_zeros(())
        )

        total = self.lambda_det * det + self.lambda_lcmd * lcmd + self.lambda_tcld * tcld
        return {"total": total, "det": det, "lcmd": lcmd, "tcld": tcld}

    # ------------------------------------------------------------------ LCMD

    def _lcmd(
        self,
        *,
        teacher_features: dict[str, torch.Tensor] | None,
        student_features: dict[str, torch.Tensor] | None,
        student_outputs: Any,
        gt_boxes: list[torch.Tensor],
    ) -> torch.Tensor:
        ft_raw = teacher_features.get(self.teacher_feature_key) if teacher_features is not None else None
        if ft_raw is None:
            raise KeyError(
                f"CLoCKDistill LCMD не получил teacher-фичу {self.teacher_feature_key!r}. "
                "Учитель должен быть собран через "
                "lwdetr_small_for_detection(clockdistill=True) "
                "(configs/model/teacher/lwdetr_clockdistill.yaml)."
            )
        ft = self._unwrap_feature(ft_raw, name="teacher memory").detach()
        if ft.ndim != 4:
            raise ValueError(f"Teacher memory должен быть [B, C, H, W], получено {tuple(ft.shape)}")

        if student_features is not None and self.student_feature_key in student_features:
            fs = self._unwrap_feature(student_features[self.student_feature_key], name="student feature")
        else:
            fs = self._student_feature_fallback(student_outputs)

        if fs.ndim != 4:
            raise ValueError(f"Student feature должен быть [B, C, H, W], получено {tuple(fs.shape)}")

        fs = self.feature_adapter(fs)
        if fs.shape[-2:] != ft.shape[-2:]:
            fs = F.interpolate(fs, size=ft.shape[-2:], mode="bilinear", align_corners=False)
        if fs.shape[1] != ft.shape[1]:
            raise ValueError(f"Student/teacher channels после adapter'а не совпадают: student={fs.shape[1]}, teacher={ft.shape[1]}")
        if fs.shape[0] != ft.shape[0]:
            raise ValueError("Batch size student и teacher feature не совпадает")

        height, width = ft.shape[-2:]
        location, scale = self._location_scale_masks(gt_boxes, height, width, device=ft.device, dtype=ft.dtype)

        error = (ft.float() - fs.float()).pow(2).mean(dim=1, keepdim=True)

        fg_term = (self.alpha * location * scale * error).sum(dim=(1, 2, 3))
        bg_term = (self.beta * (1.0 - location) * scale * error).sum(dim=(1, 2, 3))

        return (fg_term + bg_term).mean()

    def _location_scale_masks(
        self, gt_boxes: list[torch.Tensor], height: int, width: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(gt_boxes)
        location = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)
        scale = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)
        total_px = float(height * width)

        for b in range(batch_size):
            boxes = gt_boxes[b]
            if boxes.numel() == 0:
                scale[b, 0] = 1.0 / total_px
                continue

            cx, cy, bw, bh = boxes.unbind(-1)
            x1 = torch.floor((cx - bw / 2) * width).clamp(0, width - 1)
            y1 = torch.floor((cy - bh / 2) * height).clamp(0, height - 1)
            x2 = torch.ceil((cx + bw / 2) * width).clamp(1, width)
            y2 = torch.ceil((cy + bh / 2) * height).clamp(1, height)

            box_area_px = (x2 - x1).clamp_min(1.0) * (y2 - y1).clamp_min(1.0)
            weights = 1.0 / box_area_px

            xs = torch.arange(width, device=device, dtype=dtype).view(1, 1, width)
            ys = torch.arange(height, device=device, dtype=dtype).view(1, height, 1)
            inside = (
                (xs >= x1.view(-1, 1, 1)) & (xs < x2.view(-1, 1, 1)) &
                (ys >= y1.view(-1, 1, 1)) & (ys < y2.view(-1, 1, 1))
            )

            fg = inside.any(dim=0)
            # max по боксам = приоритет меньшему боксу (больший вес),
            # тот же приём, что DCKD._rasterize_weighted_boxes.
            weighted = (inside.to(dtype) * weights.view(-1, 1, 1)).amax(dim=0)

            n_bg = (total_px - fg.to(dtype).sum()).clamp_min(1.0)

            location[b, 0] = fg.to(dtype)
            scale[b, 0] = torch.where(fg, weighted, torch.full_like(weighted, 1.0) / n_bg)

        return location, scale

    def _student_feature_fallback(self, student_outputs: Any) -> torch.Tensor:
        raw = self._unwrap_yolo_output(student_outputs)
        feats = raw.get("feats") if isinstance(raw, dict) else (raw if isinstance(raw, (list, tuple)) else None)
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        feats = [f for f in (feats or []) if isinstance(f, torch.Tensor) and f.ndim == 4]
        if not feats:
            raise ValueError("LCMD не смог получить YOLO features из student_outputs")

        expected_channels = self.feature_adapter.in_channels if isinstance(self.feature_adapter, nn.Conv2d) else feats[-1].shape[1]
        matching = [f for f in feats if f.shape[1] == expected_channels]
        return matching[-1] if matching else feats[-1]

    # ------------------------------------------------------------------ TCLD

    def _tcld(
        self,
        *,
        teacher_outputs: Any,
        s_logits: torch.Tensor,
        s_boxes: torch.Tensor,
        feature_shapes: list[tuple[int, int]],
        gt_boxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
    ) -> torch.Tensor:
        decoder = self._read_optional(teacher_outputs, "clockdistill_decoder")
        memory = self._read_optional(teacher_outputs, "clockdistill_memory")
        memory_mask = self._read_optional(teacher_outputs, "clockdistill_memory_mask")
        spatial_shapes = self._read_optional(teacher_outputs, "clockdistill_spatial_shapes")
        spatial_shapes_list = self._read_optional(teacher_outputs, "clockdistill_spatial_shapes_list")
        level_start_index = self._read_optional(teacher_outputs, "clockdistill_level_start_index")
        valid_ratios = self._read_optional(teacher_outputs, "clockdistill_valid_ratios")
        class_embed = self._read_optional(teacher_outputs, "clockdistill_class_embed")
        bbox_embed = self._read_optional(teacher_outputs, "clockdistill_bbox_embed")

        if decoder is None or memory is None or class_embed is None or bbox_embed is None:
            raise KeyError(
                "В teacher_outputs нет clockdistill_* полей — CLoCKDistillLoss требует "
                "учителя, собранного через lwdetr_small_for_detection(clockdistill=True) "
                "(configs/model/teacher/lwdetr_clockdistill.yaml)."
            )

        batch_size = s_logits.shape[0]
        device = s_logits.device

        counts = [min(gt_boxes[b].shape[0], self.max_target_queries) for b in range(batch_size)]
        g_max = max(counts) if counts else 0
        if g_max == 0:
            return s_logits.sum() * 0.0

        query_boxes = torch.zeros((batch_size, g_max, 4), device=device, dtype=memory.dtype)
        query_classes = torch.zeros((batch_size, g_max), device=device, dtype=torch.long)
        valid_mask = torch.zeros((batch_size, g_max), device=device, dtype=torch.bool)

        for b in range(batch_size):
            n = counts[b]
            if n == 0:
                continue
            classes_b = gt_labels[b][:n].long()
            if (classes_b < 0).any() or (classes_b >= self.num_classes).any():
                raise ValueError(f'target[{b}]["labels"] вне диапазона [0, {self.num_classes})')
            query_boxes[b, :n] = gt_boxes[b][:n].to(dtype=memory.dtype)
            query_classes[b, :n] = classes_b
            valid_mask[b, :n] = True

        with torch.no_grad():
            target_content = self.content_embed(query_classes)
            decoder_outputs = decoder(
                inputs_embeds=target_content,
                reference_points=query_boxes,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                encoder_hidden_states=memory,
                encoder_attention_mask=memory_mask,
            )
            teacher_logits = class_embed(decoder_outputs.last_hidden_state)
            bbox_delta = bbox_embed(decoder_outputs.last_hidden_state)
            teacher_boxes = refine_bboxes(decoder_outputs.intermediate_reference_points[-1], bbox_delta)

        point_index = self._point_indices(t_boxes=teacher_boxes, feature_shapes=feature_shapes)
        num_classes_dim = s_logits.shape[-1]
        s_logits_at_points = torch.gather(s_logits, 1, point_index.unsqueeze(-1).expand(-1, -1, num_classes_dim))
        s_boxes_at_points = torch.gather(s_boxes, 1, point_index.unsqueeze(-1).expand(-1, -1, 4))

        teacher_probs_raw = self._teacher_foreground_probs(teacher_logits, temperature=1.0)
        weight = teacher_probs_raw.max(dim=-1).values * valid_mask.to(teacher_probs_raw.dtype)

        teacher_probs_soft = self._teacher_foreground_probs(teacher_logits, temperature=self.temperature)
        cls_per_point = F.binary_cross_entropy_with_logits(
            s_logits_at_points / self.temperature, teacher_probs_soft, reduction="none"
        ).mean(dim=-1) * self.temperature ** 2

        l1_per_point = (s_boxes_at_points - teacher_boxes).abs().mean(dim=-1)
        giou = self._elementwise_giou(self._cxcywh_to_xyxy(s_boxes_at_points), self._cxcywh_to_xyxy(teacher_boxes))
        giou_per_point = 1.0 - giou

        weight_sum = weight.sum().clamp_min(self.eps)
        cls_loss = (weight * cls_per_point).sum() / weight_sum
        l1_loss = (weight * l1_per_point).sum() / weight_sum
        giou_loss = (weight * giou_per_point).sum() / weight_sum

        return self.lambda_cls * cls_loss + self.lambda_l1 * l1_loss + self.lambda_giou * giou_loss

    def _point_indices(
        self, *, t_boxes: torch.Tensor, feature_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Дословно как в KDDETRLoss._point_indices (src/losses/kd_detr_loss.py):
        ближайший anchor студента по уровню FPN (stride^2 ближе всего к площади
        бокса в пикселях) и по (cx,cy) внутри уровня.
        """
        device = t_boxes.device
        batch_size, num_points = t_boxes.shape[:2]

        image_height = feature_shapes[0][0] * self.det_strides[0]
        image_width = feature_shapes[0][1] * self.det_strides[0]

        area_px = (t_boxes[..., 2] * image_width) * (t_boxes[..., 3] * image_height)
        stride_sq = torch.tensor([s * s for s in self.det_strides], device=device, dtype=area_px.dtype)
        level_of_point = (area_px.unsqueeze(-1) - stride_sq.view(1, 1, -1)).abs().argmin(dim=-1)

        flat_index = torch.zeros(batch_size, num_points, dtype=torch.long, device=device)
        level_start = 0
        for level, (height, width) in enumerate(feature_shapes):
            mask = level_of_point == level
            col = (t_boxes[..., 0] * width).floor().clamp(0, width - 1).long()
            row = (t_boxes[..., 1] * height).floor().clamp(0, height - 1).long()
            idx = level_start + row * width + col
            flat_index = torch.where(mask, idx, flat_index)
            level_start += height * width

        return flat_index

    # ----------------------------------------------------- YOLO output parsing
    # (дословно как в KDDETRLoss — тот же формат, тот же способ разбора)

    def _student_predictions(self, outputs: Any) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        raw = self._unwrap_yolo_output(outputs)
        if not (isinstance(raw, dict) and "scores" in raw and "boxes" in raw):
            raise KeyError(
                "CLoCKDistillLoss ожидает стандартный YOLOv8 output "
                '{"boxes": raw_dfl, "scores": raw_class_logits, "feats": feature_maps}.'
            )
        scores, raw_boxes, feats = raw["scores"], raw["boxes"], raw.get("feats")
        if not isinstance(scores, torch.Tensor) or not isinstance(raw_boxes, torch.Tensor):
            raise TypeError('YOLO outputs["scores"] и outputs["boxes"] должны быть Tensor')

        s_logits = self._canonical_student_logits(scores)
        feature_shapes = self._feature_shapes(feats)
        s_boxes = self._decode_yolo_boxes(raw_boxes=raw_boxes, feature_shapes=feature_shapes)
        return s_logits, s_boxes, feature_shapes

    def _unwrap_yolo_output(self, outputs: Any) -> Any:
        if isinstance(outputs, dict):
            if "one2many" in outputs:
                return self._unwrap_yolo_output(outputs["one2many"])
            if "scores" in outputs and "boxes" in outputs:
                return outputs
            for key in ("preds", "raw_outputs", "raw_preds"):
                if key in outputs:
                    candidate = self._unwrap_yolo_output(outputs[key])
                    if candidate is not None:
                        return candidate
            return outputs
        if isinstance(outputs, tuple):
            for item in reversed(outputs):
                if isinstance(item, (dict, list, tuple)):
                    candidate = self._unwrap_yolo_output(item)
                    if isinstance(candidate, dict) and ("scores" in candidate or "one2many" in candidate):
                        return candidate
            return outputs
        return outputs

    def _feature_shapes(self, feats: Any) -> list[tuple[int, int]]:
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        if not isinstance(feats, (list, tuple)) or not feats:
            raise ValueError('Для декодирования YOLO output нужен outputs["feats"] с feature maps detection head')
        shapes = [(f.shape[-2], f.shape[-1]) for f in feats if isinstance(f, torch.Tensor) and f.ndim == 4]
        if len(shapes) != len(self.det_strides):
            raise ValueError(f"Число уровней YOLO feature maps ({len(shapes)}) не совпадает с det_strides ({len(self.det_strides)})")
        return shapes

    def _decode_yolo_boxes(self, *, raw_boxes: torch.Tensor, feature_shapes: list[tuple[int, int]]) -> torch.Tensor:
        if raw_boxes.ndim != 3:
            raise ValueError(f"YOLO raw boxes должны иметь [B, 4*reg_max, N] или [B, N, 4*reg_max], получено {tuple(raw_boxes.shape)}")
        if raw_boxes.shape[1] % 4 == 0 and raw_boxes.shape[1] >= 4:
            boxes_bcn = raw_boxes
        elif raw_boxes.shape[-1] % 4 == 0 and raw_boxes.shape[-1] >= 4:
            boxes_bcn = raw_boxes.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Не удалось определить DFL dimension: shape={tuple(raw_boxes.shape)}")

        reg_max = boxes_bcn.shape[1] // 4
        feature_points = sum(h * w for h, w in feature_shapes)
        if feature_points != boxes_bcn.shape[2]:
            raise ValueError(f"Predictions ({boxes_bcn.shape[2]}) != feature_points ({feature_points})")

        batch_size, _, num_points = boxes_bcn.shape
        if reg_max > 1:
            distribution = boxes_bcn.view(batch_size, 4, reg_max, num_points).softmax(dim=2)
            projection = torch.arange(reg_max, device=boxes_bcn.device, dtype=boxes_bcn.dtype).view(1, 1, reg_max, 1)
            distances = (distribution * projection).sum(dim=2).transpose(1, 2).contiguous()
        else:
            distances = boxes_bcn.view(batch_size, 4, num_points).transpose(1, 2).contiguous()

        anchors, scales = self._normalized_yolo_anchors(feature_shapes=feature_shapes, device=boxes_bcn.device, dtype=boxes_bcn.dtype)
        left_top = distances[..., :2] / scales
        right_bottom = distances[..., 2:] / scales
        xy1 = anchors.unsqueeze(0) - left_top
        xy2 = anchors.unsqueeze(0) + right_bottom
        center = (xy1 + xy2) * 0.5
        size = xy2 - xy1
        return torch.cat((center, size), dim=-1).clamp(min=0.0, max=1.0)

    @staticmethod
    def _normalized_yolo_anchors(*, feature_shapes: list[tuple[int, int]], device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        anchors, scales = [], []
        for height, width in feature_shapes:
            y = torch.arange(height, device=device, dtype=dtype) + 0.5
            x = torch.arange(width, device=device, dtype=dtype) + 0.5
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            anchors.append(torch.stack((xx / width, yy / height), dim=-1).reshape(-1, 2))
            scales.append(torch.tensor([width, height], device=device, dtype=dtype).view(1, 2).expand(height * width, 2))
        return torch.cat(anchors, dim=0), torch.cat(scales, dim=0)

    # --------------------------------------------------------------- прочее

    def _prepare_labels(self, labels: Any, batch_size: int) -> list[dict[str, torch.Tensor]]:
        if isinstance(labels, (list, tuple)):
            return list(labels)
        if not isinstance(labels, dict):
            raise TypeError(f"labels должны быть list или dict, получено {type(labels).__name__}")

        if "batch_idx" in labels:
            batch_idx = labels["batch_idx"].long().flatten()
            class_labels = next((labels[k] for k in ("labels", "class_labels", "cls") if k in labels), None)
            boxes = next((labels[k] for k in ("kd_boxes", "bboxes", "boxes") if k in labels), None)
            if class_labels is None or boxes is None:
                raise KeyError(f"Не найдены labels/boxes. Ключи: {list(labels.keys())}")
            class_labels = class_labels.flatten()
            return [
                {"labels": class_labels[batch_idx == b], "kd_boxes": boxes[batch_idx == b]}
                for b in range(batch_size)
            ]

        class_key = next((k for k in ("labels", "class_labels", "cls") if k in labels), None)
        box_key = next((k for k in ("kd_boxes", "bboxes", "boxes") if k in labels), None)
        if class_key is None or box_key is None:
            raise KeyError(f"Не удалось разобрать targets. Ключи: {list(labels.keys())}")
        return [{"labels": labels[class_key][b].flatten(), "kd_boxes": labels[box_key][b]} for b in range(batch_size)]

    def _target_kd_boxes(self, target: dict[str, torch.Tensor]) -> torch.Tensor:
        if "kd_boxes" in target:
            boxes, name = target["kd_boxes"], 'target["kd_boxes"]'
        elif "boxes" in target:
            boxes, name = target["boxes"], 'target["boxes"]'
        else:
            raise KeyError('CLoCKDistill требует target["boxes"] или target["kd_boxes"]')

        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError(f"{name} должен иметь shape [N, 4], получено {tuple(boxes.shape)}")

        if boxes.numel() and boxes.max() > 1.001:
            size = target.get("size", target.get("orig_size"))
            if size is None:
                raise ValueError(f"{name} выглядит как absolute boxes, но нет size/orig_size для нормализации")
            size = torch.as_tensor(size, device=boxes.device, dtype=boxes.dtype).flatten()
            height, width = size[0], size[1]
            x1, y1, x2, y2 = boxes.unbind(-1)
            boxes = torch.stack(((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height), dim=-1)

        if not torch.isfinite(boxes).all():
            raise ValueError(f"{name} содержит NaN или Inf")
        return boxes

    def _teacher_foreground_probs(self, logits: torch.Tensor, *, temperature: float) -> torch.Tensor:
        if logits.shape[-1] < self.num_classes:
            raise ValueError(f"Teacher logits имеют {logits.shape[-1]} каналов, а num_classes={self.num_classes}")
        if logits.shape[-1] == self.num_classes + 1:
            return F.softmax(logits / temperature, dim=-1)[..., : self.num_classes]
        return torch.sigmoid(logits[..., : self.num_classes] / temperature)

    def _canonical_student_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError(f"student logits должны иметь [B, N, C] или [B, C, N], получено {tuple(logits.shape)}")
        if logits.shape[-1] == self.num_classes:
            return logits
        if logits.shape[1] == self.num_classes:
            return logits.transpose(1, 2).contiguous()
        raise ValueError(f"Не удалось определить class dimension: shape={tuple(logits.shape)}, num_classes={self.num_classes}")

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)

    def _elementwise_giou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        x1 = torch.maximum(boxes1[..., 0], boxes2[..., 0])
        y1 = torch.maximum(boxes1[..., 1], boxes2[..., 1])
        x2 = torch.minimum(boxes1[..., 2], boxes2[..., 2])
        y2 = torch.minimum(boxes1[..., 3], boxes2[..., 3])
        intersection = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
        area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0.0) * (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0.0)
        area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0.0) * (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0.0)
        union = (area1 + area2 - intersection).clamp_min(self.eps)
        iou = intersection / union
        xc1 = torch.minimum(boxes1[..., 0], boxes2[..., 0])
        yc1 = torch.minimum(boxes1[..., 1], boxes2[..., 1])
        xc2 = torch.maximum(boxes1[..., 2], boxes2[..., 2])
        yc2 = torch.maximum(boxes1[..., 3], boxes2[..., 3])
        enclosing = ((xc2 - xc1).clamp(min=0.0) * (yc2 - yc1).clamp(min=0.0)).clamp_min(self.eps)
        return iou - (enclosing - union) / enclosing

    @staticmethod
    def _unwrap_feature(x: Any, *, name: str) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)) and x:
            tensors = [item for item in x if isinstance(item, torch.Tensor)]
            if tensors:
                return tensors[-1]
        raise TypeError(f"{name}: ожидался Tensor или list/tuple Tensor, получено {type(x).__name__}")

    @staticmethod
    def _read_optional(obj: Any, name: str) -> Any | None:
        value = getattr(obj, name, None)
        if value is not None:
            return value
        if isinstance(obj, dict):
            return obj.get(name)
        return None
