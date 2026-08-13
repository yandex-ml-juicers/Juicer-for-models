from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou, generalized_box_iou
from src.losses.base import DistillationLoss
from src.losses.yolo_loss import YOLO

class DCKDLoss(DistillationLoss):
    """DCKD loss for LW-DETR teacher -> YOLOv8 student.

    Student outputs
    ---------------
    No YOLO wrapper is required. Standard Ultralytics YOLOv8 training outputs
    are supported directly: {"boxes": raw_dfl, "scores": raw_class_logits,
    "feats": feature_maps}. End-to-end outputs with "one2many" and older
    list/tuple outputs with one tensor per detection level are also supported.

    If kd_logits and kd_boxes are already present, they are used directly.

    Required teacher outputs
    ------------------------
    LW-DETR outputs must contain:
        logits:
            [B, Q, C + 1] when teacher_has_no_object=True.

        pred_boxes:
            [B, Q, 4], normalized cxcywh.

        query_features / decoder_hidden_state / last_hidden_state:
            [B, Q, D], final decoder query embeddings.

    Features for HoKFD
    ------------------
    student_features[student_feature_key] (from Trainer hooks, если заданы) или,
    по умолчанию, YOLO's own outputs["feats"] (последний по глубине уровень,
    подобранный по числу каналов): [B, Cs, Hs, Ws].
    teacher_features[teacher_feature_key] (от Trainer hooks) или, по умолчанию,
    teacher_outputs.projected_feature — LW-DETR оборачивается в
    _LwDetrWithProjectedFeature (src/models/factory.py), который снимает выход
    projector-слоя backbone и прикладывает его к выходу как обычное поле:
    [B, Ct, Ht, Wt].

    The feature adapter is trainable. Therefore criterion.parameters() must be
    included in the optimizer when lambda_hokfd > 0 and Cs != Ct.

    Targets
    -------
    target["kd_boxes"] in normalized cxcywh is preferred. If it is absent,
    target["boxes"] is used. Absolute xyxy boxes are normalized automatically
    when target["size"] or target["orig_size"] is available.
    """
    requires_teacher = True
    required_features: tuple[str, ...] = ()

    def __init__(
        self, 
        num_classes: int, 
        student_feature_key: str, 
        teacher_feature_key: str, 
        student_channels: int, 
        teacher_channels: int, *, 
        lambda_det: float=1.0, 
        lambda_hekld: float=1.0, 
        lambda_hokfd: float=1.0, 
        temperature: float=1.0, 
        teacher_has_no_object: bool=True,
        teacher_topk: int=100, 
        student_topk: int=1000, 
        teacher_score_threshold: float=0.0, 
        match_cls_weight: float=1.0, 
        match_l1_weight: float=5.0, 
        match_giou_weight: float=2.0, 
        local_iou_threshold: float=0.5,
        local_gt_floor: float=1.0,
        quality_gamma: float=0.5,
        attention_temperature: float=0.1,
        det_strides: tuple[int, ...]=(8, 16, 32),
        det_reg_max: int=16,
        det_box_gain: float=7.5,
        det_cls_gain: float=0.5,
        det_dfl_gain: float=1.5,
        eps: float=1e-06
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError('num_classes должен быть > 0')
        if temperature <= 0:
            raise ValueError('temperature должен быть > 0')
        if attention_temperature <= 0:
            raise ValueError('attention_temperature должен быть > 0')
        if not 0.0 <= teacher_score_threshold <= 1.0:
            raise ValueError('teacher_score_threshold должен быть в [0, 1]')
        if not 0.0 <= local_iou_threshold <= 1.0:
            raise ValueError('local_iou_threshold должен быть в [0, 1]')
        if local_gt_floor < 0.0:
            raise ValueError('local_gt_floor должен быть >= 0')
        if not 0.0 <= quality_gamma <= 1.0:
            raise ValueError('quality_gamma должен быть в [0, 1]')

        self.num_classes = num_classes
        self.student_feature_key = student_feature_key
        self.teacher_feature_key = teacher_feature_key
        self.student_required_features = (student_feature_key,)
        self.teacher_required_features = (teacher_feature_key,)
        self.lambda_det = lambda_det
        self.lambda_hekld = lambda_hekld
        self.lambda_hokfd = lambda_hokfd
        self.temperature = temperature
        self.teacher_has_no_object = teacher_has_no_object
        self.teacher_topk = teacher_topk
        self.student_topk = student_topk
        self.teacher_score_threshold = teacher_score_threshold
        self.match_cls_weight = match_cls_weight
        self.match_l1_weight = match_l1_weight
        self.match_giou_weight = match_giou_weight
        self.local_iou_threshold = local_iou_threshold
        self.local_gt_floor = local_gt_floor
        self.quality_gamma = quality_gamma
        self.attention_temperature = attention_temperature
        self.eps = eps
        # det больше не самописный Hungarian-лосс: студент — dense anchor-based
        # YOLO, ему нужен родной TAL-assigner + DFL, а не one-to-one
        # DETR-style мэтчинг. YOLO — та же обёртка, на которой учится
        # baseline (cityscapes_finetune_yolov8n), так что GT-часть DCKD и
        # baseline теперь сравнимы.
        self.task_loss = YOLO(
            num_classes=num_classes,
            strides=det_strides,
            reg_max=det_reg_max,
            box_gain=det_box_gain,
            cls_gain=det_cls_gain,
            dfl_gain=det_dfl_gain,
        )
        self.feature_adapter = nn.Identity() if student_channels == teacher_channels else nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False)

    def forward(
        self, 
        student_outputs: Any, 
        teacher_outputs: Any | None, 
        labels: list[dict[str, torch.Tensor]], 
        student_features: dict[str, torch.Tensor] | None = None, 
        teacher_features: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        s_logits, s_boxes = self._student_predictions(student_outputs)
        raw_labels = labels
        labels = self._prepare_labels(labels, batch_size=s_logits.shape[0])

        if isinstance(student_outputs, dict):
            student_outputs["kd_logits"] = s_logits
            student_outputs["kd_boxes"] = s_boxes

        # student_outputs ещё не тронут (kd_logits/kd_boxes — новые ключи, не
        # замена старых), raw_labels — исходный ultralytics-формат
        # (batch_idx/cls/bboxes), поэтому YOLO получает ровно то же,
        # что получил бы при обычном YOLO-обучении без дистилляции.
        det = self.task_loss(student_outputs, teacher_outputs=None, labels=raw_labels)["total"]

        if teacher_outputs is None:
            zero = det.new_zeros(())
            return {"total": self.lambda_det * det, "det": det, "hekld": zero, "hokfd": zero}

        t_logits = self._canonical_teacher_logits(self._read(teacher_outputs, "logits").detach())
        t_boxes = self._canonical_boxes(self._read(teacher_outputs, "pred_boxes").detach(), name="teacher pred_boxes")

        self._validate_detection_tensors(s_logits=s_logits, s_boxes=s_boxes, t_logits=t_logits, t_boxes=t_boxes)

        if self.lambda_hekld != 0.0:
            hekld = self._heterogeneous_logits_distillation(s_logits=s_logits, s_boxes=s_boxes, t_logits=t_logits, t_boxes=t_boxes)
        else:
            hekld = det.new_zeros(())

        if self.lambda_hokfd != 0.0:
            t_queries = self._read_any(teacher_outputs, ("query_features", "decoder_hidden_state", "last_hidden_state")).detach()
            hokfd = self._homogeneous_feature_distillation(
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                s_logits=s_logits,
                s_boxes=s_boxes, 
                t_logits=t_logits, 
                t_boxes=t_boxes, 
                t_queries=t_queries, 
                labels=labels, 
                student_features=student_features, 
                teacher_features=teacher_features)
        else:
            hokfd = det.new_zeros(())

        total = self.lambda_det * det + self.lambda_hekld * hekld + self.lambda_hokfd * hokfd

        return {"total": total, "det": det, "hekld": hekld, "hokfd": hokfd}

    def _student_predictions(
        self, 
        outputs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kd_logits = self._read_optional(outputs, "kd_logits")
        kd_boxes = self._read_optional(outputs, "kd_boxes")

        if isinstance(kd_logits, torch.Tensor) and isinstance(kd_boxes, torch.Tensor):
            s_logits = self._canonical_student_logits(kd_logits)
            s_boxes = self._canonical_boxes(kd_boxes, name="student kd_boxes")
            return s_logits, s_boxes

        raw = self._unwrap_yolo_output(outputs)

        if isinstance(raw, dict) and "scores" in raw and "boxes" in raw:
            scores = raw["scores"]
            raw_boxes = raw["boxes"]
            feats = raw.get("feats")

            if not isinstance(scores, torch.Tensor) or not isinstance(raw_boxes, torch.Tensor):
                raise TypeError('YOLO outputs["scores"] и outputs["boxes"] должны быть Tensor')

            s_logits = self._canonical_student_logits(scores)
            s_boxes = self._decode_yolo_boxes(raw_boxes=raw_boxes, feats=feats)

            return s_logits, s_boxes

        if isinstance(raw, (list, tuple)) and raw and all(isinstance(x, torch.Tensor) and x.ndim == 4 for x in raw):
            channels = raw[0].shape[1]
            box_channels = channels - self.num_classes

            if box_channels <= 0 or box_channels % 4 != 0:
                raise ValueError(f"Не удалось разделить старый YOLO output на DFL и class channels: channels={channels}, num_classes={self.num_classes}")

            raw_boxes = torch.cat([x[:, :box_channels].flatten(2) for x in raw], dim=2)
            scores = torch.cat([x[:, box_channels:].flatten(2) for x in raw], dim=2)

            s_logits = self._canonical_student_logits(scores)
            s_boxes = self._decode_yolo_boxes(raw_boxes=raw_boxes, feats=list(raw))

            return s_logits, s_boxes

        raise KeyError("DCKD не смог распознать student_outputs. Поддерживаются стандартный YOLOv8 output {boxes, scores, feats}, end2end {one2many, one2one}, старый list feature maps или готовые kd_logits/kd_boxes.")

    def _unwrap_yolo_output(
        self, 
        outputs: Any
    ) -> Any:
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
                    if isinstance(candidate, (list, tuple)) and candidate and all(isinstance(x, torch.Tensor) and x.ndim == 4 for x in candidate):
                        return candidate

            return outputs

        return outputs

    def _decode_yolo_boxes(
        self, 
        *, 
        raw_boxes: torch.Tensor, 
        feats: Any
    ) -> torch.Tensor:
        if raw_boxes.ndim != 3:
            raise ValueError(f"YOLO raw boxes должны иметь shape [B, 4*reg_max, N] или [B, N, 4*reg_max], получено {tuple(raw_boxes.shape)}")

        if raw_boxes.shape[1] % 4 == 0 and raw_boxes.shape[1] >= 4:
            boxes_bcn = raw_boxes
        elif raw_boxes.shape[-1] % 4 == 0 and raw_boxes.shape[-1] >= 4:
            boxes_bcn = raw_boxes.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Не удалось определить DFL dimension YOLO boxes: shape={tuple(raw_boxes.shape)}")

        reg_max = boxes_bcn.shape[1] // 4
        if reg_max <= 0:
            raise ValueError("YOLO reg_max должен быть > 0")

        feature_shapes = self._feature_shapes(feats)
        feature_points = sum(height * width for height, width in feature_shapes)

        if feature_points != boxes_bcn.shape[2]:
            raise ValueError(f"Количество YOLO predictions не совпадает с feature maps: predictions={boxes_bcn.shape[2]}, feature_points={feature_points}")

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

    def _feature_shapes(
        self, 
        feats: Any
    ) -> list[tuple[int, int]]:
        if isinstance(feats, torch.Tensor):
            feats = [feats]

        if not isinstance(feats, (list, tuple)) or not feats:
            raise ValueError('Для декодирования стандартного YOLO output нужен outputs["feats"] с feature maps detection head')

        shapes: list[tuple[int, int]] = []

        for feat in feats:
            if not isinstance(feat, torch.Tensor) or feat.ndim != 4:
                raise ValueError("Каждый YOLO feature map должен быть Tensor [B, C, H, W]")
            shapes.append((feat.shape[-2], feat.shape[-1]))

        return shapes

    @staticmethod
    def _normalized_yolo_anchors(
        *, 
        feature_shapes: list[tuple[int, int]], 
        device: torch.device, 
        dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchors: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []

        for height, width in feature_shapes:
            y = torch.arange(height, device=device, dtype=dtype) + 0.5
            x = torch.arange(width, device=device, dtype=dtype) + 0.5
            yy, xx = torch.meshgrid(y, x, indexing="ij")

            anchors.append(torch.stack((xx / width, yy / height), dim=-1).reshape(-1, 2))
            scales.append(torch.tensor([width, height], device=device, dtype=dtype).view(1, 2).expand(height * width, 2))

        return torch.cat(anchors, dim=0), torch.cat(scales, dim=0)

    def _heterogeneous_logits_distillation(
        self, 
        *, 
        s_logits: torch.Tensor, 
        s_boxes: torch.Tensor, 
        t_logits: torch.Tensor, 
        t_boxes: torch.Tensor
    ) -> torch.Tensor:

        batch_size = s_logits.shape[0]
        cost_matrices: list[torch.Tensor | None] = [None] * batch_size
        teacher_idx_per_image: list[torch.Tensor | None] = [None] * batch_size
        student_idx_per_image: list[torch.Tensor | None] = [None] * batch_size

        # Фаза A: только GPU-работа, без единого .cpu()/.item(). Все cost-матрицы
        # батча копятся тут, поэтому CUDA-стрим может закинуть в очередь все
        # b-итерации подряд, не останавливаясь на каждой.
        for b in range(batch_size):
            # temperature=1.0 здесь намеренно, а не self.temperature: для
            # выбора пар (кто с кем матчится) нужны острые, неразмытые
            # вероятности учителя — иначе Hungarian-мэтчинг начинает путать
            # похожие по мягкому распределению классы. self.temperature
            # применяется только в _matched_logit_loss, на самом лоссе —
            # там и должна работать классическая Hinton-температура.
            t_probs = self._teacher_foreground_probs(t_logits[b], temperature=1.0)
            s_probs = torch.sigmoid(s_logits[b])
            t_scores = t_probs.max(dim=-1).values
            s_scores = s_probs.max(dim=-1).values

            if self.teacher_score_threshold > 0.0:
                valid_t = torch.nonzero(t_scores >= self.teacher_score_threshold, as_tuple=False).squeeze(1)
            else:
                valid_t = torch.arange(t_scores.numel(), device=t_scores.device)
            if valid_t.numel() == 0 or s_scores.numel() == 0:
                continue
            if valid_t.numel() > self.teacher_topk:
                local_top = torch.topk(t_scores[valid_t], k=self.teacher_topk, sorted=False).indices
                t_idx = valid_t[local_top]
            else:
                t_idx = valid_t
            s_idx = torch.topk(s_scores, k=min(self.student_topk, s_scores.numel()), sorted=False).indices

            if t_idx.numel() == 0 or s_idx.numel() == 0:
                continue

            cost = self._matching_cost(t_probs=t_probs[t_idx], t_boxes=t_boxes[b, t_idx], s_probs=s_probs[s_idx], s_boxes=s_boxes[b, s_idx])

            cost_matrices[b] = cost.detach().float()
            teacher_idx_per_image[b] = t_idx
            student_idx_per_image[b] = s_idx

        # Единая точка синхронизации на весь батч вместо одной на каждую
        # картинку: раньше .cpu().numpy() внутри цикла форсировал sync
        # CUDA-стрима 32 раза подряд, простаивая GPU между вызовами scipy.
        # Первый .cpu() здесь дожидается уже полностью посчитанной на GPU
        # очереди (Фаза A), остальные — быстрые, GPU уже свободен.
        cost_matrices_cpu = [
            cost.cpu().numpy() if cost is not None else None
            for cost in cost_matrices
        ]

        # Фаза B: чистый CPU, без обращений к GPU — Hungarian-матчинг, как раньше.
        losses: list[torch.Tensor] = []
        for b in range(batch_size):
            cost_cpu = cost_matrices_cpu[b]
            if cost_cpu is None:
                continue

            t_idx = teacher_idx_per_image[b]
            s_idx = student_idx_per_image[b]

            row, col = linear_sum_assignment(cost_cpu)
            row = torch.as_tensor(row, device=s_logits.device, dtype=torch.long)
            col = torch.as_tensor(col, device=s_logits.device, dtype=torch.long)

            teacher_indices = t_idx[row]
            student_indices = s_idx[col]

            losses.append(self._matched_logit_loss(student_logits=s_logits[b, student_indices], teacher_logits=t_logits[b, teacher_indices]))

        if not losses:
            return s_logits.sum() * 0.0

        return torch.stack(losses).mean()

    def _matching_cost(
        self, 
        *, 
        t_probs: torch.Tensor, 
        t_boxes: torch.Tensor, 
        s_probs: torch.Tensor, 
        s_boxes: torch.Tensor
    ) -> torch.Tensor:
        # Не классический DETR-cost (-p(target_class)): у учителя нет одной
        # "правильной" метки — logits дают мягкое multi-label распределение
        # по всем классам, поэтому cost — симметричная BCE между двумя
        # распределениями (учитель как soft-target), а не индекс в hard label.
        t = t_probs[:, None, :]
        s = s_probs[None, :, :].clamp(self.eps, 1.0 - self.eps)
        cls_cost = -(t * torch.log(s) + (1.0 - t) * torch.log(1.0 - s)).mean(dim=-1)
        l1_cost = torch.cdist(t_boxes, s_boxes, p=1)
        giou_cost = -generalized_box_iou(self._cxcywh_to_xyxy(t_boxes), self._cxcywh_to_xyxy(s_boxes))

        return self.match_cls_weight * cls_cost + self.match_l1_weight * l1_cost + self.match_giou_weight * giou_cost

    def _matched_logit_loss(self, *, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        temperature = self.temperature
        teacher_probs = self._teacher_foreground_probs(teacher_logits, temperature=temperature)
        loss = F.binary_cross_entropy_with_logits(student_logits / temperature, teacher_probs, reduction='mean')
        return loss * temperature ** 2

    def _homogeneous_feature_distillation(
        self,
        *,
        student_outputs: Any,
        teacher_outputs: Any,
        s_logits: torch.Tensor,
        s_boxes: torch.Tensor,
        t_logits: torch.Tensor,
        t_boxes: torch.Tensor,
        t_queries: torch.Tensor,
        labels: list[dict[str, torch.Tensor]],
        student_features: dict[str, torch.Tensor] | None,
        teacher_features: dict[str, torch.Tensor] | None
    ) -> torch.Tensor:
        if student_features is not None and self.student_feature_key in student_features:
            fs = self._unwrap_feature(student_features[self.student_feature_key], name="student feature")

        else:
            raw = self._unwrap_yolo_output(student_outputs)

            if isinstance(raw, dict):
                feats = raw.get("feats")
            elif isinstance(raw, (list, tuple)):
                feats = raw
            else:
                feats = None

            if isinstance(feats, torch.Tensor):
                feats = [feats]
            if not isinstance(feats, (list, tuple)) or not feats:
                raise ValueError("HoKFD не смог получить YOLO features из student_outputs")

            feats = [feat for feat in feats if isinstance(feat, torch.Tensor) and feat.ndim == 4]

            if not feats:
                raise ValueError("В YOLO outputs не найдены feature maps [B, C, H, W]")

            if isinstance(self.feature_adapter, nn.Conv2d):
                expected_channels = self.feature_adapter.in_channels
            else:
                expected_channels = t_queries.shape[-1]

            matching_features = [feat for feat in feats if feat.shape[1] == expected_channels]

            if matching_features:
                fs = matching_features[-1]
            else:
                fs = feats[-1]

        if fs.ndim != 4:
            raise ValueError(f"YOLO feature должен иметь [B, C, H, W], получено {tuple(fs.shape)}")
        fs = self.feature_adapter(fs)
        if t_queries.ndim != 3:
            raise ValueError(f"Teacher queries должны иметь [B, Q, D], получено {tuple(t_queries.shape)}")
        if fs.shape[0] != t_queries.shape[0]:
            raise ValueError(f"Batch size student feature и teacher queries не совпадает: student={fs.shape[0]}, teacher={t_queries.shape[0]}")
        if fs.shape[1] != t_queries.shape[-1]:
            raise ValueError(
                f"Каналы YOLO feature после adapter должны совпадать с размерностью LW-DETR queries. "
                f"student C={fs.shape[1]}, teacher D={t_queries.shape[-1]}"
            )

        gt_boxes = [self._target_kd_boxes(target).to(device=s_boxes.device, dtype=s_boxes.dtype) for target in labels]

        # Приоритет — Trainer-хуки (student_features/teacher_features), если
        # они когда-нибудь будут заведены для этой пары моделей; по умолчанию
        # teacher_outputs.projected_feature — обычное поле выхода учителя,
        # его прикладывает _LwDetrWithProjectedFeature (src/models/factory.py).
        if teacher_features is not None and self.teacher_feature_key in teacher_features:
            ft_raw = teacher_features[self.teacher_feature_key]
        else:
            ft_raw = self._read_optional(teacher_outputs, self.teacher_feature_key)

        if ft_raw is not None:
            ft = self._unwrap_feature(ft_raw, name="teacher feature").detach()
            if ft.ndim != 4:
                raise ValueError(f"Teacher feature должен иметь [B, C, H, W], получено {tuple(ft.shape)}")
            if fs.shape[-2:] != ft.shape[-2:]:
                fs = F.interpolate(fs, size=ft.shape[-2:], mode="bilinear", align_corners=False)
            if fs.shape[1] != ft.shape[1]:
                raise ValueError(f"Student и teacher feature channels не совпадают: student={fs.shape[1]}, teacher={ft.shape[1]}")

            global_mask = self._global_mask(ft=ft, t_queries=t_queries, t_logits=t_logits, t_boxes=t_boxes, gt_boxes=gt_boxes)
            local_mask = self._local_mask(spatial_size=ft.shape[-2:], s_logits=s_logits, s_boxes=s_boxes, gt_boxes=gt_boxes)

            fusion = self._normalize_mask(global_mask.float() + local_mask.float()).detach()
            feature_error = (ft.float() - fs.float()).pow(2).mean(dim=1, keepdim=True)

            numerator = (fusion * feature_error).sum()
            denominator = fusion.sum().clamp_min(self.eps)

            return numerator / denominator

        batch_size, channels, height, width = fs.shape

        student_tokens = fs.flatten(2).transpose(1, 2)
        student_tokens_norm = F.normalize(student_tokens.float(), dim=-1)

        teacher_queries = t_queries.detach().float()
        teacher_queries_norm = F.normalize(teacher_queries, dim=-1)

        similarity = torch.einsum("bnc,bqc->bnq", student_tokens_norm, teacher_queries_norm)
        similarity = similarity / self.attention_temperature

        teacher_probs = self._teacher_foreground_probs(t_logits.float(), temperature=1.0)
        cls_quality = teacher_probs.max(dim=-1).values

        loc_quality = torch.zeros_like(cls_quality)

        for b in range(batch_size):
            if gt_boxes[b].numel() == 0:
                continue

            teacher_boxes_xyxy = self._cxcywh_to_xyxy(t_boxes[b].float())
            gt_boxes_xyxy = self._cxcywh_to_xyxy(gt_boxes[b].float())

            ious = box_iou(teacher_boxes_xyxy, gt_boxes_xyxy)
            loc_quality[b] = ious.max(dim=-1).values

        gamma = self.quality_gamma

        quality = cls_quality.clamp_min(self.eps).pow(gamma) * loc_quality.clamp_min(self.eps).pow(1.0 - gamma)

        attention_logits = similarity + torch.log(quality.clamp_min(self.eps)).unsqueeze(1)
        query_attention = F.softmax(attention_logits, dim=-1).detach()

        teacher_tokens = torch.einsum("bnq,bqc->bnc", query_attention, teacher_queries)

        global_mask = (query_attention * quality.unsqueeze(1)).sum(dim=-1)
        global_mask = global_mask.view(batch_size, 1, height, width)

        local_mask = self._local_mask(
            spatial_size=(height, width),
            s_logits=s_logits,
            s_boxes=s_boxes,
            gt_boxes=gt_boxes
        )

        fusion = self._normalize_mask(global_mask.float() + local_mask.float()).detach()

        feature_error = (student_tokens.float() - teacher_tokens).pow(2).mean(dim=-1)
        feature_error = feature_error.view(batch_size, 1, height, width)

        numerator = (fusion * feature_error).sum()
        denominator = fusion.sum().clamp_min(self.eps)

        return numerator / denominator

    def _global_mask(
        self, 
        *, 
        ft: torch.Tensor, 
        t_queries: torch.Tensor, 
        t_logits: torch.Tensor, 
        t_boxes: torch.Tensor, 
        gt_boxes: list[torch.Tensor]
    ) -> torch.Tensor:
        batch_size, channels, height, width = ft.shape
        spatial = F.normalize(ft.flatten(2).transpose(1, 2), dim=-1)
        queries = F.normalize(t_queries, dim=-1)

        similarity = torch.einsum('bqc,bnc->bqn', queries, spatial)
        attention = F.softmax(similarity / self.attention_temperature, dim=-1)
        teacher_probs = self._teacher_foreground_probs(t_logits, temperature=1.0)

        cls_quality = teacher_probs.max(dim=-1).values
        loc_quality = torch.zeros_like(cls_quality)

        for b in range(batch_size):
            if gt_boxes[b].numel() == 0:
                continue
            ious = box_iou(self._cxcywh_to_xyxy(t_boxes[b]), self._cxcywh_to_xyxy(gt_boxes[b]))
            loc_quality[b] = ious.max(dim=-1).values

        gamma = self.quality_gamma
        quality = cls_quality.clamp_min(self.eps).pow(gamma) * loc_quality.clamp_min(self.eps).pow(1.0 - gamma)
        weighted_attention = attention * quality.unsqueeze(-1)

        mask = weighted_attention.sum(dim=1)
        normalizer = quality.sum(dim=1, keepdim=True).clamp_min(self.eps)
        mask = mask / normalizer

        return mask.view(batch_size, 1, height, width)

    def _local_mask(
        self, 
        *, 
        spatial_size: tuple[int, int], 
        s_logits: torch.Tensor, 
        s_boxes: torch.Tensor, 
        gt_boxes: list[torch.Tensor]
    ) -> torch.Tensor:
        height, width = spatial_size
        probs = torch.sigmoid(s_logits)
        scores = probs.max(dim=-1).values
        masks: list[torch.Tensor] = []

        for b in range(s_boxes.shape[0]):
            components: list[torch.Tensor] = []

            # Floor по каждому GT-боксу: local_mask не должна занулиться там,
            # где студент ещё не научился уверенно детектировать объект — а
            # именно туда feature-дистилляция должна давить сильнее всего.
            # Растеризуется векторно (все GT-боксы картинки разом), итог
            # объединяется через max ниже — тот же эффект, что раньше давали
            # присваивания в срез по каждому боксу отдельно.
            if self.local_gt_floor > 0.0 and gt_boxes[b].numel() > 0:
                floor_weights = gt_boxes[b].new_full((gt_boxes[b].shape[0],), self.local_gt_floor)
                components.append(self._rasterize_weighted_boxes(gt_boxes[b], floor_weights, height, width))

            if gt_boxes[b].numel() > 0:
                k = min(self.student_topk, scores[b].numel())

                if k > 0:
                    idx = torch.topk(scores[b], k=k, sorted=False).indices
                    boxes = s_boxes[b, idx].detach()
                    box_scores = scores[b, idx].detach()
                    iou = box_iou(self._cxcywh_to_xyxy(boxes), self._cxcywh_to_xyxy(gt_boxes[b]))
                    keep = iou.max(dim=-1).values >= self.local_iou_threshold

                    # torch.maximum по student-боксам раньше могла только
                    # поднять floor, никогда не опустить — здесь то же самое:
                    # компонента kept-боксов участвует в max наравне с floor.
                    if keep.any():
                        components.append(self._rasterize_weighted_boxes(boxes[keep], box_scores[keep], height, width))

            if components:
                mask = torch.stack(components, dim=0).amax(dim=0)
            else:
                mask = s_boxes.new_zeros((1, height, width))

            masks.append(mask)

        return torch.stack(masks, dim=0)

    @staticmethod
    def _as_loss_tensor(value: Any) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value.mean() if value.ndim > 0 else value
        if isinstance(value, (tuple, list)) and value:
            first = value[0]
            if isinstance(first, torch.Tensor):
                return first.mean() if first.ndim > 0 else first
        return None

    @staticmethod
    def _ensure_differentiable_loss(loss: torch.Tensor, *, name: str) -> torch.Tensor:
        if not loss.is_floating_point():
            raise TypeError(f'{name} должен быть floating-point Tensor')
        if not loss.requires_grad:
            raise ValueError(f'{name} не требует grad. Не передавай detached loss items; нужен настоящий training loss YOLO.')
        return loss

    def _teacher_foreground_probs(self, logits: torch.Tensor, *, temperature: float) -> torch.Tensor:
        if logits.shape[-1] < self.num_classes:
            raise ValueError(f'Teacher logits имеют {logits.shape[-1]} каналов, а num_classes={self.num_classes}')

        if logits.shape[-1] == self.num_classes + 1:
            probs = F.softmax(logits / temperature, dim=-1)

            return probs[..., :self.num_classes]

        logits = logits[..., :self.num_classes]

        return torch.sigmoid(logits / temperature)

    def _canonical_student_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError(f'student kd_logits должны иметь [B, N, C] или [B, C, N], получено {tuple(logits.shape)}')
        if logits.shape[-1] == self.num_classes:
            return logits
        if logits.shape[1] == self.num_classes:
            return logits.transpose(1, 2).contiguous()
        raise ValueError(f'Не удалось определить class dimension student kd_logits: shape={tuple(logits.shape)}, num_classes={self.num_classes}. Передай только raw classification logits YOLO, без DFL channels.')

    def _canonical_teacher_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError(f'teacher logits должны иметь [B, Q, C(+1)], получено {tuple(logits.shape)}')
        return logits

    def _canonical_boxes(self, boxes: torch.Tensor, *, name: str) -> torch.Tensor:
        if boxes.ndim != 3:
            raise ValueError(f'{name} должны иметь [B, N, 4] или [B, 4, N], получено {tuple(boxes.shape)}')
        if boxes.shape[-1] == 4:
            result = boxes
        elif boxes.shape[1] == 4:
            result = boxes.transpose(1, 2).contiguous()
        else:
            raise ValueError(f'Не удалось определить box dimension для {name}: shape={tuple(boxes.shape)}')
        self._validate_cxcywh(result, name=name)
        return result

    def _target_kd_boxes(
        self, 
        target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if "kd_boxes" in target:
            boxes = target["kd_boxes"]
            name = 'target["kd_boxes"]'
        elif "boxes" in target:
            boxes = target["boxes"]
            name = 'target["boxes"]'
        else:
            raise KeyError('DCKD требует target["boxes"] или target["kd_boxes"]')

        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError(f"{name} должен иметь shape [N, 4], получено {tuple(boxes.shape)}")

        if boxes.numel() and boxes.max() > 1.001:
            size = target.get("size", target.get("orig_size"))

            if size is None:
                raise ValueError(f"{name} выглядит как absolute boxes, но в target нет size/orig_size для нормализации")

            size = torch.as_tensor(size, device=boxes.device, dtype=boxes.dtype).flatten()

            if size.numel() != 2:
                raise ValueError("target size/orig_size должен содержать [height, width]")

            height, width = size[0], size[1]
            x1, y1, x2, y2 = boxes.unbind(-1)
            boxes = torch.stack(((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height), dim=-1)
            name = f"{name} converted absolute xyxy -> normalized cxcywh"

        self._validate_cxcywh(boxes, name=name)

        return boxes

    def _prepare_labels(
        self,
        labels: list[dict[str, torch.Tensor]] | dict[str, torch.Tensor],
        batch_size: int
    ) -> list[dict[str, torch.Tensor]]:
        if isinstance(labels, (list, tuple)):
            return list(labels)

        if not isinstance(labels, dict):
            raise TypeError(f"labels должны быть list или dict, получено {type(labels).__name__}")

        if "batch_idx" in labels:
            batch_idx = labels["batch_idx"].long().flatten()

            if "labels" in labels:
                class_labels = labels["labels"]
            elif "class_labels" in labels:
                class_labels = labels["class_labels"]
            elif "cls" in labels:
                class_labels = labels["cls"]
            else:
                raise KeyError(f"Не найдены labels/class_labels/cls. Ключи: {list(labels.keys())}")

            if "kd_boxes" in labels:
                boxes = labels["kd_boxes"]
            elif "bboxes" in labels:
                boxes = labels["bboxes"]
            elif "boxes" in labels:
                boxes = labels["boxes"]
            else:
                raise KeyError(f"Не найдены kd_boxes/bboxes/boxes. Ключи: {list(labels.keys())}")

            class_labels = class_labels.flatten()
            targets = []

            for b in range(batch_size):
                mask = batch_idx == b
                targets.append({"labels": class_labels[mask], "kd_boxes": boxes[mask]})

            return targets

        class_key = None

        for key in ("labels", "class_labels", "cls"):
            if key in labels:
                class_key = key
                break

        box_key = None

        for key in ("kd_boxes", "bboxes", "boxes"):
            if key in labels:
                box_key = key
                break

        if class_key is None or box_key is None:
            raise KeyError(f"Не удалось разобрать targets. Ключи: {list(labels.keys())}")

        class_labels = labels[class_key]
        boxes = labels[box_key]
        targets = []

        for b in range(batch_size):
            targets.append({"labels": class_labels[b].flatten(), "kd_boxes": boxes[b]})

        return targets

    def _validate_cxcywh(self, boxes: torch.Tensor, *, name: str) -> None:
        if not torch.isfinite(boxes).all():
            raise ValueError(f'{name} содержит NaN или Inf')

        if boxes.numel() == 0:
            return

        if (boxes[..., 2:] < 0).any():
            raise ValueError(f'{name}: width/height должны быть >= 0')

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)

    @staticmethod
    def _rasterize_weighted_boxes(boxes: torch.Tensor, weights: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Растеризует N cxcywh-боксов (нормализованных) на сетку (height,
        width), каждому присваивая вес weights[i]; итог — поэлементный max по
        боксам. Векторный эквивалент прежнего python-цикла + _box_pixel_bounds:
        те же floor/ceil-границы пикселей, что и раньше, но для всех боксов
        сразу и без единого .item()/.cpu() (а значит без GPU-стопов на каждый
        бокс — критично для плотных Cityscapes-сцен с десятками объектов на
        картинку).
        """
        cx, cy, bw, bh = boxes.unbind(-1)
        x1 = torch.floor((cx - bw / 2) * width).clamp(0, width - 1)
        y1 = torch.floor((cy - bh / 2) * height).clamp(0, height - 1)
        x2 = torch.ceil((cx + bw / 2) * width).clamp(1, width)
        y2 = torch.ceil((cy + bh / 2) * height).clamp(1, height)

        xs = torch.arange(width, device=boxes.device, dtype=boxes.dtype).view(1, 1, width)
        ys = torch.arange(height, device=boxes.device, dtype=boxes.dtype).view(1, height, 1)

        # Полуоткрытые интервалы [x1, x2) x [y1, y2) — то же самое, что раньше
        # давал срез mask[:, y1:y2, x1:x2]. Вырожденные боксы (x2<=x1 после
        # floor/ceil/clamp) естественно не дают вклада — без явной проверки.
        inside = (
            (xs >= x1.view(-1, 1, 1)) & (xs < x2.view(-1, 1, 1)) &
            (ys >= y1.view(-1, 1, 1)) & (ys < y2.view(-1, 1, 1))
        )  # [N, H, W]

        filled = inside.to(weights.dtype) * weights.view(-1, 1, 1)
        return filled.amax(dim=0, keepdim=True)  # [1, H, W]

    @staticmethod
    def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
        batch_size = mask.shape[0]
        flat = mask.flatten(1)
        max_value = flat.max(dim=1).values.view(batch_size, 1, 1, 1).clamp_min(1e-06)
        return mask / max_value

    @staticmethod
    def _unwrap_feature(x: Any, *, name: str) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)) and x:
            tensors = [item for item in x if isinstance(item, torch.Tensor)]
            if tensors:
                return tensors[-1]
        raise TypeError(f'{name}: ожидался Tensor или list/tuple Tensor, получено {type(x).__name__}')

    @staticmethod
    def _read(obj: Any, name: str) -> torch.Tensor:
        value = DCKDLoss._read_optional(obj, name)
        if not isinstance(value, torch.Tensor):
            raise KeyError(f'В outputs отсутствует Tensor `{name}`')
        return value

    @staticmethod
    def _read_any(obj: Any, names: tuple[str, ...]) -> torch.Tensor:
        for name in names:
            value = DCKDLoss._read_optional(obj, name)
            if isinstance(value, torch.Tensor):
                return value
        raise KeyError(f'Не найден teacher query Tensor. Ожидался один из ключей: {names}')

    @staticmethod
    def _read_optional(obj: Any, name: str) -> Any | None:
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    def _validate_detection_tensors(self, *, s_logits: torch.Tensor, s_boxes: torch.Tensor, t_logits: torch.Tensor, t_boxes: torch.Tensor) -> None:
        if s_logits.shape[0] != t_logits.shape[0]:
            raise ValueError('Batch size teacher и student logits не совпадает')
        if s_boxes.shape[0] != s_logits.shape[0]:
            raise ValueError('Batch size student boxes/logits не совпадает')
        if t_boxes.shape[0] != t_logits.shape[0]:
            raise ValueError('Batch size teacher boxes/logits не совпадает')
        if s_boxes.shape[1] != s_logits.shape[1]:
            raise ValueError(f'Количество YOLO kd_boxes и kd_logits должно совпадать: boxes={s_boxes.shape[1]}, logits={s_logits.shape[1]}')
        if t_boxes.shape[1] != t_logits.shape[1]:
            raise ValueError(f'Количество LW-DETR pred_boxes и logits должно совпадать: boxes={t_boxes.shape[1]}, logits={t_logits.shape[1]}')
        if s_logits.shape[-1] != self.num_classes:
            raise ValueError(f'student kd_logits должны иметь ровно {self.num_classes} class channels, получено {s_logits.shape[-1]}')
        if t_logits.shape[-1] < self.num_classes:
            raise ValueError(f'Недостаточно class channels у teacher: ожидалось >= {self.num_classes}, получено {t_logits.shape[-1]}')