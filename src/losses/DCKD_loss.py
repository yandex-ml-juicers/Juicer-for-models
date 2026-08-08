from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou, generalized_box_iou

from src.losses.base import DistillationLoss


class DCKDLoss(DistillationLoss):
    """DCKD: DETR teacher -> CNN detector student.

    Expected student outputs:
        loss or loss_dict               supervised detector loss
        kd_logits: [B, Ns, C]           raw pre-NMS classification logits
        kd_boxes:  [B, Ns, 4]           normalized cxcywh boxes

    Expected teacher outputs:
        logits:     [B, Nt, C(+1)]       DETR class logits
        pred_boxes: [B, Nt, 4]           normalized cxcywh boxes
        query_features: [B, Nt, D]       final decoder query embeddings

    FeatureExtractor dictionaries:
        student_features[student_feature_key]: [B, Cs, Hs, Ws]
        teacher_features[teacher_feature_key]: [B, Ct, Ht, Wt]

    For HoKFD, query_features last dim must equal Ct. Pick a teacher feature
    after DETR channel projection if necessary.
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
        *,
        lambda_det: float = 1.0,
        lambda_hekld: float = 1.0,
        lambda_hokfd: float = 1.0,
        temperature: float = 1.0,
        student_cls_mode: str = "sigmoid",
        teacher_has_no_object: bool = True,
        teacher_topk: int = 100,
        student_topk: int = 1000,
        match_cls_weight: float = 1.0,
        match_l1_weight: float = 5.0,
        match_giou_weight: float = 2.0,
        local_iou_threshold: float = 0.5,
        quality_gamma: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if student_cls_mode not in {"sigmoid", "softmax"}:
            raise ValueError("student_cls_mode должен быть 'sigmoid' или 'softmax'")
        if temperature <= 0:
            raise ValueError("temperature должен быть > 0")
        if not 0 <= local_iou_threshold <= 1:
            raise ValueError("local_iou_threshold должен быть в [0, 1]")

        self.num_classes = num_classes
        self.student_feature_key = student_feature_key
        self.teacher_feature_key = teacher_feature_key
        self.student_required_features = (student_feature_key,)
        self.teacher_required_features = (teacher_feature_key,)

        self.lambda_det = lambda_det
        self.lambda_hekld = lambda_hekld
        self.lambda_hokfd = lambda_hokfd
        self.temperature = temperature
        self.student_cls_mode = student_cls_mode
        self.teacher_has_no_object = teacher_has_no_object
        self.teacher_topk = teacher_topk
        self.student_topk = student_topk
        self.match_cls_weight = match_cls_weight
        self.match_l1_weight = match_l1_weight
        self.match_giou_weight = match_giou_weight
        self.local_iou_threshold = local_iou_threshold
        self.quality_gamma = quality_gamma
        self.eps = eps

        self.feature_adapter = (
            nn.Identity()
            if student_channels == teacher_channels
            else nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False)
        )

    def forward(
        self,
        student_outputs: Any,
        teacher_outputs: Any | None,
        labels: list[dict[str, torch.Tensor]],
        student_features: dict[str, torch.Tensor] | None = None,
        teacher_features: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        det = self._detector_loss(student_outputs)

        # В evaluation trainer вызывает criterion без teacher.
        if teacher_outputs is None:
            zero = det.new_zeros(())
            return {
                "total": self.lambda_det * det,
                "det": det,
                "hekld": zero,
                "hokfd": zero,
            }

        s_logits = self._read(student_outputs, "kd_logits")
        s_boxes = self._read(student_outputs, "kd_boxes")
        t_logits = self._read(teacher_outputs, "logits").detach()
        t_boxes = self._read(teacher_outputs, "pred_boxes").detach()
        t_queries = self._read_any(
            teacher_outputs,
            ("query_features", "decoder_hidden_state", "last_hidden_state"),
        ).detach()

        self._validate_detection_tensors(s_logits, s_boxes, t_logits, t_boxes)

        hekld = self._heterogeneous_logits_distillation(
            s_logits=s_logits,
            s_boxes=s_boxes,
            t_logits=t_logits,
            t_boxes=t_boxes,
        )

        hokfd = self._homogeneous_feature_distillation(
            s_logits=s_logits,
            s_boxes=s_boxes,
            t_logits=t_logits,
            t_boxes=t_boxes,
            t_queries=t_queries,
            labels=labels,
            student_features=student_features,
            teacher_features=teacher_features,
        )

        total = (
            self.lambda_det * det
            + self.lambda_hekld * hekld
            + self.lambda_hokfd * hokfd
        )
        return {
            "total": total,
            "det": det,
            "hekld": hekld,
            "hokfd": hokfd,
        }

    # ------------------------------------------------------------------
    # HeKLD: Hungarian matching + logits distillation
    # ------------------------------------------------------------------

    def _heterogeneous_logits_distillation(
        self,
        *,
        s_logits: torch.Tensor,
        s_boxes: torch.Tensor,
        t_logits: torch.Tensor,
        t_boxes: torch.Tensor,
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []

        for b in range(s_logits.shape[0]):
            tp = self._teacher_foreground_probs(t_logits[b], temperature=1.0)
            sp = self._student_probs(s_logits[b], temperature=1.0)

            t_score = tp.max(dim=-1).values
            s_score = sp.max(dim=-1).values

            t_idx = torch.topk(
                t_score,
                k=min(self.teacher_topk, t_score.numel()),
                sorted=False,
            ).indices
            s_idx = torch.topk(
                s_score,
                k=min(self.student_topk, s_score.numel()),
                sorted=False,
            ).indices

            if t_idx.numel() == 0 or s_idx.numel() == 0:
                continue

            cost = self._matching_cost(
                t_probs=tp[t_idx],
                t_boxes=t_boxes[b, t_idx],
                s_probs=sp[s_idx],
                s_boxes=s_boxes[b, s_idx],
            )
            row, col = linear_sum_assignment(cost.detach().float().cpu().numpy())
            row = torch.as_tensor(row, device=s_logits.device, dtype=torch.long)
            col = torch.as_tensor(col, device=s_logits.device, dtype=torch.long)

            ti = t_idx[row]
            si = s_idx[col]
            losses.append(self._matched_logit_loss(s_logits[b, si], t_logits[b, ti]))

        if not losses:
            return s_logits.sum() * 0.0
        return torch.stack(losses).mean()

    def _matching_cost(
        self,
        *,
        t_probs: torch.Tensor,
        t_boxes: torch.Tensor,
        s_probs: torch.Tensor,
        s_boxes: torch.Tensor,
    ) -> torch.Tensor:
        # [Nt, Ns] classification cost.
        if self.student_cls_mode == "sigmoid":
            t = t_probs[:, None, :]
            s = s_probs[None, :, :].clamp(self.eps, 1.0 - self.eps)
            cls_cost = -(
                t * torch.log(s) + (1.0 - t) * torch.log(1.0 - s)
            ).mean(dim=-1)
        else:
            t = t_probs / t_probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            cls_cost = -(t[:, None, :] * torch.log(s_probs[None].clamp_min(self.eps))).sum(-1)

        l1_cost = torch.cdist(t_boxes, s_boxes, p=1)
        giou_cost = -generalized_box_iou(
            self._cxcywh_to_xyxy(t_boxes),
            self._cxcywh_to_xyxy(s_boxes),
        )

        return (
            self.match_cls_weight * cls_cost
            + self.match_l1_weight * l1_cost
            + self.match_giou_weight * giou_cost
        )

    def _matched_logit_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        t = self.temperature
        teacher_probs = self._teacher_foreground_probs(teacher_logits, temperature=t)

        if self.student_cls_mode == "sigmoid":
            # Dense CNN heads (RetinaNet/FCOS-like) usually use independent sigmoid classes.
            return F.binary_cross_entropy_with_logits(
                student_logits / t,
                teacher_probs,
                reduction="mean",
            )

        teacher_probs = teacher_probs / teacher_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.eps)
        student_log_probs = F.log_softmax(student_logits / t, dim=-1)
        return F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        ) * (t**2)

    # ------------------------------------------------------------------
    # HoKFD: global teacher mask + local student mask + masked feature KD
    # ------------------------------------------------------------------

    def _homogeneous_feature_distillation(
        self,
        *,
        s_logits: torch.Tensor,
        s_boxes: torch.Tensor,
        t_logits: torch.Tensor,
        t_boxes: torch.Tensor,
        t_queries: torch.Tensor,
        labels: list[dict[str, torch.Tensor]],
        student_features: dict[str, torch.Tensor] | None,
        teacher_features: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if student_features is None or teacher_features is None:
            raise ValueError("DCKD HoKFD требует student_features и teacher_features")

        fs = self._unwrap_feature(student_features[self.student_feature_key])
        ft = self._unwrap_feature(teacher_features[self.teacher_feature_key]).detach()

        fs = self.feature_adapter(fs)
        if fs.shape[-2:] != ft.shape[-2:]:
            fs = F.interpolate(fs, size=ft.shape[-2:], mode="bilinear", align_corners=False)

        if t_queries.shape[-1] != ft.shape[1]:
            raise ValueError(
                "Для global mask размер query_features должен совпадать с channels "
                f"teacher feature: query D={t_queries.shape[-1]}, feature C={ft.shape[1]}. "
                "Возьми feature после DETR input/channel projection."
            )

        gt_boxes = [self._target_boxes_cxcywh_norm(x) for x in labels]
        global_mask = self._global_mask(
            ft=ft,
            t_queries=t_queries,
            t_logits=t_logits,
            t_boxes=t_boxes,
            gt_boxes=gt_boxes,
        )
        local_mask = self._local_mask(
            spatial_size=ft.shape[-2:],
            s_logits=s_logits,
            s_boxes=s_boxes,
            gt_boxes=gt_boxes,
        )

        # Fusion attention mask. The mask is guidance, not an optimization target.
        fusion = self._normalize_mask(global_mask + local_mask).detach()

        diff = (ft - fs).pow(2).mean(dim=1, keepdim=True)
        return (fusion * diff).sum() / fusion.sum().clamp_min(self.eps)

    def _global_mask(
        self,
        *,
        ft: torch.Tensor,
        t_queries: torch.Tensor,
        t_logits: torch.Tensor,
        t_boxes: torch.Tensor,
        gt_boxes: list[torch.Tensor],
    ) -> torch.Tensor:
        bsz, channels, height, width = ft.shape
        spatial = F.normalize(ft.flatten(2).transpose(1, 2), dim=-1)
        queries = F.normalize(t_queries, dim=-1)

        # Query-to-spatial similarity; softmax gives one spatial attention map/query.
        similarity = torch.einsum("bqc,bnc->bqn", queries, spatial)
        attention = F.softmax(similarity / (channels**0.5), dim=-1)

        teacher_probs = self._teacher_foreground_probs(t_logits, temperature=1.0)
        cls_quality = teacher_probs.max(dim=-1).values

        loc_quality = torch.zeros_like(cls_quality)
        for b in range(bsz):
            if gt_boxes[b].numel() == 0:
                continue
            ious = box_iou(
                self._cxcywh_to_xyxy(t_boxes[b]),
                self._cxcywh_to_xyxy(gt_boxes[b]),
            )
            loc_quality[b] = ious.max(dim=-1).values

        gamma = self.quality_gamma
        quality = (
            cls_quality.clamp_min(self.eps).pow(gamma)
            * loc_quality.clamp_min(self.eps).pow(1.0 - gamma)
        )

        mask = (attention * quality.unsqueeze(-1)).sum(dim=1)
        mask = mask / quality.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return mask.view(bsz, 1, height, width)

    def _local_mask(
        self,
        *,
        spatial_size: tuple[int, int],
        s_logits: torch.Tensor,
        s_boxes: torch.Tensor,
        gt_boxes: list[torch.Tensor],
    ) -> torch.Tensor:
        height, width = spatial_size
        probs = self._student_probs(s_logits, temperature=1.0)
        scores = probs.max(dim=-1).values
        masks: list[torch.Tensor] = []

        for b in range(s_boxes.shape[0]):
            mask = s_boxes.new_zeros((1, height, width))
            if gt_boxes[b].numel() == 0:
                masks.append(mask)
                continue

            k = min(self.student_topk, scores[b].numel())
            idx = torch.topk(scores[b], k=k, sorted=False).indices
            boxes = s_boxes[b, idx].detach()
            box_scores = scores[b, idx].detach()

            iou = box_iou(
                self._cxcywh_to_xyxy(boxes),
                self._cxcywh_to_xyxy(gt_boxes[b]),
            )
            keep = iou.max(dim=-1).values >= self.local_iou_threshold

            for box, score in zip(boxes[keep], box_scores[keep]):
                cx, cy, bw, bh = box
                x1 = int(torch.floor((cx - bw / 2) * width).clamp(0, width - 1).item())
                y1 = int(torch.floor((cy - bh / 2) * height).clamp(0, height - 1).item())
                x2 = int(torch.ceil((cx + bw / 2) * width).clamp(1, width).item())
                y2 = int(torch.ceil((cy + bh / 2) * height).clamp(1, height).item())
                if x2 > x1 and y2 > y1:
                    mask[:, y1:y2, x1:x2] = torch.maximum(
                        mask[:, y1:y2, x1:x2], score
                    )

            masks.append(mask)

        return torch.stack(masks, dim=0)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _detector_loss(self, outputs: Any) -> torch.Tensor:
        loss = self._read_optional(outputs, "loss")
        if isinstance(loss, torch.Tensor):
            return loss

        loss_dict = self._read_optional(outputs, "loss_dict")
        if loss_dict is None and isinstance(outputs, dict):
            # torchvision-style training output: {'loss_classifier': ..., ...}
            if outputs and all(isinstance(v, torch.Tensor) for v in outputs.values()):
                loss_dict = outputs

        if isinstance(loss_dict, dict):
            values = [v for k, v in loss_dict.items() if k.startswith("loss")]
            if values:
                return torch.stack([v if v.ndim == 0 else v.mean() for v in values]).sum()

        raise KeyError(
            "Не найден supervised detection loss. Student должен вернуть scalar `loss` "
            "или `loss_dict`."
        )

    def _teacher_foreground_probs(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        probs = F.softmax(logits / temperature, dim=-1)
        if self.teacher_has_no_object:
            probs = probs[..., : self.num_classes]
        elif probs.shape[-1] != self.num_classes:
            probs = probs[..., : self.num_classes]
        return probs

    def _student_probs(self, logits: torch.Tensor, *, temperature: float) -> torch.Tensor:
        logits = logits[..., : self.num_classes]
        if self.student_cls_mode == "sigmoid":
            return torch.sigmoid(logits / temperature)
        return F.softmax(logits / temperature, dim=-1)

    def _target_boxes_cxcywh_norm(self, target: dict[str, torch.Tensor]) -> torch.Tensor:
        if "kd_boxes" in target:
            return target["kd_boxes"]

        boxes = target.get("boxes")
        if boxes is None:
            raise KeyError("target должен содержать `kd_boxes` или `boxes`")

        # DCKD loss intentionally requires normalized cxcywh targets. If your normal
        # detector targets are xyxy/absolute, add a separate `kd_boxes` field in prepare_targets.
        if boxes.numel() and (boxes.min() < 0 or boxes.max() > 1):
            raise ValueError(
                "target['boxes'] не похожи на normalized cxcywh. "
                "Добавь target['kd_boxes'] в формате normalized cxcywh."
            )
        return boxes

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack(
            (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
            dim=-1,
        )

    @staticmethod
    def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
        b = mask.shape[0]
        flat = mask.flatten(1)
        max_v = flat.max(dim=1).values.view(b, 1, 1, 1).clamp_min(1e-6)
        return mask / max_v

    @staticmethod
    def _unwrap_feature(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)) and x and isinstance(x[-1], torch.Tensor):
            return x[-1]
        raise TypeError(f"Ожидался Tensor feature, получено {type(x).__name__}")

    @staticmethod
    def _read(obj: Any, name: str) -> torch.Tensor:
        value = DCKDLoss._read_optional(obj, name)
        if value is None:
            raise KeyError(f"В outputs отсутствует `{name}`")
        return value

    @staticmethod
    def _read_any(obj: Any, names: tuple[str, ...]) -> torch.Tensor:
        for name in names:
            value = DCKDLoss._read_optional(obj, name)
            if isinstance(value, torch.Tensor):
                return value
        raise KeyError(f"Не найден ни один из ключей: {names}")

    @staticmethod
    def _read_optional(obj: Any, name: str) -> Any | None:
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    def _validate_detection_tensors(
        self,
        s_logits: torch.Tensor,
        s_boxes: torch.Tensor,
        t_logits: torch.Tensor,
        t_boxes: torch.Tensor,
    ) -> None:
        if s_logits.ndim != 3 or t_logits.ndim != 3:
            raise ValueError("kd logits должны иметь shape [B, N, C]")
        if s_boxes.shape[-1] != 4 or t_boxes.shape[-1] != 4:
            raise ValueError("kd boxes должны иметь shape [B, N, 4]")
        if s_logits.shape[0] != t_logits.shape[0]:
            raise ValueError("Batch size teacher и student не совпадает")
        if s_logits.shape[-1] < self.num_classes:
            raise ValueError("В student kd_logits меньше num_classes классов")
        if t_logits.shape[-1] < self.num_classes:
            raise ValueError("В teacher logits меньше num_classes классов")