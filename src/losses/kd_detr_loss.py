from __future__ import annotations
from typing import Any
import torch
import torch.nn.functional as F
from src.losses.base import DistillationLoss
from src.losses.yolov8n_loss import YOLOv8Loss


class KDDETRLoss(DistillationLoss):
    """KD-DETR (Wang et al., CVPR 2024, https://arxiv.org/abs/2211.08071) для
    LW-DETR (учитель) -> YOLOv8n (студент).

    Что делает статья
    ------------------
    Проблема, которую решает статья: object queries учителя и студента
    "egocentric" — их число и порядок не совпадают, поэтому нет прямого
    соответствия между предсказанием учителя и предсказанием студента.
    Решение — "specialized object queries": набор координатно заданных,
    необучаемых probe-точек, общих по конструкции для обеих моделей
    ("general distillation points" — случайные/сеточные позиции; "specific
    distillation points" — переиспользованные реальные запросы учителя).
    Для каждой точки берётся вес по уверенности учителя
    w_i = max_c p_teacher(c|q_i), и лосс

        L_distill = Σ w_i · (λ_cls·KL_T(teacher‖student) + λ_L1·L1(box) + λ_GIoU·GIoU(box))

    с λ_cls=1, λ_L1=5, λ_GIoU=2 (значения из статьи).

    Что здесь адаптировано под YOLOv8n
    -----------------------------------
    У студента нет decoder и нет query — он не может "принять" общую
    probe-точку так, как это делает учитель. Поэтому "точка дистилляции"
    здесь — не query, а координата (cx,cy,w,h): учитель отвечает на неё,
    прогоняя decoder ещё раз с этой точкой как reference_points
    (src/models/lwdetr_kd_detr.py, teacher_outputs.probe_logits/probe_boxes —
    general points), а студент отвечает "бесплатно": он и так делает dense
    per-anchor предсказание, поэтому достаточно прочитать его выход в
    ближайшем anchor'е того уровня FPN, чей stride ближе всего к масштабу
    объекта. Specific points — это все реальные запросы учителя
    (teacher_outputs.logits/pred_boxes) без дополнительного decoder-прохода.

    Требует учителя, собранного с num_probe_points > 0 (см.
    src.models.factory.lwdetr_small_for_detection).

    Классификационный член — не буквальный softmax-KL из статьи: у student
    (YOLO) классификация multi-label (независимый sigmoid на класс), а не
    закрытый K-way softmax, и teacher в этом проекте сконфигурирован без
    "no-object" класса (config.num_labels каналов, не +1) — то же
    несоответствие параметризации, что уже решено в DCKD
    (src/losses/dckd_loss.py): teacher-логиты приводятся к multi-label
    foreground-вероятностям, и по ним считается температурный BCE — тот же
    сюррогат KL, что и там, ради сравнимости между лоссами этого проекта.
    """

    requires_teacher = True
    required_features: tuple[str, ...] = ()

    def __init__(
        self,
        num_classes: int,
        *,
        lambda_distill: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_l1: float = 5.0,
        lambda_giou: float = 2.0,
        temperature: float = 2.0,
        teacher_has_no_object: bool = False,
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

        self.num_classes = num_classes
        self.lambda_distill = lambda_distill
        self.lambda_cls = lambda_cls
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.temperature = temperature
        self.teacher_has_no_object = teacher_has_no_object
        self.det_strides = det_strides
        self.eps = eps

        # Task loss студента не зависит от статьи: тот же YOLOv8Loss, что у
        # baseline (cityscapes_finetune_yolov8n) и у DCKD — GT-часть у всех
        # трёх экспериментов одинакова, сравнение честное.
        self.task_loss = YOLOv8Loss(
            num_classes=num_classes,
            strides=det_strides,
            reg_max=det_reg_max,
            box_gain=det_box_gain,
            cls_gain=det_cls_gain,
            dfl_gain=det_dfl_gain,
        )

    def forward(
        self,
        student_outputs: Any,
        teacher_outputs: Any | None,
        labels: list[dict[str, torch.Tensor]],
        student_features: dict[str, torch.Tensor] | None = None,
        teacher_features: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        s_logits, s_boxes, feature_shapes = self._student_predictions(student_outputs)

        if isinstance(student_outputs, dict):
            student_outputs["kd_logits"] = s_logits
            student_outputs["kd_boxes"] = s_boxes

        det = self.task_loss(student_outputs, teacher_outputs=None, labels=labels)["total"]

        if teacher_outputs is None:
            zero = det.new_zeros(())
            return {"total": det, "det": det, "distill": zero, "cls": zero, "l1": zero, "giou": zero}

        probe_logits = self._read_optional(teacher_outputs, "probe_logits")
        probe_boxes = self._read_optional(teacher_outputs, "probe_boxes")
        if not isinstance(probe_logits, torch.Tensor) or not isinstance(probe_boxes, torch.Tensor):
            raise KeyError(
                "В teacher_outputs нет probe_logits/probe_boxes — KDDETRLoss требует учителя, "
                "собранного с num_probe_points > 0 (src.models.factory.lwdetr_small_for_detection). "
                "Обычный конфиг model/teacher/lwdetr.yaml для этого лосса не годится."
            )

        specific_logits = self._read(teacher_outputs, "logits")
        specific_boxes = self._read(teacher_outputs, "pred_boxes")

        t_logits = torch.cat([probe_logits.detach(), specific_logits.detach()], dim=1)
        t_boxes = self._canonical_boxes(
            torch.cat([probe_boxes.detach(), specific_boxes.detach()], dim=1), name="teacher points"
        )

        if t_logits.shape[0] != s_logits.shape[0]:
            raise ValueError("Batch size teacher и student не совпадает")

        point_index = self._point_indices(t_boxes=t_boxes, feature_shapes=feature_shapes)
        num_classes = s_logits.shape[-1]

        s_logits_at_points = torch.gather(
            s_logits, 1, point_index.unsqueeze(-1).expand(-1, -1, num_classes)
        )
        s_boxes_at_points = torch.gather(
            s_boxes, 1, point_index.unsqueeze(-1).expand(-1, -1, 4)
        )

        # w_i — на "сырых" вероятностях учителя (temperature=1.0): вес точки
        # должен отражать реальную уверенность учителя, а не размытое
        # температурой распределение. Мягкая цель для BCE считается отдельно,
        # с self.temperature — так же разделены веса и цель в DCKD.
        teacher_probs_raw = self._teacher_foreground_probs(t_logits, temperature=1.0)
        weight = teacher_probs_raw.max(dim=-1).values  # [B, P]

        teacher_probs_soft = self._teacher_foreground_probs(t_logits, temperature=self.temperature)
        cls_per_point = F.binary_cross_entropy_with_logits(
            s_logits_at_points / self.temperature, teacher_probs_soft, reduction="none"
        ).mean(dim=-1) * self.temperature ** 2

        l1_per_point = (s_boxes_at_points - t_boxes).abs().mean(dim=-1)

        giou = self._elementwise_giou(
            self._cxcywh_to_xyxy(s_boxes_at_points), self._cxcywh_to_xyxy(t_boxes)
        )
        giou_per_point = 1.0 - giou

        weight_sum = weight.sum().clamp_min(self.eps)
        cls_loss = (weight * cls_per_point).sum() / weight_sum
        l1_loss = (weight * l1_per_point).sum() / weight_sum
        giou_loss = (weight * giou_per_point).sum() / weight_sum

        distill = self.lambda_cls * cls_loss + self.lambda_l1 * l1_loss + self.lambda_giou * giou_loss
        total = det + self.lambda_distill * distill

        return {
            "total": total,
            "det": det,
            "distill": distill,
            "cls": cls_loss,
            "l1": l1_loss,
            "giou": giou_loss,
        }

    def _point_indices(
        self,
        *,
        t_boxes: torch.Tensor,
        feature_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Для каждой точки учителя (cx,cy,w,h) — индекс ближайшего anchor'а
        студента (в развёрнутом по всем уровням [sum(H*W)] представлении).

        Уровень FPN выбирается по площади бокса в пикселях: тот, чей
        stride^2 ближе всего к площади (эвристика "бокс должен занимать
        порядка одной ячейки сетки на своём уровне" — тот же принцип, что и
        у multi-scale assignment в FCOS/YOLO, только не по порогам площади,
        а по ближайшему stride). Внутри уровня — просто ближайшая по (cx,cy)
        ячейка сетки.
        """
        device = t_boxes.device
        batch_size, num_points = t_boxes.shape[:2]

        image_height = feature_shapes[0][0] * self.det_strides[0]
        image_width = feature_shapes[0][1] * self.det_strides[0]

        area_px = (t_boxes[..., 2] * image_width) * (t_boxes[..., 3] * image_height)  # [B, P]
        stride_sq = torch.tensor(
            [stride * stride for stride in self.det_strides], device=device, dtype=area_px.dtype
        )
        level_cost = (area_px.unsqueeze(-1) - stride_sq.view(1, 1, -1)).abs()
        level_of_point = level_cost.argmin(dim=-1)  # [B, P]

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

    def _student_predictions(
        self, outputs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        raw = self._unwrap_yolo_output(outputs)

        if not (isinstance(raw, dict) and "scores" in raw and "boxes" in raw):
            raise KeyError(
                "KDDETRLoss ожидает стандартный YOLOv8 output "
                '{"boxes": raw_dfl, "scores": raw_class_logits, "feats": feature_maps}.'
            )

        scores = raw["scores"]
        raw_boxes = raw["boxes"]
        feats = raw.get("feats")

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

        shapes: list[tuple[int, int]] = []
        for feat in feats:
            if not isinstance(feat, torch.Tensor) or feat.ndim != 4:
                raise ValueError("Каждый YOLO feature map должен быть Tensor [B, C, H, W]")
            shapes.append((feat.shape[-2], feat.shape[-1]))

        if len(shapes) != len(self.det_strides):
            raise ValueError(
                f"Число уровней YOLO feature maps ({len(shapes)}) не совпадает с det_strides ({len(self.det_strides)})"
            )

        return shapes

    def _decode_yolo_boxes(
        self, *, raw_boxes: torch.Tensor, feature_shapes: list[tuple[int, int]]
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

    @staticmethod
    def _normalized_yolo_anchors(
        *, feature_shapes: list[tuple[int, int]], device: torch.device, dtype: torch.dtype
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

    def _teacher_foreground_probs(self, logits: torch.Tensor, *, temperature: float) -> torch.Tensor:
        if logits.shape[-1] < self.num_classes:
            raise ValueError(f"Teacher logits имеют {logits.shape[-1]} каналов, а num_classes={self.num_classes}")

        if logits.shape[-1] == self.num_classes + 1:
            probs = F.softmax(logits / temperature, dim=-1)
            return probs[..., : self.num_classes]

        logits = logits[..., : self.num_classes]
        return torch.sigmoid(logits / temperature)

    def _canonical_student_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError(f"student logits должны иметь [B, N, C] или [B, C, N], получено {tuple(logits.shape)}")
        if logits.shape[-1] == self.num_classes:
            return logits
        if logits.shape[1] == self.num_classes:
            return logits.transpose(1, 2).contiguous()
        raise ValueError(f"Не удалось определить class dimension student logits: shape={tuple(logits.shape)}, num_classes={self.num_classes}")

    def _canonical_boxes(self, boxes: torch.Tensor, *, name: str) -> torch.Tensor:
        if boxes.ndim != 3:
            raise ValueError(f"{name} должны иметь [B, N, 4] или [B, 4, N], получено {tuple(boxes.shape)}")
        if boxes.shape[-1] == 4:
            result = boxes
        elif boxes.shape[1] == 4:
            result = boxes.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Не удалось определить box dimension для {name}: shape={tuple(boxes.shape)}")

        if not torch.isfinite(result).all():
            raise ValueError(f"{name} содержит NaN или Inf")

        return result

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)

    def _elementwise_giou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """GIoU поточечно (boxes1[..., i] против boxes2[..., i]), а не полная
        NxM матрица (torchvision.ops.generalized_box_iou): точки уже
        сопоставлены индексом anchor'а, кросс-произведение не нужно.
        """
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
    def _read(obj: Any, name: str) -> torch.Tensor:
        value = KDDETRLoss._read_optional(obj, name)
        if not isinstance(value, torch.Tensor):
            raise KeyError(f"В outputs отсутствует Tensor `{name}`")
        return value

    @staticmethod
    def _read_optional(obj: Any, name: str) -> Any | None:
        # Атрибут — в приоритете: transformers.ModelOutput сам является
        # подклассом dict, но probe_logits/probe_boxes (см.
        # src/models/lwdetr_kd_detr.py) — не объявленные dataclass-поля,
        # ModelOutput.__setattr__ кладёт такие только в __dict__, не в
        # dict-ключи, поэтому obj.get(name) их не находит, а getattr — находит.
        value = getattr(obj, name, None)
        if value is not None:
            return value
        if isinstance(obj, dict):
            return obj.get(name)
        return None
