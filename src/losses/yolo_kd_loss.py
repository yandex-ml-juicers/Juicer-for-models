from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.base import DistillationLoss
from src.losses.yolo_loss import YOLO


class YOLOKDLoss(DistillationLoss):
    """Гомогенная YOLO->YOLO дистилляция: учитель и студент — обе dense
    YOLO-модели с одинаковой anchor-сеткой на одном разрешении (тот же набор
    strides, тот же reg_max) — проверено эмпирически: raw student.boxes и
    teacher.boxes имеют идентичную форму [B, 4*reg_max, N], student.scores и
    teacher.scores — идентичную [B, num_classes, N]. Поэтому, в отличие от
    DCKD/KD-DETR/CLoCKDistill (там учитель — DETR с query, нужен матчинг —
    Hungarian или ближайший anchor), здесь матчинг не нужен вообще: сравниваем
    логиты/фичи в тех же самых позициях напрямую.

    Три KD-компоненты поверх det:
      - logit: температурный BCE между student.scores и teacher.scores
        (multi-label sigmoid у обеих моделей — никакого сюрприза
        softmax-vs-sigmoid, как было с DETR-учителем).
      - box: температурный KL между student.boxes и teacher.boxes,
        трактуемыми как 4 независимых reg_max-way распределения (то, чем DFL
        и является по конструкции) — без декодирования в (cx,cy,w,h) и без
        L1/GIoU: сравниваем сырое распределение, не точку.
      - feat: SmoothL1 между адаптированной student-фичей и teacher-фичей
        одного уровня FPN (тот же H,W, разные каналы — обучаемый Conv1x1,
        как в DCKD/CLoCKDistill). Без масок по GT (в отличие от HoKFD/LCMD) —
        фичи и так выровнены 1:1 по пространству, специально держим этот
        компонент максимально простым.

    Важно про масштаб: det — тот же YOLO()-лосс, что и everywhere в проекте,
    с той же ultralytics-конвенцией "лосс * batch_size" (см. src/losses/
    yolo_loss.py и историю тюнинга DCKD/KD-DETR/CLoCKDistill). logit/box/feat
    — обычные средние, batch_size-независимые. λ по умолчанию НЕ 1.0 — грубая
    поправка на этот множитель, чтобы не наступить на те же грабли с первого
    прогона, но это стартовая точка, не финально откалиброванные веса: перед
    боевым прогоном стоит свериться с train_gradshare_* в history.csv, как
    делали для DCKD/KD-DETR/CLoCKDistill.
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
        lambda_logit: float = 30.0,
        lambda_box: float = 30.0,
        lambda_feat: float = 1.0,
        temperature: float = 2.0,
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
        self.student_feature_key = student_feature_key
        self.teacher_feature_key = teacher_feature_key
        self.student_required_features = (student_feature_key,)
        self.teacher_required_features = (teacher_feature_key,)
        self.reg_max = det_reg_max

        self.lambda_det = lambda_det
        self.lambda_logit = lambda_logit
        self.lambda_box = lambda_box
        self.lambda_feat = lambda_feat
        self.temperature = temperature
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

    @property
    def gradient_probe_weights(self) -> dict[str, float]:
        return {
            "det": self.lambda_det,
            "logit": self.lambda_logit,
            "box": self.lambda_box,
            "feat": self.lambda_feat,
        }

    def forward(
        self,
        student_outputs: Any,
        teacher_outputs: Any | None,
        labels: list[dict[str, torch.Tensor]],
        student_features: dict[str, torch.Tensor] | None = None,
        teacher_features: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        det = self.task_loss(student_outputs, teacher_outputs=None, labels=labels)["total"]

        if teacher_outputs is None:
            zero = det.new_zeros(())
            return {"total": self.lambda_det * det, "det": det, "logit": zero, "box": zero, "feat": zero}

        if self.lambda_logit != 0.0 or self.lambda_box != 0.0:
            # Прямое сравнение по позиции anchor'а работает только если
            # учитель тоже dense YOLO (та же сетка). Для query-based учителя
            # (например RT-DETR) выставляют lambda_logit=lambda_box=0.0 —
            # тогда сюда не заходим и не пытаемся читать student.scores/boxes
            # в форме, которую query-based учитель не может дать.
            s_scores = self._read(student_outputs, "scores")
            s_boxes = self._read(student_outputs, "boxes")
            t_scores = self._read(teacher_outputs, "scores").detach()
            t_boxes = self._read(teacher_outputs, "boxes").detach()

            if s_scores.shape != t_scores.shape:
                raise ValueError(
                    f"YOLOKDLoss ожидает одинаковую anchor-сетку у студента и учителя "
                    f"(тот же image_size/strides/num_classes): student.scores={tuple(s_scores.shape)}, "
                    f"teacher.scores={tuple(t_scores.shape)}"
                )
            if s_boxes.shape != t_boxes.shape:
                raise ValueError(
                    f"YOLOKDLoss ожидает одинаковый reg_max у студента и учителя: "
                    f"student.boxes={tuple(s_boxes.shape)}, teacher.boxes={tuple(t_boxes.shape)}"
                )

            logit = self._logit_kd(s_scores, t_scores) if self.lambda_logit != 0.0 else det.new_zeros(())
            box = self._box_kd(s_boxes, t_boxes) if self.lambda_box != 0.0 else det.new_zeros(())
        else:
            logit = det.new_zeros(())
            box = det.new_zeros(())

        feat = self._feature_kd(student_features, teacher_features)

        total = (
            self.lambda_det * det
            + self.lambda_logit * logit
            + self.lambda_box * box
            + self.lambda_feat * feat
        )
        return {"total": total, "det": det, "logit": logit, "box": box, "feat": feat}

    def _logit_kd(self, s_scores: torch.Tensor, t_scores: torch.Tensor) -> torch.Tensor:
        # multi-label sigmoid classification у обеих моделей — прямой
        # температурный BCE, тот же Hinton-style temperature^2, что и везде
        # в проекте, но без сюрприза softmax-vs-sigmoid: учитель тут тоже YOLO.
        t = self.temperature
        soft_target = torch.sigmoid(t_scores / t)
        loss = F.binary_cross_entropy_with_logits(s_scores / t, soft_target, reduction="mean")
        return loss * t ** 2

    def _box_kd(self, s_boxes: torch.Tensor, t_boxes: torch.Tensor) -> torch.Tensor:
        # DFL: 4 независимых reg_max-way распределения на anchor. Сравниваем
        # сырое распределение через KL, а не декодированную точку — не нужно
        # ни anchor-геометрии, ни L1/GIoU.
        batch_size, _channels, num_anchors = s_boxes.shape
        t = self.temperature
        s = s_boxes.view(batch_size, 4, self.reg_max, num_anchors).transpose(2, 3)
        te = t_boxes.view(batch_size, 4, self.reg_max, num_anchors).transpose(2, 3)

        student_log_probs = F.log_softmax(s / t, dim=-1)
        teacher_probs = F.softmax(te / t, dim=-1)
        # mean, не batchmean: batchmean делит сумму только на batch_size, а не
        # на число anchor'ов (~10 тыс. на нашем разрешении) — при таком числе
        # позиций это раздуло бы box ровно в ту же категорию проблемы
        # "лосс * batch_size", которую весь этот разговор чинили у
        # DCKD/KD-DETR/CLoCKDistill, только на три порядка сильнее.
        loss = F.kl_div(student_log_probs, teacher_probs, reduction="mean")
        return loss * t ** 2

    def _feature_kd(
        self,
        student_features: dict[str, torch.Tensor] | None,
        teacher_features: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if not student_features or not teacher_features:
            raise KeyError(
                "YOLOKDLoss требует Trainer-хуков student_features/teacher_features "
                f"(student_feature_key={self.student_feature_key!r}, "
                f"teacher_feature_key={self.teacher_feature_key!r})"
            )
        fs = self.feature_adapter(student_features[self.student_feature_key])
        ft = teacher_features[self.teacher_feature_key].detach()
        if fs.shape[-2:] != ft.shape[-2:]:
            raise ValueError(
                f"YOLOKDLoss ожидает совпадающее пространственное разрешение фич "
                f"(тот же уровень FPN): student={tuple(fs.shape)}, teacher={tuple(ft.shape)}"
            )
        return F.smooth_l1_loss(fs, ft, reduction="mean")

    @staticmethod
    def _read(outputs: Any, key: str) -> torch.Tensor:
        if isinstance(outputs, dict) and key in outputs:
            value = outputs[key]
        elif (
            isinstance(outputs, (list, tuple))
            and len(outputs) == 2
            and isinstance(outputs[1], dict)
            and key in outputs[1]
        ):
            # YOLO в eval()-режиме (учитель в Option 1: yolov9c -> yolov9t)
            # отдаёт (decoded_predictions, raw_dict) вместо прямого dict,
            # который даёт train-режим — тот же raw_dict внутри, на [1].
            # Проверено эмпирически: те же ключи/формы boxes/scores/feats.
            value = outputs[1][key]
        else:
            value = getattr(outputs, key, None)
        if not isinstance(value, torch.Tensor):
            raise KeyError(f"YOLOKDLoss: не нашёл {key!r} в outputs ({type(outputs)})")
        return value
