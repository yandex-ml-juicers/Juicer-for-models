"""Мультимасштабный прогон сегментатора — для учителя дистилляции.

Стандартный приём оценки (mmsegmentation зовёт его slide/whole + ms+flip):
кадр прогоняется в нескольких масштабах и с отражением, предсказания
усредняются по ВЕРОЯТНОСТЯМ и возвращаются на исходную сетку. На Cityscapes
это стабильно даёт +1..2 mIoU без всякого дообучения — просто потому, что
разные масштабы ошибаются в разных местах.

Для дистилляции это ровно то, что нужно: цена — несколько лишних forward'ов
учителя (он и так под no_grad), а таргет становится заметно чище.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class MultiScaleInference(nn.Module):
    """Обёртка над сегментатором: усредняет предсказания по масштабам и flip'у.

    Наружу отдаёт ЛОГИТЫ (логарифм усреднённой вероятности), а не вероятности:
    так модель остаётся взаимозаменяемой с обычной, и softmax внутри любого
    KD-лосса вернёт именно усреднённое распределение. Константа, на которую
    log p отличается от «настоящих» логитов, softmax'у безразлична; при
    temperature != 1 масштаб всё же меняется — это цена приёма, и она
    компенсируется подбором температуры.

    Масштаб 1.0 прогоняется ПОСЛЕДНИМ и без отражения — намеренно: forward-хуки
    FeatureExtractor срабатывают на каждом прогоне, и в карты признаков
    (их снимают feature-based лоссы) обязан попасть кадр в родном разрешении
    и без зеркала, иначе они разъехались бы с ученическими.

    Args:
        model: сегментатор, отдающий логиты [B, C, H, W] в разрешении входа.
        scales: множители разрешения. 1.0 добавляется автоматически.
        flip: добавлять ли к каждому масштабу горизонтально отражённый прогон.
    """

    def __init__(
        self,
        model: nn.Module,
        scales: Sequence[float] = (0.75, 1.0, 1.25),
        flip: bool = True,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        scales = [float(scale) for scale in scales]
        if any(scale <= 0 for scale in scales):
            raise ValueError(f"scales должны быть > 0, получено {scales}")

        # module: под этим именем обёртку снимает unwrap_model, поэтому имена
        # тапов в конфигах лоссов ("taps.stage3") остаются теми же.
        self.module = model
        self.scales = [scale for scale in dict.fromkeys(scales) if scale != 1.0] + [1.0]
        self.flip = bool(flip)
        self.align_corners = align_corners

    def _predict(self, images: torch.Tensor, size: torch.Size) -> torch.Tensor:
        logits = self.module(images)
        if logits.shape[2:] != size:
            logits = F.interpolate(
                logits, size=size, mode="bilinear", align_corners=self.align_corners
            )
        return logits.float().softmax(dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        size = images.shape[2:]
        total: torch.Tensor | None = None
        passes = 0

        for scale in self.scales:
            if scale == 1.0:
                scaled = images
            else:
                scaled = F.interpolate(
                    images,
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=self.align_corners,
                    recompute_scale_factor=False,
                )

            # Отражение идёт первым, чтобы последним прогоном остался
            # неотражённый кадр — тот, чьи карты признаков увидят хуки.
            if self.flip:
                flipped = self._predict(scaled.flip(-1), size).flip(-1)
                total = flipped if total is None else total + flipped
                passes += 1

            probabilities = self._predict(scaled, size)
            total = probabilities if total is None else total + probabilities
            passes += 1

        return (total / passes).clamp_min(1e-8).log()
