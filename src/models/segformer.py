"""SegFormer (Xie et al., NeurIPS 2021, arXiv:2105.15203) — учитель сегментации.

Обёртка над transformers.SegformerForSemanticSegmentation, приводящая модель
к контракту пайплайна:

- forward(images) -> Tensor [B, num_classes, H, W] (а не SemanticSegmenterOutput
  с логитами в 1/4 разрешения), поэтому SegFormer взаимозаменяем с U-Net
  в SegmentationTrainer без правок тренера;
- признаки стадий энкодера выведены на канонические тапы taps.stage1..stage4,
  общие с моделями-учениками — это то, что делает возможной feature-дистилляцию
  между разными архитектурами (см. src/models/feature_taps.py).
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from src.models.feature_taps import STAGE_TAPS, FeatureTaps

# Спецификации MiT-энкодеров B0..B5 (Table 7 статьи; совпадают с конфигами
# nvidia/mit-b*). Захардкожены, чтобы вариант pretrained=null собирался
# полностью офлайн, без похода на Hugging Face за config.json.
SEGFORMER_VARIANTS: dict[str, dict] = {
    "b0": {"depths": [2, 2, 2, 2], "hidden_sizes": [32, 64, 160, 256], "decoder_hidden_size": 256},
    "b1": {"depths": [2, 2, 2, 2], "hidden_sizes": [64, 128, 320, 512], "decoder_hidden_size": 256},
    "b2": {"depths": [3, 4, 6, 3], "hidden_sizes": [64, 128, 320, 512], "decoder_hidden_size": 768},
    "b3": {"depths": [3, 4, 18, 3], "hidden_sizes": [64, 128, 320, 512], "decoder_hidden_size": 768},
    "b4": {"depths": [3, 8, 27, 3], "hidden_sizes": [64, 128, 320, 512], "decoder_hidden_size": 768},
    "b5": {"depths": [3, 6, 40, 3], "hidden_sizes": [64, 128, 320, 512], "decoder_hidden_size": 768},
}

# Общая часть конфига MiT — одинакова у всех вариантов, меняются только
# глубины и ширины стадий.
SEGFORMER_COMMON: dict = {
    "num_encoder_blocks": 4,
    "patch_sizes": [7, 3, 3, 3],
    "strides": [4, 2, 2, 2],
    "num_attention_heads": [1, 2, 5, 8],
    "sr_ratios": [8, 4, 2, 1],
    "mlp_ratios": [4, 4, 4, 4],
}

# Готовые чекпоинты на Hugging Face.
IMAGENET_REPO = "nvidia/mit-{variant}"
CITYSCAPES_REPO = "nvidia/segformer-{variant}-finetuned-cityscapes-1024-1024"


def segformer_config(variant: str, num_classes: int) -> SegformerConfig:
    if variant not in SEGFORMER_VARIANTS:
        raise ValueError(
            f"Неизвестный вариант SegFormer: {variant!r}. Доступны: {sorted(SEGFORMER_VARIANTS)}"
        )
    return SegformerConfig(
        num_labels=num_classes,
        **SEGFORMER_COMMON,
        **SEGFORMER_VARIANTS[variant],
    )


class SegFormer(nn.Module):
    """SegFormer-B{0..5} с логитами в разрешении входа.

    pretrained:
        None          — случайная инициализация целиком;
        "imagenet"    — энкодер MiT с ImageNet (nvidia/mit-b*), голова
                        декодера инициализируется случайно. Это штатный
                        рецепт статьи: SegFormer с нуля на Cityscapes
                        (2975 картинок) не обучается до вменяемого mIoU;
        "cityscapes"  — полностью дообученная модель nvidia/segformer-*-
                        finetuned-cityscapes-1024-1024 (учитель «из коробки»).
    """

    def __init__(
        self,
        variant: str = "b2",
        num_classes: int = 19,
        pretrained: str | None = "imagenet",
        cache_dir: str | None = None,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if variant not in SEGFORMER_VARIANTS:
            raise ValueError(
                f"Неизвестный вариант SegFormer: {variant!r}. Доступны: {sorted(SEGFORMER_VARIANTS)}"
            )

        self.variant = variant
        self.align_corners = align_corners
        self.hidden_sizes: Sequence[int] = SEGFORMER_VARIANTS[variant]["hidden_sizes"]

        if pretrained is None:
            self.model = SegformerForSemanticSegmentation(segformer_config(variant, num_classes))
        elif pretrained == "imagenet":
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                IMAGENET_REPO.format(variant=variant),
                num_labels=num_classes,
                cache_dir=cache_dir,
                # id2label в nvidia/mit-* описывает 1000 классов ImageNet,
                # а голова у нас на num_classes: без этого флага загрузка падает.
                ignore_mismatched_sizes=True,
            )
        elif pretrained == "cityscapes":
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                CITYSCAPES_REPO.format(variant=variant),
                num_labels=num_classes,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=True,
            )
        else:
            raise ValueError(
                f"pretrained должен быть None | 'imagenet' | 'cityscapes', получено {pretrained!r}"
            )

        self.taps = FeatureTaps(STAGE_TAPS)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=images, output_hidden_states=True)

        # Прогон через тапы нужен ради побочного эффекта — срабатывания
        # forward-хуков FeatureExtractor. Возвращаемые значения не нужны:
        # признаки уже потреблены головой декодера внутри self.model.
        for name, hidden_state in zip(STAGE_TAPS, outputs.hidden_states):
            self.taps.tap(name, hidden_state)

        # Голова SegFormer отдаёт логиты в 1/4 разрешения — апсемплим сами,
        # чтобы наружу модель выглядела как обычный сегментатор.
        return F.interpolate(
            outputs.logits,
            size=images.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
