"""Mask2Former (Cheng et al., CVPR 2022, arXiv:2112.01527) — учитель сегментации.

Обёртка над transformers.Mask2FormerForUniversalSegmentation, приводящая
mask-classification выход модели к контракту пайплайна:
forward(images) -> Tensor [B, num_classes, H, W] (как у SegFormer/SegNeXt/U-Net),
чтобы Mask2Former был взаимозаменяем с остальными сегментаторами в
SegmentationTrainer без правок тренера.

Mask2Former — НЕ пиксельный классификатор: голова декодера отдаёт
`num_queries` масок (`masks_queries_logits`, [B, Q, h, w]) и распределение по
классам для каждой маски (`class_queries_logits`, [B, Q, num_classes+1] —
"+1" это null-класс, маска "объект отсутствует"). Плотная карта — это
P(класс c в пикселе p) = sum_q P(класс c | маска q) * P(маска q активна в p),
то есть softmax по классам (без null) да сигмоида по маскам, свёрнутые
einsum'ом. Это ровно то же преобразование, что делает
`Mask2FormerImageProcessor.post_process_semantic_segmentation` — здесь оно
переписано под тензоры уже в разрешении входа (без промежуточного ресайза
в фиксированные 384x384, которым пользуется процессор ради батчинга разных
исходных размеров — нам он не нужен, наш пайплайн и так работает с
фиксированным крупом/eval-размером).

Дальше — тот же приём, что и в MultiScaleInference (src/models/multi_scale.py):
наружу отдаётся log(p), а не "сырые" логиты — softmax внутри любого KD-лосса
(Hinton/CWD/DIST/pixel-KD/BPKD) на этом log(p) вернёт то же самое p.

Только УЧИТЕЛЬ: поддерживается исключительно pretrained="cityscapes" (готовые
чекпоинты facebook/mask2former-swin-*-cityscapes-semantic, 19 классов из
коробки). Со случайной инициализацией / ImageNet-энкодером Mask2Former в
этом репозитории не собирается — по задаче не нужен (учеником не ставится),
а без этой ветки код и конфиг заметно проще.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Mask2FormerForUniversalSegmentation

from src.models.feature_taps import STAGE_TAP_STRIDES, STAGE_TAPS, FeatureTaps

# Готовые чекпоинты на Hugging Face, дообученные на Cityscapes (семантическая
# сегментация, 19 классов — не путать с одноимёнными *-panoptic/-instance).
# base заведён на Swin с ImageNet-21k энкодером (это в самом имени репозитория
# у HF), у остальных — ImageNet-1k.
MASK2FORMER_VARIANTS: dict[str, str] = {
    "tiny": "facebook/mask2former-swin-tiny-cityscapes-semantic",
    "small": "facebook/mask2former-swin-small-cityscapes-semantic",
    "base": "facebook/mask2former-swin-base-IN21k-cityscapes-semantic",
    "large": "facebook/mask2former-swin-large-cityscapes-semantic",
}

# Страйд -> имя канонической стадии, для сопоставления encoder_hidden_states
# в forward() ниже (там же объяснение, почему не жёстко по индексу списка).
_TAP_NAME_BY_STRIDE: dict[int, str] = {stride: name for name, stride in STAGE_TAP_STRIDES.items()}


def dense_log_probs_from_queries(
    class_queries_logits: torch.Tensor,
    masks_queries_logits: torch.Tensor,
    size: tuple[int, int],
    align_corners: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Query-выход Mask2Former -> лог-вероятности [B, num_classes, H, W].

    Вынесено из Mask2Former.forward() отдельной чистой функцией (без модели)
    ради unit-теста без скачивания весов: подать можно любые
    class_queries_logits/masks_queries_logits, например синтетические.

    Args:
        class_queries_logits: [B, Q, num_classes+1] (последний класс — null).
        masks_queries_logits: [B, Q, h, w] — любое разрешение, апсемплится до size.
        size: (H, W) разрешения выхода (обычно — разрешение входа модели).
    """
    masks_queries_logits = F.interpolate(
        masks_queries_logits, size=size, mode="bilinear", align_corners=align_corners
    )
    masks_probs = masks_queries_logits.sigmoid()
    # Null-класс (последний) — "эта маска ничему не соответствует", в
    # плотную карту не переносится.
    class_probs = class_queries_logits.softmax(dim=-1)[..., :-1]

    # P(класс c в пикселе p) = sum_q P(класс c | q) * P(маска q в p).
    probs = torch.einsum("bqc,bqhw->bchw", class_probs, masks_probs)
    # Сумма по q не даёт распределения (маски перекрываются/не покрывают весь
    # кадр) — нормируем по классам, чтобы softmax(log(p)) в лоссах ученика
    # вернул именно это p, а не что-то отмасштабированное.
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(eps)
    return probs.clamp_min(eps).log()


class Mask2Former(nn.Module):
    """Mask2Former с логитами (лог-вероятностями) в разрешении входа.

    Args:
        variant: tiny (47M) | small (69M) | base (107M, IN21k) | large (216M).
            Учитель гоняется forward'ом каждый шаг обучения ученика — крупные
            варианты заметно замедляют дистилляцию, выбирайте по бюджету GPU.
        pretrained: только "cityscapes" (см. докстринг модуля).
        eps: пол для log/деления при нормировке псевдо-вероятностей на
            вырожденных пикселях (сумма по классам близка к нулю).

    Тапы признаков (`taps.stage1..stage4`, для feature-based KD вроде FitNets/
    HeteroAKD) — из `encoder_hidden_states` Swin-энкодера (backbone модели).
    Страйд каждой карты определяется ПО ФАКТИЧЕСКОМУ размеру относительно
    входа (как у TimmUNet), а не жёстко зашит: если для какого-то варианта
    он не совпадёт с каноническими 4/8/16/32, тап просто не сработает в этом
    прогоне — а не подставит карту не с той стадии.
    """

    def __init__(
        self,
        variant: str = "tiny",
        num_classes: int = 19,
        pretrained: str | None = "cityscapes",
        cache_dir: str | None = None,
        align_corners: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if variant not in MASK2FORMER_VARIANTS:
            raise ValueError(
                f"Неизвестный вариант Mask2Former: {variant!r}. "
                f"Доступны: {sorted(MASK2FORMER_VARIANTS)}"
            )
        if pretrained != "cityscapes":
            raise ValueError(
                f"Mask2Former здесь — только учитель с готовыми весами: "
                f"pretrained должен быть 'cityscapes', получено {pretrained!r}. "
                f"Ученика/случайную инициализацию собрать нельзя — по задаче "
                f"не нужно, см. докстринг src/models/mask2former.py."
            )

        self.variant = variant
        self.align_corners = align_corners
        self.eps = eps

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            MASK2FORMER_VARIANTS[variant], cache_dir=cache_dir
        )

        if self.model.config.num_labels != num_classes:
            raise ValueError(
                f"Чекпоинт {MASK2FORMER_VARIANTS[variant]!r} обучен на "
                f"{self.model.config.num_labels} классах, а запрошено "
                f"num_classes={num_classes}. Готовых весов на другое число "
                f"классов нет."
            )

        self.taps = FeatureTaps(STAGE_TAPS)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=images, output_hidden_states=True)

        input_h = images.shape[2]
        for hidden_state in outputs.encoder_hidden_states:
            # Прогон через тап — ради побочного эффекта (хук FeatureExtractor).
            reduction = input_h // hidden_state.shape[2]
            name = _TAP_NAME_BY_STRIDE.get(reduction)
            if name is not None:
                self.taps.tap(name, hidden_state)

        return dense_log_probs_from_queries(
            outputs.class_queries_logits,
            outputs.masks_queries_logits,
            size=images.shape[2:],
            align_corners=self.align_corners,
            eps=self.eps,
        )
