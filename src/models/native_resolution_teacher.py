"""Учитель, которому вход апсемплится до разрешения, на котором его мерили/
дообучали, а выход даунсемплится обратно к разрешению кропа ученика.

Контекст (см. outputs/claude-analis/analysis.md): SegFormer-B5-cityscapes
дообучен nvidia на 1024x1024, а во время дистилляции учитель получает тот же
кроп, что и ученик (см. cityscapes_seg.yaml: crop_size: [512, 1024]) — вдвое
меньше по каждой стороне и с другим соотношением сторон (2:1 вместо 1:1).
Attention-энкодеры вроде MiT заметно чувствительнее CNN к разрешению входа
(нет свёрточного scale-инварианта, позиционная информация завязана на страйд
патч-мёрджинга), поэтому это не безобидная деталь: учитель на таком кропе
может отдавать заметно менее качественные (более шумные) таргеты, чем
показывает на eval.

MultiScaleInference (см. src/models/multi_scale.py) — родственный, но другой
приём: он ансамблирует несколько масштабов ВОКРУГ текущего разрешения входа
(например 0.75/1.0/1.25 от 512x1024), чего мало, чтобы дотянуться до
1024x1024/2048. NativeResolutionTeacher вместо этого один раз апсемплит вход
в scale_factor раз (по умолчанию 2.0 — ровно во сколько раз crop_size меньше
image_size в cityscapes_seg.yaml, то есть 512x1024 -> 1024x2048, тот же
масштаб, на котором учителя меряет scripts/eval.py eval_slot=teacher),
прогоняет forward и даунсемплит логиты обратно к исходному размеру ДО того,
как их увидит KD-лосс — пиксельная привязка к ученику (нужна для CWD/BPKD/
pixel-KD/DIST) не рвётся, бюджет и разрешение кропа СТУДЕНТА не меняются.

Дёшево скомпоновать с MultiScaleInference можно, обернув эту обёртку ею:
    MultiScaleInference(NativeResolutionTeacher(teacher), scales=[0.75, 1.0, 1.25])
— каждый масштаб MultiScaleInference заново апсемплится до родного
разрешения внутри NativeResolutionTeacher, лишнего кода не нужно (обе
обёртки соблюдают контракт "выход того же разрешения, что вход").

Важное ограничение: НЕ совместим с feature-based KD (FitNets/HeteroAKD,
required_features != ()) — карты, которые снимает FeatureExtractor, будут в
разрешении АПСЕМПЛЕННОГО входа и разъедутся с ученическими попиксельно.
Для логит-based методов (BPKD, CWD, DIST, pixel-KD, ванильный Hinton)
ограничение не действует — они карт не используют.
"""

import torch
import torch.nn.functional as F
from torch import nn


class NativeResolutionTeacher(nn.Module):
    """Апсемплит вход учителю, даунсемплит его логиты обратно.

    Args:
        model: сегментатор, отдающий логиты [B, C, H, W] в разрешении входа.
        scale_factor: во сколько раз увеличить вход перед прогоном через
            учителя. 2.0 — дефолт под cityscapes_seg.yaml (crop_size вдвое
            меньше image_size по каждой стороне); для другого датасета/кропа
            посчитайте свой (image_size / crop_size).
        align_corners: тот же флаг, что у остальных сегментаторов проекта —
            должен совпадать с align_corners самого учителя, иначе апсемпл
            входа и обратный даунсемпл логитов разъедутся с внутренними
            интерполяциями модели на доли пикселя.
    """

    def __init__(
        self,
        model: nn.Module,
        scale_factor: float = 2.0,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if scale_factor <= 0:
            raise ValueError(f"scale_factor должен быть > 0, получено {scale_factor}")
        # module: то же имя атрибута, что у MultiScaleInference — по нему
        # unwrap_model снимает обёртку (см. src/models/feature_extractor.py),
        # а FeatureExtractor кидает внятную ValueError "слой не найден",
        # если эту обёртку всё же скрестить с feature-based лоссом.
        self.module = model
        self.scale_factor = float(scale_factor)
        self.align_corners = align_corners

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        size = images.shape[2:]
        upscaled = F.interpolate(
            images,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=self.align_corners,
            recompute_scale_factor=False,
        )
        logits = self.module(upscaled)
        if logits.shape[2:] != size:
            logits = F.interpolate(
                logits, size=size, mode="bilinear", align_corners=self.align_corners
            )
        return logits
