"""ESPNetv2 — два варианта декодера поверх одного EESP-энкодера:
ESPNetV2 (наш UNetDoubleConv, декодер постоянен между студентами проекта) и
ESPNetV2Native (декодер оригинальной статьи, EESPNet_Seg — на порядок
легче, но stage4 у него физически нет).

Размеры/forward — со случайными весами. Загрузка официального
ImageNet-чекпоинта — отдельный тест, качает файл (~1-7MB на вариант) один
раз в data/weights/espnetv2/ и переиспользует его при повторных запусках.
"""

import pytest
import torch

from src.models import ESPNETV2_VARIANTS, ESPNetV2, ESPNetV2Native, FeatureExtractor
from src.models.espnetv2 import EESPNet, convert_espnetv2_state_dict

NUM_CLASSES = 5

# Энкодер-только (без classifier и без хвостовых level5.3/4, см. докстринг
# espnetv2.py) — то, что реально попадает в модель. Замерено тестом, само
# число проверяет одновременно и config-формулу (base/s/k), и то, что
# лишние ImageNet-classifier-слои не просочились в декодер.
EXPECTED_ENCODER_SIZES = {"s050": 0.145, "s100": 0.505, "s125": 0.766, "s150": 1.081, "s200": 1.873}


@pytest.mark.parametrize("variant", sorted(ESPNETV2_VARIANTS))
def test_encoder_sizes_match_the_scale_formula(variant):
    encoder = EESPNet(s=ESPNETV2_VARIANTS[variant]["s"])
    millions = sum(p.numel() for p in encoder.parameters()) / 1e6
    assert abs(millions - EXPECTED_ENCODER_SIZES[variant]) < 0.01, millions


def test_forward_returns_logits_in_input_resolution():
    model = ESPNetV2(variant="s050", num_classes=NUM_CLASSES).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 96))
    assert logits.shape == (2, NUM_CLASSES, 64, 96)


def test_taps_are_available_for_feature_distillation():
    """Ради тапов модель и подключается к FitNets/HeteroAKD: имена общие
    с SegFormer/SegNeXt/timm-U-Net, страйды 4/8/16/32."""
    model = ESPNetV2(variant="s050", num_classes=NUM_CLASSES).eval()
    assert model.tap_channels == {"stage1": 32, "stage2": 64, "stage3": 128, "stage4": 256}

    extractor = FeatureExtractor(model, ["taps.stage1", "taps.stage4"])
    try:
        with torch.no_grad():
            model(torch.randn(1, 3, 64, 96))
        assert extractor.features["taps.stage1"].shape[1:] == (32, 16, 24)
        assert extractor.features["taps.stage4"].shape[1:] == (256, 2, 3)
    finally:
        extractor.remove()


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="ESPNetv2"):
        ESPNetV2(variant="xl")


def test_channel_config_is_divisible_by_k_for_every_official_variant():
    """EESP делит nOut на k=4 веток без остатка (assert внутри __init__) —
    если формула config когда-нибудь поменяется, здесь а не в середине
    обучения."""
    for spec in ESPNETV2_VARIANTS.values():
        EESPNet(s=spec["s"])  # не должен упасть


class TestCheckpointConversion:
    def test_backbone_keys_get_the_encoder_prefix(self):
        converted = convert_espnetv2_state_dict({"level1.conv.weight": torch.zeros(1)})
        assert list(converted) == ["encoder.level1.conv.weight"]

    def test_classifier_is_dropped(self):
        converted = convert_espnetv2_state_dict(
            {"level1.conv.weight": torch.zeros(1), "classifier.weight": torch.zeros(1)}
        )
        assert list(converted) == ["encoder.level1.conv.weight"]

    def test_level5_classifier_expansion_tail_is_dropped(self):
        """level5.3/level5.4 расширяют ширину под 1000-классовый classifier
        (depthwise+groupwise conv) — сегментационному декодеру не нужны, в
        модели не инстанцируются вовсе, поэтому эти ключи должны выпасть,
        а не осесть как unexpected при strict=False."""
        converted = convert_espnetv2_state_dict(
            {
                "level5.2.module_act.weight": torch.zeros(1),  # последний реальный EESP-блок
                "level5.3.conv.weight": torch.zeros(1),
                "level5.4.bn.weight": torch.zeros(1),
            }
        )
        assert list(converted) == ["encoder.level5.2.module_act.weight"]


@pytest.mark.parametrize("variant", ["s050"])
def test_real_imagenet_checkpoint_loads_with_no_missing_encoder_keys(variant):
    """Официальный чекпоинт sacmehta/ESPNetv2 должен покрыть КАЖДЫЙ параметр
    энкодера без единого расхождения имени/формы — иначе порт архитектуры
    разошёлся с оригиналом (см. докстринг espnetv2.py)."""
    from src.models.factory import espnetv2_for_segmentation

    model = espnetv2_for_segmentation(variant=variant, num_classes=NUM_CLASSES, pretrained="imagenet")
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 64, 96))
    assert torch.isfinite(logits).all()


class TestESPNetV2Native:
    """Декодер оригинальной статьи (EESPNet_Seg) — см. докстринг ESPNetV2Native."""

    # И энкодер (level5/level5_0 удалены — не нужны без стадии 32), и декодер
    # (работает в num_classes-мерном пространстве, а не в широких каналах
    # энкодера) здесь заметно легче ESPNetV2 при том же variant — см. таблицу
    # в configs/model/student/espnetv2_native.yaml.
    EXPECTED_TOTAL_SIZES = {"s050": 0.099, "s100": 0.340, "s125": 0.515, "s150": 0.725, "s200": 1.253}

    @pytest.mark.parametrize("variant", sorted(ESPNETV2_VARIANTS))
    def test_sizes_are_an_order_of_magnitude_smaller_than_the_unet_decoder(self, variant):
        native = ESPNetV2Native(variant=variant, num_classes=19)
        hybrid = ESPNetV2(variant=variant, num_classes=19)
        native_millions = sum(p.numel() for p in native.parameters()) / 1e6
        hybrid_millions = sum(p.numel() for p in hybrid.parameters()) / 1e6

        assert abs(native_millions - self.EXPECTED_TOTAL_SIZES[variant]) < 0.01, native_millions
        assert native_millions < hybrid_millions

    def test_level5_is_not_part_of_the_model(self):
        """del self.encoder.level5* в __init__ — как в оригинале
        (EESPNet_Seg удаляет их за ненадобностью), а не просто не вызывается:
        иначе эти веса тихо жили бы в .parameters() мёртвым грузом."""
        model = ESPNetV2Native(variant="s050", num_classes=NUM_CLASSES)
        assert not hasattr(model.encoder, "level5")
        assert not hasattr(model.encoder, "level5_0")

    def test_forward_returns_logits_in_input_resolution(self):
        model = ESPNetV2Native(variant="s050", num_classes=NUM_CLASSES).eval()
        with torch.no_grad():
            logits = model(torch.randn(2, 3, 96, 160))
        assert logits.shape == (2, NUM_CLASSES, 96, 160)
        assert torch.isfinite(logits).all()

    def test_taps_stop_at_stage3_stage4_does_not_exist(self):
        """У этой модели физически нет страйда 32 (сегментации хватает 16,
        см. докстринг ESPNetV2Native) — в отличие от ESPNetV2, taps.stage4
        здесь не объявлен вовсе, как у классического UNet(depth=4)."""
        model = ESPNetV2Native(variant="s050", num_classes=NUM_CLASSES).eval()
        assert model.tap_channels == {"stage1": 32, "stage2": 64, "stage3": 128}
        assert set(model.taps.keys()) == {"stage1", "stage2", "stage3"}

        extractor = FeatureExtractor(model, ["taps.stage1", "taps.stage3"])
        try:
            with torch.no_grad():
                model(torch.randn(1, 3, 96, 160))
            assert extractor.features["taps.stage1"].shape[1:] == (32, 24, 40)
            assert extractor.features["taps.stage3"].shape[1:] == (128, 6, 10)
        finally:
            extractor.remove()

        with pytest.raises(ValueError, match="stage4"):
            FeatureExtractor(model, ["taps.stage4"])

    def test_unknown_variant_is_rejected(self):
        with pytest.raises(ValueError, match="ESPNetv2"):
            ESPNetV2Native(variant="xl")

    def test_real_imagenet_checkpoint_loads_with_no_missing_encoder_keys(self):
        """Тот же официальный чекпоинт, что у ESPNetV2 — level5/level5_0 в
        нём есть, но этой модели не нужны (лишние ключи молча отброшены)."""
        from src.models.factory import espnetv2_native_for_segmentation

        model = espnetv2_native_for_segmentation(variant="s050", num_classes=NUM_CLASSES, pretrained="imagenet")
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 96, 160))
        assert torch.isfinite(logits).all()
