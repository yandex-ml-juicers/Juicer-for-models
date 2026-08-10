"""SegNeXt (MSCAN + LightHamHead) и мультимасштабный прогон учителя.

Сети здесь нет: модели собираются со случайными весами.
"""

import pytest
import torch

from src.models import (
    SEGNEXT_VARIANTS,
    FeatureExtractor,
    MultiScaleInference,
    SegNeXt,
    unwrap_model,
)
from src.models.segnext import convert_mmseg_state_dict

NUM_CLASSES = 5

# Размеры из статьи (Table 6), млн параметров. Совпадение с точностью до
# десятых — самая надёжная проверка того, что архитектура воспроизведена
# верно и чужие веса в неё встанут.
EXPECTED_SIZES = {"t": 4.3, "s": 13.9, "b": 27.6, "l": 48.9}


@pytest.mark.parametrize("variant", sorted(SEGNEXT_VARIANTS))
def test_variant_sizes_match_the_paper(variant):
    model = SegNeXt(variant=variant, num_classes=19)
    millions = sum(p.numel() for p in model.parameters()) / 1e6
    assert abs(millions - EXPECTED_SIZES[variant]) < 0.2, millions


def test_forward_returns_logits_in_input_resolution():
    model = SegNeXt(variant="t", num_classes=NUM_CLASSES).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 96))
    assert logits.shape == (2, NUM_CLASSES, 64, 96)


def test_taps_are_available_for_feature_distillation():
    """Ради тапов модель и подключается к FitNets/HeteroAKD: имена общие
    с SegFormer и timm-U-Net, страйды 4/8/16/32."""
    model = SegNeXt(variant="t", num_classes=NUM_CLASSES).eval()
    extractor = FeatureExtractor(model, ["taps.stage1", "taps.stage4"])
    try:
        with torch.no_grad():
            model(torch.randn(1, 3, 64, 96))
        assert extractor.features["taps.stage1"].shape[1:] == (32, 16, 24)
        assert extractor.features["taps.stage4"].shape[1:] == (256, 2, 3)
    finally:
        extractor.remove()


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="SegNeXt"):
        SegNeXt(variant="xl")


class TestCheckpointConversion:
    def test_mmseg_names_are_translated(self):
        converted = convert_mmseg_state_dict(
            {
                "backbone.patch_embed1.proj.0.weight": torch.zeros(1),
                "decode_head.squeeze.conv.weight": torch.zeros(1),
                "auxiliary_head.conv_seg.weight": torch.zeros(1),
            }
        )
        assert set(converted) == {
            "encoder.patch_embed1.proj.0.weight",
            "decoder.squeeze.conv.weight",
        }

    def test_original_repository_wraps_the_depthwise_conv(self):
        """В репозитории SegNeXt (в отличие от mmsegmentation) свёртка
        внутри MLP лежит на уровень глубже."""
        converted = convert_mmseg_state_dict(
            {"backbone.block1.0.mlp.dwconv.dwconv.weight": torch.zeros(1)}
        )
        assert list(converted) == ["encoder.block1.0.mlp.dwconv.weight"]

    def test_bare_encoder_checkpoint_is_accepted(self):
        """Веса MSCAN с ImageNet лежат без префикса вовсе."""
        converted = convert_mmseg_state_dict({"block1.0.norm1.weight": torch.zeros(1)})
        assert list(converted) == ["encoder.block1.0.norm1.weight"]


class _StubSegmenter(torch.nn.Module):
    """Сегментатор-заглушка: логиты в разрешении входа, никакой случайности."""

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Conv2d(3, NUM_CLASSES, kernel_size=3, padding=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(images)


class TestMultiScaleInference:
    def test_output_is_a_log_probability_of_the_input_size(self):
        model = SegNeXt(variant="t", num_classes=NUM_CLASSES).eval()
        wrapped = MultiScaleInference(model, scales=[0.5, 1.0], flip=True)

        with torch.no_grad():
            logits = wrapped(torch.randn(2, 3, 64, 96))

        assert logits.shape == (2, NUM_CLASSES, 64, 96)
        assert torch.allclose(logits.exp().sum(dim=1), torch.ones(2, 64, 96), atol=1e-4)

    def test_single_scale_without_flip_keeps_the_prediction(self):
        """Один масштаб без отражения — это обычный forward, только
        выраженный в log-вероятностях: argmax обязан совпасть.

        Модель здесь — детерминированная заглушка, а не SegNeXt: у того
        внутри NMF со случайной инициализацией базисов, и два его прогона
        по одному кадру и так отличаются (см. src/models/segnext.py).
        """
        model = _StubSegmenter().eval()
        images = torch.randn(1, 3, 64, 96)

        with torch.no_grad():
            plain = model(images).argmax(dim=1)
            wrapped = MultiScaleInference(model, scales=[1.0], flip=False)(images).argmax(dim=1)

        assert torch.equal(plain, wrapped)

    def test_scale_one_runs_last_so_hooks_see_the_native_resolution(self):
        """Хуки срабатывают на каждом прогоне, и в картах признаков обязан
        остаться кадр родного размера — иначе feature-лоссы получат карту
        не того масштаба."""
        model = SegNeXt(variant="t", num_classes=NUM_CLASSES).eval()
        wrapped = MultiScaleInference(model, scales=[0.5, 1.0, 1.5], flip=True)

        extractor = FeatureExtractor(wrapped, ["taps.stage4"])
        try:
            with torch.no_grad():
                wrapped(torch.randn(1, 3, 64, 96))
            assert extractor.features["taps.stage4"].shape[2:] == (2, 3)
        finally:
            extractor.remove()

    def test_wrapper_is_transparent_for_layer_names(self):
        model = SegNeXt(variant="t", num_classes=NUM_CLASSES)
        assert unwrap_model(MultiScaleInference(model)) is model

    def test_scales_are_deduplicated_and_always_contain_one(self):
        model = SegNeXt(variant="t", num_classes=NUM_CLASSES)
        assert MultiScaleInference(model, scales=[1.0, 0.5, 1.0]).scales == [0.5, 1.0]
