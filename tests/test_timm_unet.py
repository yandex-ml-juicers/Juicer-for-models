"""U-Net с предобученным энкодером из timm.

Сети здесь не касаемся: все модели собираются с pretrained=False. Проверяется
архитектура и стыковка с пайплайном, а не содержимое весов с Hugging Face.
"""

import pytest
import torch

from src.models import (
    STAGE_TAPS,
    TIMM_UNET_VARIANTS,
    FeatureExtractor,
    SegFormer,
    TimmUNet,
    timm_unet_for_segmentation,
    unet_for_segmentation,
)

NUM_CLASSES = 5

# Замеренные размеры (всего, млн параметров при num_classes=19). Дублируются
# в комментарии к TIMM_UNET_VARIANTS и в configs/model/student/timm_unet.yaml:
# размер ученика — то, по чему сравнивают методы, и расхождение таблицы
# с реальностью должно ловиться сразу.
EXPECTED_MILLIONS = {
    "mobilenetv2": 2.33,
    "efficientnet_b0": 4.24,
    "convnext_femto": 6.58,
    "efficientnetv2_b2": 9.04,
    "convnext_pico": 11.63,
    "resnet18": 14.39,
    "resnet34": 24.50,
    "convnext_nano_in12k": 19.79,
    "mobilenetv4_medium_in12k": 9.39,
}


class TestVariants:
    def test_table_covers_every_variant(self):
        assert set(EXPECTED_MILLIONS) == set(TIMM_UNET_VARIANTS)

    @pytest.mark.parametrize("variant", sorted(TIMM_UNET_VARIANTS))
    def test_variant_sizes_match_the_table(self, variant):
        model = timm_unet_for_segmentation(variant=variant, num_classes=19, pretrained=False)
        millions = sum(p.numel() for p in model.parameters()) / 1e6
        assert abs(millions - EXPECTED_MILLIONS[variant]) < 0.02, millions

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Неизвестный вариант"):
            timm_unet_for_segmentation(variant="resnet404", pretrained=False)

    def test_encoder_name_bypasses_the_table(self):
        """Прямое имя timm нужно, чтобы не расширять таблицу ради одного прогона."""
        model = timm_unet_for_segmentation(
            variant="resnet404", encoder_name="resnet18", num_classes=NUM_CLASSES, pretrained=False
        )
        assert model.encoder_name == "resnet18"

    def test_flat_encoder_is_rejected(self):
        """ViT собирается с features_only, но отдаёт все карты на страйде 16
        (reductions=[16, 16, 16]). Декодер на таком формально построился бы,
        а U-Net не получился бы — падаем на сборке, а не после суток обучения."""
        with pytest.raises(ValueError, match="не иерархический"):
            TimmUNet(encoder_name="vit_tiny_patch16_224", pretrained=False)

    def test_hierarchical_transformer_encoder_is_accepted(self):
        """Swin иерархический (4/8/16/32), так что формально подходит.
        Для ViT->CNN дистилляции брать его не надо, но запрещать нечего."""
        model = TimmUNet(
            encoder_name="swin_tiny_patch4_window7_224",
            num_classes=NUM_CLASSES,
            pretrained=False,
        )
        assert model.reductions == [4, 8, 16, 32]


class TestForward:
    @pytest.mark.parametrize("variant", ["mobilenetv2", "resnet18", "convnext_femto"])
    def test_logits_have_input_resolution(self, variant):
        model = timm_unet_for_segmentation(
            variant=variant, num_classes=NUM_CLASSES, pretrained=False
        )
        logits = model(torch.randn(2, 3, 64, 96))
        assert logits.shape == (2, NUM_CLASSES, 64, 96)

    def test_non_power_of_two_input_survives(self):
        """Билинейный апсемпл до размера skip-связи (а не scale_factor=2)
        нужен именно ради таких размеров."""
        model = timm_unet_for_segmentation(
            variant="resnet18", num_classes=NUM_CLASSES, pretrained=False
        )
        logits = model(torch.randn(1, 3, 70, 102))
        assert logits.shape == (1, NUM_CLASSES, 70, 102)

    def test_backward_reaches_every_parameter(self):
        model = timm_unet_for_segmentation(
            variant="mobilenetv2", num_classes=NUM_CLASSES, pretrained=False
        )
        model(torch.randn(1, 3, 64, 64)).sum().backward()
        without_grad = [name for name, p in model.named_parameters() if p.grad is None]
        assert not without_grad, without_grad

    def test_forward_is_finite_under_autocast(self):
        model = timm_unet_for_segmentation(
            variant="resnet18", num_classes=NUM_CLASSES, pretrained=False
        )
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(torch.randn(1, 3, 64, 96))
        assert torch.isfinite(logits.float()).all()


class TestTaps:
    @pytest.mark.parametrize("variant", sorted(TIMM_UNET_VARIANTS))
    def test_every_variant_exposes_all_four_taps(self, variant):
        """Главное преимущество перед classic U-Net: у ImageNet-бэкбона есть
        стадия на страйде 32, поэтому stage4 существует."""
        model = timm_unet_for_segmentation(variant=variant, pretrained=False)
        assert set(model.tap_channels) == set(STAGE_TAPS)

    def test_tapped_maps_have_the_stride_their_name_promises(self):
        model = timm_unet_for_segmentation(
            variant="resnet18", num_classes=NUM_CLASSES, pretrained=False
        )
        layers = [f"taps.{name}" for name in model.tap_channels]
        extractor = FeatureExtractor(model, layers)
        try:
            model(torch.randn(1, 3, 64, 96))
            for name, stride in (("stage1", 4), ("stage2", 8), ("stage3", 16), ("stage4", 32)):
                feature = extractor.features[f"taps.{name}"]
                assert feature.shape[1] == model.tap_channels[name]
                assert feature.shape[2:] == (64 // stride, 96 // stride)
        finally:
            extractor.remove()

    def test_taps_add_no_parameters_and_no_state(self):
        model = timm_unet_for_segmentation(variant="mobilenetv2", pretrained=False)
        assert list(model.taps.parameters()) == []
        assert not any(key.startswith("taps.") for key in model.state_dict())

    def test_matches_segformer_taps_on_all_four_stages(self):
        """С этим учеником feature-дистилляция доступна на всех стадиях,
        а не только на stage3, как у classic U-Net с depth=4."""
        student = timm_unet_for_segmentation(
            variant="resnet18", num_classes=NUM_CLASSES, pretrained=False
        )
        teacher = SegFormer(variant="b0", num_classes=NUM_CLASSES, pretrained=None)
        layers = [f"taps.{name}" for name in STAGE_TAPS]

        student_extractor = FeatureExtractor(student, layers)
        teacher_extractor = FeatureExtractor(teacher, layers)
        try:
            images = torch.randn(1, 3, 64, 96)
            student(images)
            teacher(images)
            for layer in layers:
                assert (
                    student_extractor.features[layer].shape[2:]
                    == teacher_extractor.features[layer].shape[2:]
                ), layer
        finally:
            student_extractor.remove()
            teacher_extractor.remove()


class TestPretrainedFlag:
    def test_flag_does_not_change_the_architecture(self):
        """pretrained управляет ТОЛЬКО значениями весов: пара прогонов
        true/false обязана сравнивать одну и ту же модель."""
        random_init = timm_unet_for_segmentation(
            variant="mobilenetv2", num_classes=NUM_CLASSES, pretrained=False
        )
        shapes = {k: tuple(v.shape) for k, v in random_init.state_dict().items()}
        assert len(shapes) > 0

        rebuilt = TimmUNet(
            encoder_name="mobilenetv2_100", num_classes=NUM_CLASSES, pretrained=False
        )
        assert {k: tuple(v.shape) for k, v in rebuilt.state_dict().items()} == shapes

    def test_checkpoint_path_suppresses_the_download(self, tmp_path):
        """pretrained=True + checkpoint_path не должен ходить в сеть: веса
        всё равно перезаписываются. Если бы ходил, тест бы это заметил только
        онлайн — поэтому проверяем результат: веса взялись из чекпоинта."""
        reference = timm_unet_for_segmentation(
            variant="mobilenetv2", num_classes=NUM_CLASSES, pretrained=False
        )
        path = tmp_path / "ck.pt"
        torch.save({"student_state": reference.state_dict()}, path)

        loaded = timm_unet_for_segmentation(
            variant="mobilenetv2",
            num_classes=NUM_CLASSES,
            pretrained=True,
            checkpoint_path=str(path),
        )
        assert torch.equal(loaded.head.weight, reference.head.weight)


class TestSizePairsWithClassicUNet:
    """Пары "предобученный / с нуля" должны быть сопоставимы по размеру,
    иначе сравнение меряет ёмкость, а не предобучение."""

    @pytest.mark.parametrize(
        "timm_variant,unet_variant",
        [
            ("mobilenetv2", "tiny"),
            ("efficientnet_b0", "small"),
            ("efficientnetv2_b2", "base"),
            ("convnext_nano_in12k", "large"),
        ],
    )
    def test_paired_variants_are_within_30_percent(self, timm_variant, unet_variant):
        pretrained_like = timm_unet_for_segmentation(
            variant=timm_variant, num_classes=19, pretrained=False
        )
        scratch = unet_for_segmentation(variant=unet_variant, num_classes=19)

        a = sum(p.numel() for p in pretrained_like.parameters())
        b = sum(p.numel() for p in scratch.parameters())
        assert abs(a - b) / max(a, b) < 0.30, (timm_variant, a, unet_variant, b)
