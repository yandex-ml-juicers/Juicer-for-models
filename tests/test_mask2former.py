"""Mask2Former: конвертация query-выхода в плотную карту + сборка модели.

Быстрая часть (без сети) — математика dense_log_probs_from_queries на
синтетических тензорах и валидация аргументов конструктора. Реальная загрузка
чекпоинта с Hugging Face и настоящий forward — в TestReal (маркер
`integration`, тянет ~180MB весов tiny-варианта при первом запуске):

    pytest tests/test_mask2former.py -m "not integration"   # без сети
    pytest tests/test_mask2former.py                        # всё, включая сеть
"""

import pytest
import torch

from src.models.feature_extractor import FeatureExtractor
from src.models.feature_taps import STAGE_TAP_STRIDES
from src.models.mask2former import MASK2FORMER_VARIANTS, Mask2Former, dense_log_probs_from_queries

NUM_CLASSES = 19


class TestDenseLogProbsFromQueries:
    """Чистая функция-конвертер — без модели, без сети."""

    def test_output_shape(self):
        batch, queries, classes, h, w = 2, 5, NUM_CLASSES, 4, 6
        class_logits = torch.randn(batch, queries, classes + 1)
        mask_logits = torch.randn(batch, queries, 8, 12)  # другое разрешение — апсемплится

        out = dense_log_probs_from_queries(class_logits, mask_logits, size=(h, w))
        assert out.shape == (batch, classes, h, w)

    def test_is_a_valid_log_distribution(self):
        """exp(out) обязано суммироваться в 1 по классам — это лог-вероятности,
        а не сырые логиты (softmax(out) должен вернуть то же самое, без
        повторного нормирования)."""
        class_logits = torch.randn(3, 7, NUM_CLASSES + 1)
        mask_logits = torch.randn(3, 7, 5, 5)

        out = dense_log_probs_from_queries(class_logits, mask_logits, size=(5, 5))
        probs = out.exp()
        assert torch.allclose(probs.sum(dim=1), torch.ones(3, 5, 5), atol=1e-4)
        # softmax(log p) == p, с точностью до пересчёта через logsumexp
        assert torch.allclose(out.softmax(dim=1), probs, atol=1e-4)

    def test_single_confident_query_dominates_its_class(self):
        """Одна маска с уверенным классом c и активная везде -> результат
        должен явно отдавать предпочтение классу c во всех пикселях."""
        batch, queries, h, w = 1, 3, 4, 4
        target_class = 2

        class_logits = torch.full((batch, queries, NUM_CLASSES + 1), -10.0)
        class_logits[0, 0, target_class] = 10.0   # запрос 0 уверенно "класс 2"
        class_logits[0, 1, -1] = 10.0              # запрос 1 уверенно "null"
        class_logits[0, 2, -1] = 10.0              # запрос 2 уверенно "null"

        mask_logits = torch.full((batch, queries, h, w), -10.0)
        mask_logits[0, 0] = 10.0  # запрос 0 активен везде

        out = dense_log_probs_from_queries(class_logits, mask_logits, size=(h, w))
        assert (out.argmax(dim=1) == target_class).all()

    def test_no_nan_or_inf_on_degenerate_input(self):
        """Все маски неактивны (сигмоида ~0 везде) — сумма по классам близка
        к нулю, но clamp_min(eps) обязан не дать NaN/inf."""
        class_logits = torch.randn(1, 4, NUM_CLASSES + 1)
        mask_logits = torch.full((1, 4, 3, 3), -50.0)

        out = dense_log_probs_from_queries(class_logits, mask_logits, size=(3, 3))
        assert torch.isfinite(out).all()


class TestConstructorValidation:
    """Проверки, срабатывающие ДО обращения к сети (variant/pretrained)."""

    def test_unknown_variant_is_rejected(self):
        with pytest.raises(ValueError, match="вариант"):
            Mask2Former(variant="nano")

    def test_only_cityscapes_pretrained_is_supported(self):
        with pytest.raises(ValueError, match="pretrained"):
            Mask2Former(pretrained="imagenet")

    def test_only_cityscapes_pretrained_rejects_none_too(self):
        with pytest.raises(ValueError, match="pretrained"):
            Mask2Former(pretrained=None)

    def test_variant_table_has_the_four_promised_sizes(self):
        assert set(MASK2FORMER_VARIANTS) == {"tiny", "small", "base", "large"}
        for repo in MASK2FORMER_VARIANTS.values():
            assert repo.startswith("facebook/mask2former-swin-")
            assert repo.endswith("-cityscapes-semantic")


@pytest.mark.integration
class TestReal:
    """Настоящая загрузка facebook/mask2former-swin-tiny-cityscapes-semantic
    с Hugging Face (кэшируется — повторные запуски сети не требуют)."""

    @pytest.fixture()
    def model(self):
        # from_pretrained кэширует веса локально, повторный вызов не лезет в
        # сеть — не страшно строить модель заново на каждый тест.
        return Mask2Former(variant="tiny", num_classes=NUM_CLASSES)

    def test_forward_returns_log_probs_in_input_resolution(self, model):
        images = torch.randn(1, 3, 128, 256)
        with torch.no_grad():
            out = model(images)

        assert out.shape == (1, NUM_CLASSES, 128, 256)
        assert torch.isfinite(out).all()
        probs = out.exp()
        assert torch.allclose(probs.sum(dim=1), torch.ones(1, 128, 256), atol=1e-3)

    def test_stage_taps_are_populated_at_canonical_strides(self, model):
        extractor = FeatureExtractor(model, [f"taps.{name}" for name in STAGE_TAP_STRIDES])
        try:
            images = torch.randn(1, 3, 128, 256)
            with torch.no_grad():
                model(images)

            assert set(extractor.features) == {f"taps.{name}" for name in STAGE_TAP_STRIDES}
            for name, stride in STAGE_TAP_STRIDES.items():
                feature = extractor.features[f"taps.{name}"]
                assert feature.shape[2] == 128 // stride
                assert feature.shape[3] == 256 // stride
        finally:
            extractor.remove()

    def test_num_labels_mismatch_with_the_checkpoint_is_rejected(self):
        """Чекпоинт tiny обучен на 19 классах — запрос другого числа обязан
        падать с понятной ошибкой, а не молча обрезать/дополнить голову."""
        with pytest.raises(ValueError, match="класс"):
            Mask2Former(variant="tiny", num_classes=5)
