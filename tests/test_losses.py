import pytest
import torch
import torch.nn.functional as F

from src.losses import CrossEntropy, FeatureKD, HintonKD


@pytest.fixture()
def batch():
    generator = torch.Generator().manual_seed(0)
    student_logits = torch.randn(8, 10, generator=generator)
    teacher_logits = torch.randn(8, 10, generator=generator)
    labels = torch.randint(0, 10, (8,), generator=generator)
    return student_logits, teacher_logits, labels


class TestCrossEntropy:
    def test_matches_functional(self, batch):
        student_logits, _, labels = batch
        criterion = CrossEntropy()
        result = criterion(student_logits, None, labels)
        assert torch.allclose(result["total"], F.cross_entropy(student_logits, labels))

    def test_does_not_require_teacher(self):
        assert CrossEntropy.requires_teacher is False
        assert CrossEntropy().required_features == ()


class TestHintonKD:
    def test_alpha_zero_is_pure_ce(self, batch):
        student_logits, teacher_logits, labels = batch
        criterion = HintonKD(temperature=4.0, alpha=0.0)
        result = criterion(student_logits, teacher_logits, labels)
        assert torch.allclose(result["total"], F.cross_entropy(student_logits, labels))

    def test_kd_term_zero_when_student_equals_teacher(self, batch):
        student_logits, _, labels = batch
        criterion = HintonKD(temperature=4.0, alpha=1.0)
        result = criterion(student_logits, student_logits.clone(), labels)
        assert result["kd"].abs().item() < 1e-6
        assert result["total"].abs().item() < 1e-6

    def test_blend(self, batch):
        student_logits, teacher_logits, labels = batch
        alpha = 0.7
        criterion = HintonKD(temperature=2.0, alpha=alpha)
        result = criterion(student_logits, teacher_logits, labels)
        expected = (1 - alpha) * result["ce"] + alpha * result["kd"]
        assert torch.allclose(result["total"], expected)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            HintonKD(alpha=1.5)

    def test_none_teacher_raises(self, batch):
        student_logits, _, labels = batch
        with pytest.raises(TypeError, match="requires_teacher"):
            HintonKD()(student_logits, None, labels)


class TestFeatureKD:
    LAYERS = {"block": {"student_channels": 4, "teacher_channels": 4, "weight": 1.0}}

    def _identity_criterion(self) -> FeatureKD:
        criterion = FeatureKD(layers=self.LAYERS, temperature=4.0)
        with torch.no_grad():
            criterion.adapters.adapters["block"].weight.copy_(torch.eye(4).reshape(4, 4, 1, 1))
        return criterion

    def test_adapters_are_trainable_parameters(self):
        criterion = FeatureKD(layers=self.LAYERS)
        params = list(criterion.parameters())
        assert params, "адаптеры должны попадать в criterion.parameters()"
        assert all(p.requires_grad for p in params)

    def test_declares_required_features(self):
        assert FeatureKD(layers=self.LAYERS).required_features == ("block",)

    def test_feature_loss_zero_for_identical_features(self, batch):
        student_logits, teacher_logits, labels = batch
        criterion = self._identity_criterion()
        features = {"block": torch.randn(8, 4, 8, 8)}
        result = criterion(
            student_logits,
            teacher_logits,
            labels,
            student_features=features,
            teacher_features={"block": features["block"].clone()},
        )
        assert result["feature"].abs().item() < 1e-6
        assert result["feature_block"].abs().item() < 1e-6

    def test_spatial_mismatch_is_interpolated(self, batch):
        student_logits, teacher_logits, labels = batch
        criterion = self._identity_criterion()
        result = criterion(
            student_logits,
            teacher_logits,
            labels,
            student_features={"block": torch.randn(8, 4, 4, 4)},
            teacher_features={"block": torch.randn(8, 4, 8, 8)},
        )
        assert torch.isfinite(result["total"])

    def test_missing_feature_raises(self, batch):
        student_logits, teacher_logits, labels = batch
        criterion = FeatureKD(layers=self.LAYERS)
        with pytest.raises(KeyError, match="block"):
            criterion(
                student_logits,
                teacher_logits,
                labels,
                student_features={},
                teacher_features={},
            )

    def test_total_is_weighted_sum(self, batch):
        student_logits, teacher_logits, labels = batch
        criterion = FeatureKD(
            layers=self.LAYERS, temperature=4.0, ce_weight=0.5, logits_weight=2.0, feature_weight=3.0
        )
        features = {"block": torch.randn(8, 4, 8, 8)}
        result = criterion(
            student_logits,
            teacher_logits,
            labels,
            student_features=features,
            teacher_features={"block": torch.randn(8, 4, 8, 8)},
        )
        expected = 0.5 * result["ce"] + 2.0 * result["kd"] + 3.0 * result["feature"]
        assert torch.allclose(result["total"], expected)
