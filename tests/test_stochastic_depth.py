"""Stochastic depth: где включается, что при этом не ломается."""

from functools import partial

import pytest
import torch
from torch import nn
from torchvision import models as tv_models

from src.models import UNet, apply_stochastic_depth, linear_drop_path_rates
from src.models.factory import torchvision_model_for_classification, unet_for_segmentation
from src.models.stochastic_depth import StochasticDepthBatchNorm2d


class TestRateSchedule:
    def test_linear_schedule_starts_at_zero_and_ends_at_the_rate(self):
        rates = linear_drop_path_rates(4, 0.3)
        assert rates[0] == 0.0
        assert rates[-1] == pytest.approx(0.3)
        assert rates == sorted(rates)

    def test_single_block_gets_the_full_rate(self):
        assert linear_drop_path_rates(1, 0.2) == [0.2]

    def test_zero_blocks_raise(self):
        with pytest.raises(ValueError):
            linear_drop_path_rates(0, 0.1)


class TestApplyToResNet:
    def test_every_block_gets_a_drop_path(self):
        model = apply_stochastic_depth(tv_models.resnet18(weights=None), 0.2)
        patched = [m for m in model.modules() if isinstance(m, StochasticDepthBatchNorm2d)]
        # ResNet-18: 8 BasicBlock'ов.
        assert len(patched) == 8
        assert patched[0].drop_path.p == 0.0
        assert patched[-1].drop_path.p == pytest.approx(0.2)

    def test_bottleneck_networks_are_patched_at_bn3(self):
        model = apply_stochastic_depth(tv_models.resnet50(weights=None), 0.1)
        for block in model.layer1:
            assert isinstance(block.bn3, StochasticDepthBatchNorm2d)
            assert not isinstance(block.bn2, StochasticDepthBatchNorm2d)

    def test_state_dict_keys_are_unchanged(self):
        """Ключевое требование: чекпоинт модели, обученной со stochastic depth,
        обязан грузиться в модель без него (так ученик первого этапа становится
        учителем второго, load_state_dict идёт со strict=True)."""
        plain = tv_models.resnet18(weights=None)
        patched = apply_stochastic_depth(tv_models.resnet18(weights=None), 0.2)

        assert list(plain.state_dict()) == list(patched.state_dict())
        plain.load_state_dict(patched.state_dict())

    def test_batch_norm_state_is_carried_over(self):
        model = tv_models.resnet18(weights=None)
        with torch.no_grad():
            model.layer1[0].bn2.weight.fill_(0.7)
            model.layer1[0].bn2.running_mean.fill_(0.3)

        apply_stochastic_depth(model, 0.2)

        assert torch.allclose(model.layer1[0].bn2.weight, torch.full((64,), 0.7))
        assert torch.allclose(model.layer1[0].bn2.running_mean, torch.full((64,), 0.3))

    def test_eval_output_is_identical_to_the_unpatched_model(self):
        """На инференсе выключений нет вовсе — иначе метрика зависела бы
        от случайности."""
        torch.manual_seed(0)
        plain = tv_models.resnet18(weights=None).eval()
        patched = apply_stochastic_depth(tv_models.resnet18(weights=None), 0.5).eval()
        patched.load_state_dict(plain.state_dict())

        images = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            assert torch.allclose(plain(images), patched(images), atol=1e-6)

    def test_training_output_is_stochastic(self):
        torch.manual_seed(0)
        model = apply_stochastic_depth(tv_models.resnet18(weights=None), 0.9).train()
        images = torch.randn(4, 3, 64, 64)

        with torch.no_grad():
            outputs = [model(images) for _ in range(5)]

        assert any(not torch.allclose(outputs[0], other) for other in outputs[1:])

    def test_backward_still_reaches_the_student(self):
        torch.manual_seed(0)
        model = apply_stochastic_depth(tv_models.resnet18(weights=None), 0.1).train()
        model(torch.randn(2, 3, 64, 64)).sum().backward()

        assert model.conv1.weight.grad is not None
        assert model.conv1.weight.grad.abs().sum() > 0

    @pytest.mark.parametrize("kwargs", [{"drop_path_rate": 1.0}, {"mode": "cosine"}])
    def test_invalid_arguments_raise(self, kwargs):
        with pytest.raises(ValueError):
            apply_stochastic_depth(tv_models.resnet18(weights=None), **{"drop_path_rate": 0.1, **kwargs})

    def test_uniform_mode_gives_every_block_the_same_rate(self):
        model = apply_stochastic_depth(tv_models.resnet18(weights=None), 0.15, mode="uniform")
        rates = {
            m.drop_path.p for m in model.modules() if isinstance(m, StochasticDepthBatchNorm2d)
        }
        assert rates == {0.15}


class TestUnsupportedArchitectures:
    def test_model_without_residual_blocks_raises(self):
        """Молчаливый no-op здесь опаснее ошибки: конфиг с drop_path_rate=0.1
        выглядел бы рабочим, а регуляризации в обучении не было бы."""
        with pytest.raises(ValueError, match="residual-блоков"):
            apply_stochastic_depth(UNet(num_classes=4, base_channels=8, depth=2), 0.1)

    def test_shufflenet_raises(self):
        with pytest.raises(ValueError, match="residual-блоков"):
            apply_stochastic_depth(tv_models.shufflenet_v2_x0_5(weights=None), 0.1)

    def test_non_batchnorm_residual_tail_raises(self):
        """Точка вставки drop-path выведена из устройства блока с BatchNorm.
        Для нестандартного norm_layer она неизвестна, и угадывать её нельзя."""
        model = tv_models.resnet18(weights=None, norm_layer=partial(nn.GroupNorm, 8))
        with pytest.raises(TypeError, match="BatchNorm2d"):
            apply_stochastic_depth(model, 0.1)


class TestFactoryWiring:
    def test_torchvision_factory_enables_stochastic_depth(self):
        model = torchvision_model_for_classification(
            "resnet18", num_classes=10, drop_path_rate=0.1
        )
        assert any(isinstance(m, StochasticDepthBatchNorm2d) for m in model.modules())

    def test_torchvision_factory_default_is_off(self):
        model = torchvision_model_for_classification("resnet18", num_classes=10)
        assert not any(isinstance(m, StochasticDepthBatchNorm2d) for m in model.modules())


class TestUNetDropout:
    def test_dropout_adds_no_state(self):
        """Dropout2d не имеет ни параметров, ни буферов — чекпоинты U-Net,
        обученного с dropout и без, обязаны оставаться взаимозаменяемыми."""
        plain = unet_for_segmentation(variant="tiny", num_classes=4)
        regularized = unet_for_segmentation(variant="tiny", num_classes=4, dropout=0.2)

        assert list(plain.state_dict()) == list(regularized.state_dict())
        plain.load_state_dict(regularized.state_dict())

    def test_eval_output_is_deterministic(self):
        torch.manual_seed(0)
        model = unet_for_segmentation(variant="tiny", num_classes=4, dropout=0.5).eval()
        images = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            assert torch.allclose(model(images), model(images))

    def test_training_output_is_stochastic(self):
        torch.manual_seed(0)
        model = unet_for_segmentation(variant="tiny", num_classes=4, dropout=0.5).train()
        images = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            outputs = [model(images) for _ in range(5)]

        assert any(not torch.allclose(outputs[0], other) for other in outputs[1:])

    def test_tap_is_taken_before_dropout(self):
        """Тап stage3 у U-Net с depth=4 — это боттлнек, и по нему идёт
        feature-дистилляция. Сравнивать с учителем надо чистую карту,
        а не ту, где случайные каналы обнулены."""
        from src.models import FeatureExtractor

        torch.manual_seed(0)
        model = unet_for_segmentation(variant="tiny", num_classes=4, dropout=0.9).train()
        extractor = FeatureExtractor(model, ["taps.stage3"])
        try:
            model(torch.randn(1, 3, 32, 32))
            feature = extractor.features["taps.stage3"]
            zero_channels = (feature.abs().sum(dim=(2, 3)) == 0).float().mean().item()
            assert zero_channels < 0.5, "в тап попали обнулённые dropout'ом каналы"
        finally:
            extractor.remove()

    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError):
            UNet(num_classes=4, dropout=1.0)
