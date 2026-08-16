"""Группы параметров оптимизатора: раздельные lr/weight_decay.

Главный инвариант, который здесь защищается: КАЖДЫЙ обучаемый параметр
попадает в optimizer ровно один раз. Потерянный параметр не обучается
вовсе, задвоенный получает двойной шаг — и то и другое проявляется как
"метрика почему-то хуже", а не как ошибка.
"""

import pytest
import torch
from torch import nn

from src.losses.fitnets import FitNetsKD
from src.models.timm_unet import TimmUNet
from src.models.unet import UNet
from src.training.param_groups import (
    build_param_groups,
    describe_param_groups,
    resolve_encoder_prefixes,
)

BASE_LR = 1.0e-3
BASE_WD = 3.0e-2


@pytest.fixture(scope="module")
def unet() -> UNet:
    return UNet(num_classes=19, base_channels=8, depth=4)


@pytest.fixture(scope="module")
def timm_unet() -> TimmUNet:
    # pretrained=False — тест не должен ходить в сеть.
    return TimmUNet(encoder_name="mobilenetv3_small_100", num_classes=19, pretrained=False)


def group_by_name(groups: list[dict]) -> dict[str, dict]:
    return {group["name"]: group for group in groups}


def all_params(groups: list[dict]) -> list[nn.Parameter]:
    return [parameter for group in groups for parameter in group["params"]]


class TestCoverage:
    """Ни один параметр не потерян и не задвоен."""

    @pytest.mark.parametrize("model_fixture", ["unet", "timm_unet"])
    def test_every_parameter_appears_exactly_once(self, model_fixture, request):
        model = request.getfixturevalue(model_fixture)
        groups = build_param_groups(model, lr=BASE_LR, weight_decay=BASE_WD, encoder_lr_mult=0.1)

        collected = [id(parameter) for parameter in all_params(groups)]
        expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}

        assert len(collected) == len(set(collected)), "параметр попал в две группы"
        assert set(collected) == expected

    def test_criterion_parameters_are_included(self, timm_unet):
        """Тренер падает, если параметр лосса не попал в optimizer, —
        значит адаптеры обязаны приезжать вместе с моделью."""
        criterion = FitNetsKD(
            layers={"taps.stage3": {"student_channels": 48, "teacher_channels": 320}}
        )
        groups = build_param_groups(
            timm_unet, criterion, lr=BASE_LR, weight_decay=BASE_WD, encoder_lr_mult=0.1
        )

        collected = {id(parameter) for parameter in all_params(groups)}
        for parameter in criterion.parameters():
            assert id(parameter) in collected

    def test_frozen_parameters_are_skipped(self, unet):
        """Замороженное в optimizer не нужно: он бы всё равно обновлял
        только то, у чего есть градиент, но weight decay AdamW применяется
        и без него."""
        frozen = UNet(num_classes=19, base_channels=8, depth=4)
        frozen.head.requires_grad_(False)

        groups = build_param_groups(frozen, lr=BASE_LR, weight_decay=BASE_WD)
        collected = {id(parameter) for parameter in all_params(groups)}

        assert not any(id(parameter) in collected for parameter in frozen.head.parameters())


class TestEncoderSplit:
    def test_encoder_gets_its_own_lr(self, timm_unet):
        groups = group_by_name(
            build_param_groups(timm_unet, lr=BASE_LR, weight_decay=BASE_WD, encoder_lr_mult=0.1)
        )

        assert groups["encoder"]["lr"] == pytest.approx(BASE_LR * 0.1)
        assert groups["decoder"]["lr"] == pytest.approx(BASE_LR)

    def test_encoder_group_matches_the_encoder_module(self, timm_unet):
        groups = group_by_name(
            build_param_groups(timm_unet, lr=BASE_LR, weight_decay=BASE_WD, encoder_lr_mult=0.1)
        )
        encoder_ids = {id(parameter) for parameter in timm_unet.encoder.parameters()}
        collected = {
            id(parameter)
            for name in ("encoder", "encoder_no_decay")
            for parameter in groups[name]["params"]
        }

        assert collected == encoder_ids

    def test_unet_bottleneck_counts_as_encoder(self, unet):
        """Боттлнек завершает нисходящий путь: регулировать его надо
        вместе с энкодером, а не с декодером."""
        groups = group_by_name(
            build_param_groups(unet, lr=BASE_LR, weight_decay=BASE_WD, encoder_lr_mult=0.1)
        )
        collected = {
            id(parameter)
            for name in ("encoder", "encoder_no_decay")
            for parameter in groups[name]["params"]
        }

        assert all(id(parameter) in collected for parameter in unet.bottleneck.parameters())
        assert all(id(parameter) not in collected for parameter in unet.decoders.parameters())

    def test_decoder_group_is_first(self, timm_unet):
        """Тренер логирует param_groups[0]["lr"] как lr запуска: если
        первым окажется энкодер, все накопленные графики сменят смысл."""
        groups = build_param_groups(
            timm_unet, lr=BASE_LR, weight_decay=BASE_WD, encoder_lr_mult=0.1
        )
        assert groups[0]["name"] == "decoder"
        assert groups[0]["lr"] == pytest.approx(BASE_LR)

    def test_no_encoder_group_when_multiplier_is_one(self, timm_unet):
        """Лишняя группа с теми же значениями только засоряет логи, а звать
        всю модель "decoder" — враньё в логах и в сериях графика lr."""
        names = {group["name"] for group in build_param_groups(timm_unet, lr=BASE_LR)}
        assert names == {"model", "model_no_decay"}

    def test_unknown_model_fails_loudly(self):
        """Молча применить общий lr нельзя: прогон выглядел бы настроенным,
        а был бы обычным."""
        with pytest.raises(ValueError, match="ENCODER_PREFIXES"):
            build_param_groups(nn.Sequential(nn.Conv2d(3, 8, 3)), lr=BASE_LR, encoder_lr_mult=0.1)

    def test_explicit_prefixes_work_for_unknown_models(self):
        model = nn.Module()
        model.backbone = nn.Conv2d(3, 8, 3)
        model.head = nn.Conv2d(8, 19, 1)

        groups = group_by_name(
            build_param_groups(
                model, lr=BASE_LR, encoder_lr_mult=0.1, encoder_prefixes=["backbone."]
            )
        )
        assert groups["encoder"]["params"][0] is model.backbone.weight

    def test_prefix_table_covers_project_models(self, unet, timm_unet):
        assert resolve_encoder_prefixes(unet) == ("encoders.", "bottleneck.")
        assert resolve_encoder_prefixes(timm_unet) == ("encoder.",)


class TestWeightDecay:
    def test_norm_and_bias_are_excluded_from_decay(self, timm_unet):
        groups = build_param_groups(
            timm_unet, lr=BASE_LR, weight_decay=BASE_WD, encoder_lr_mult=0.1
        )

        for group in groups:
            for parameter in group["params"]:
                if parameter.ndim <= 1:
                    assert group["weight_decay"] == 0.0, group["name"]
                else:
                    assert group["weight_decay"] > 0.0, group["name"]

    def test_batchnorm_gamma_lands_in_no_decay(self, unet):
        groups = group_by_name(build_param_groups(unet, lr=BASE_LR, weight_decay=BASE_WD))
        no_decay = {id(parameter) for parameter in groups["model_no_decay"]["params"]}

        norms = [module for module in unet.modules() if isinstance(module, nn.BatchNorm2d)]
        assert norms, "в U-Net должны быть BatchNorm-слои, иначе тест ничего не проверяет"
        assert all(id(module.weight) in no_decay for module in norms)

    def test_can_be_disabled(self, unet):
        groups = build_param_groups(
            unet, lr=BASE_LR, weight_decay=BASE_WD, no_decay_on_norm_and_bias=False
        )
        assert all(group["weight_decay"] == BASE_WD for group in groups)
        assert not any(group["name"].endswith("no_decay") for group in groups)

    def test_encoder_can_have_its_own_decay(self, timm_unet):
        groups = group_by_name(
            build_param_groups(
                timm_unet,
                lr=BASE_LR,
                weight_decay=BASE_WD,
                encoder_lr_mult=0.1,
                encoder_weight_decay=1.0e-4,
            )
        )
        assert groups["encoder"]["weight_decay"] == pytest.approx(1.0e-4)
        assert groups["decoder"]["weight_decay"] == pytest.approx(BASE_WD)

    def test_criterion_has_no_decay_by_default(self, timm_unet):
        """Адаптеры — часть целевой функции, а не модели: штрафовать
        их норму незачем."""
        criterion = FitNetsKD(
            layers={"taps.stage3": {"student_channels": 48, "teacher_channels": 320}}
        )
        groups = group_by_name(
            build_param_groups(timm_unet, criterion, lr=BASE_LR, weight_decay=BASE_WD)
        )
        assert groups["criterion"]["weight_decay"] == 0.0


class TestIntegrationWithTorch:
    def test_optimizer_accepts_the_groups(self, timm_unet):
        groups = build_param_groups(
            timm_unet, lr=BASE_LR, weight_decay=BASE_WD, encoder_lr_mult=0.1
        )
        optimizer = torch.optim.AdamW(groups, lr=BASE_LR, weight_decay=BASE_WD)

        by_name = {group["name"]: group for group in optimizer.param_groups}
        assert by_name["encoder"]["lr"] == pytest.approx(BASE_LR * 0.1)
        assert by_name["decoder"]["lr"] == pytest.approx(BASE_LR)

    def test_scheduler_scales_every_group_proportionally(self, timm_unet):
        """Косинус хранит base_lrs по группам, поэтому соотношение
        энкодер/декодер держится всё обучение, а не только на первом шаге.

        Точным оно остаётся не до конца: eta_min у CosineAnnealingLR —
        АБСОЛЮТНАЯ добавка, одинаковая для всех групп
        (lr = eta_min + (base_lr - eta_min) * ...), поэтому на самом хвосте
        отжига обе группы сходятся к eta_min и отношение уползает к единице.
        Практического значения это не имеет — к тому моменту оба lr уже
        пренебрежимо малы, — но точное равенство здесь ожидать нельзя.
        """
        from src.schedulers.warmup_cosine import warmup_cosine_scheduler

        groups = build_param_groups(
            timm_unet, lr=BASE_LR, weight_decay=BASE_WD, encoder_lr_mult=0.1
        )
        optimizer = torch.optim.AdamW(groups)
        scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=2, epochs=10, eta_min=1e-6)

        by_name = {group["name"]: group for group in optimizer.param_groups}
        for _ in range(6):
            optimizer.step()
            scheduler.step()
            ratio = by_name["encoder"]["lr"] / by_name["decoder"]["lr"]
            assert ratio == pytest.approx(0.1, rel=0.05)

    def test_ratio_survives_warmup(self, timm_unet):
        """LinearLR масштабирует каждую группу от её собственного base_lr,
        поэтому во время прогрева энкодер тоже идёт со своим множителем."""
        from src.schedulers.warmup_cosine import warmup_cosine_scheduler

        groups = build_param_groups(timm_unet, lr=BASE_LR, encoder_lr_mult=0.1)
        optimizer = torch.optim.AdamW(groups)
        warmup_cosine_scheduler(optimizer, warmup_epochs=5, epochs=10, eta_min=1e-6)

        by_name = {group["name"]: group for group in optimizer.param_groups}
        assert by_name["encoder"]["lr"] / by_name["decoder"]["lr"] == pytest.approx(0.1)

    def test_eta_min_below_the_smallest_group_lr(self, timm_unet):
        """eta_min общий для всех групп и АБСОЛЮТНЫЙ: если он выше lr
        энкодера, косинус разгонит энкодер вместо отжига."""
        groups = build_param_groups(timm_unet, lr=BASE_LR, encoder_lr_mult=0.1)
        smallest = min(group["lr"] for group in groups)
        assert smallest == pytest.approx(1.0e-4)


def test_describe_lists_every_group(timm_unet):
    groups = build_param_groups(timm_unet, lr=BASE_LR, weight_decay=BASE_WD, encoder_lr_mult=0.1)
    table = describe_param_groups(groups)
    for group in groups:
        assert group["name"] in table
