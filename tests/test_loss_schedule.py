"""Расписание весов лосса по эпохам."""

import pytest
import torch

from src.losses import CompositeLoss, CrossEntropy, DiceLoss, PixelWiseKD
from src.training import LossWeightScheduler
from src.training.loss_schedule import WeightSchedule


class TestWeightSchedule:
    def test_linear_goes_from_start_to_end(self):
        schedule = WeightSchedule("w", "linear", start=1.0, end=0.0, start_epoch=1, end_epoch=11)
        assert schedule.value_at(1, 20) == 1.0
        assert schedule.value_at(6, 20) == pytest.approx(0.5)
        assert schedule.value_at(11, 20) == 0.0

    def test_value_is_held_outside_the_interval(self):
        schedule = WeightSchedule("w", "linear", start=1.0, end=0.0, start_epoch=5, end_epoch=10)
        assert schedule.value_at(1, 20) == 1.0
        assert schedule.value_at(50, 20) == 0.0

    def test_cosine_is_flat_at_the_ends(self):
        schedule = WeightSchedule("w", "cosine", start=1.0, end=0.0, start_epoch=0, end_epoch=10)
        # У косинуса производная на концах нулевая: вес почти не двигается
        # на первом шаге и резко идёт в середине.
        assert schedule.value_at(1, 10) > 0.95
        assert schedule.value_at(5, 10) == pytest.approx(0.5)

    def test_end_epoch_defaults_to_the_last_epoch_of_the_run(self):
        schedule = WeightSchedule("w", "linear", start=1.0, end=0.0, start_epoch=1)
        assert schedule.value_at(100, 100) == 0.0

    def test_constant_ignores_the_epoch(self):
        schedule = WeightSchedule("w", "constant", start=0.3)
        assert schedule.value_at(1, 10) == 0.3
        assert schedule.value_at(10, 10) == 0.3

    def test_unknown_schedule_is_reported(self):
        with pytest.raises(ValueError, match="расписание"):
            WeightSchedule("w", "exponential", start=1.0, end=0.0).value_at(5, 10)


class TestLossWeightScheduler:
    def test_sets_the_weight_on_the_loss(self):
        criterion = DiceLoss(ce_weight=1.0, dice_weight=1.0)
        scheduler = LossWeightScheduler(
            criterion,
            {"dice_weight": {"schedule": "linear", "start": 1.0, "end": 0.0}},
            total_epochs=10,
        )

        scheduler.step(10)
        assert criterion.dice_weight == 0.0

    def test_the_loss_actually_changes(self):
        """Проверка не на атрибут, а на значение: вес должен читаться
        в forward, иначе расписание меняло бы декорацию."""
        criterion = PixelWiseKD(alpha=0.5)
        logits = torch.randn(1, 3, 4, 4)
        labels = torch.randint(0, 3, (1, 4, 4))

        scheduler = LossWeightScheduler(
            criterion, {"alpha": {"schedule": "linear", "start": 1.0, "end": 0.0}}, total_epochs=10
        )
        scheduler.step(10)

        result = criterion(logits, torch.randn(1, 3, 4, 4), labels)
        assert torch.allclose(result["total"], result["ce"])

    def test_reaches_terms_of_a_composition(self):
        criterion = CompositeLoss(
            {"ce": CrossEntropy(), "seg": DiceLoss()}, weights={"ce": 1.0, "seg": 1.0}
        )
        scheduler = LossWeightScheduler(
            criterion,
            {
                "weights.seg": {"schedule": "linear", "start": 1.0, "end": 0.0},
                "losses.seg.dice_weight": {"schedule": "constant", "start": 0.5},
            },
            total_epochs=10,
        )

        scheduler.step(10)
        assert criterion.weights["seg"] == 0.0
        assert criterion.losses["seg"].dice_weight == 0.5

    def test_reported_values_are_named_for_the_log(self):
        criterion = DiceLoss()
        scheduler = LossWeightScheduler(
            criterion, {"dice_weight": {"schedule": "constant", "start": 0.4}}, total_epochs=5
        )
        assert scheduler.step(1) == {"loss_weight_dice_weight": 0.4}

    def test_typo_in_the_path_fails_at_construction(self):
        """Опечатка обязана всплыть на старте, а не через час обучения —
        молча менять несуществующий атрибут хуже, чем упасть."""
        with pytest.raises(ValueError, match="нет веса"):
            LossWeightScheduler(
                DiceLoss(), {"dise_weight": {"schedule": "constant", "start": 1.0}}, total_epochs=5
            )

    def test_non_numeric_target_is_rejected(self):
        with pytest.raises(TypeError):
            LossWeightScheduler(
                DiceLoss(), {"forward": {"schedule": "constant", "start": 1.0}}, total_epochs=5
            )
