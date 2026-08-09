"""Dice Loss: базовые инварианты."""

import pytest
import torch
import torch.nn.functional as F

from src.losses import DiceLoss

NUM_CLASSES = 4
IGNORE_INDEX = 255


@pytest.fixture()
def batch():
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(2, NUM_CLASSES, 8, 12, generator=generator)
    labels = torch.randint(0, NUM_CLASSES, (2, 8, 12), generator=generator)
    labels[0, 0, :4] = IGNORE_INDEX
    return logits, labels


def test_no_teacher_required():
    assert DiceLoss().requires_teacher is False


def test_total_is_weighted_sum(batch):
    logits, labels = batch
    result = DiceLoss(ce_weight=0.5, dice_weight=2.0, ignore_index=IGNORE_INDEX)(
        logits, None, labels
    )
    assert torch.allclose(result["total"], 0.5 * result["ce"] + 2.0 * result["dice"])


def test_perfect_prediction_drives_dice_to_zero(batch):
    """Уверенное совпадение с разметкой — Dice близок к 1, лосс к 0."""
    _, labels = batch
    confident = F.one_hot(labels.clamp(max=NUM_CLASSES - 1), NUM_CLASSES)
    confident = confident.permute(0, 3, 1, 2).float() * 50.0

    result = DiceLoss(ignore_index=IGNORE_INDEX)(confident, None, labels)
    assert result["dice"].item() < 1e-2


def test_worst_prediction_is_worse_than_random(batch):
    logits, labels = batch
    criterion = DiceLoss(ignore_index=IGNORE_INDEX)

    wrong = F.one_hot((labels.clamp(max=NUM_CLASSES - 1) + 1) % NUM_CLASSES, NUM_CLASSES)
    wrong = wrong.permute(0, 3, 1, 2).float() * 50.0

    assert criterion(wrong, None, labels)["dice"] > criterion(logits, None, labels)["dice"]


def test_ignore_index_excluded(batch):
    """Void-пиксели не должны влиять: портим предсказание ровно на них —
    значение лосса обязано остаться прежним."""
    logits, labels = batch
    criterion = DiceLoss(ignore_index=IGNORE_INDEX)
    before = criterion(logits, None, labels)["dice"]

    corrupted = logits.clone()
    corrupted[0, :, 0, :4] = 50.0

    assert torch.allclose(before, criterion(corrupted, None, labels)["dice"])


def test_all_ignored_gives_zero_dice_loss():
    """Кадр целиком из void: делить не на что, но и падать не на чем."""
    logits = torch.randn(1, NUM_CLASSES, 4, 4)
    labels = torch.full((1, 4, 4), IGNORE_INDEX)

    result = DiceLoss(ce_weight=0.0, ignore_index=IGNORE_INDEX)(logits, None, labels)
    assert torch.isfinite(result["dice"])
    assert result["dice"].abs().item() < 1e-6


def test_gradient_reaches_the_logits(batch):
    logits, labels = batch
    logits = logits.clone().requires_grad_(True)

    DiceLoss(ce_weight=0.0, ignore_index=IGNORE_INDEX)(logits, None, labels)["total"].backward()

    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_is_finite_and_fp32_under_autocast(batch):
    """Тренер зовёт лосс внутри autocast, а GradScaler масштабирует total —
    скаляр обязан выходить в fp32."""
    logits, labels = batch
    criterion = DiceLoss(ignore_index=IGNORE_INDEX)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = criterion(logits.to(torch.bfloat16), None, labels)

    for name, value in result.items():
        assert torch.isfinite(value), name
        assert value.dtype == torch.float32, f"{name}: {value.dtype}"


def test_invalid_smooth_raises():
    with pytest.raises(ValueError):
        DiceLoss(smooth=0.0)
