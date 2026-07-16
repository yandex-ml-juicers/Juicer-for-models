import torch

from src.utils.metrics import AverageMeter, accuracy


def test_accuracy_perfect_and_zero():
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    assert accuracy(logits, torch.tensor([0, 1])) == 1.0
    assert accuracy(logits, torch.tensor([1, 0])) == 0.0


def test_accuracy_partial():
    logits = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert accuracy(logits, torch.tensor([0, 1, 1, 0])) == 0.5


def test_average_meter_is_sample_weighted():
    meter = AverageMeter()
    meter.update(1.0, n=3)  # полный батч
    meter.update(0.0, n=1)  # неполный последний батч
    assert meter.avg == 0.75  # (1*3 + 0*1) / 4, а не (1 + 0) / 2


def test_average_meter_empty_is_safe():
    assert AverageMeter().avg == 0.0
