import pytest
import torch
from omegaconf import OmegaConf

from src.data import build_dataloaders
from src.models.factory import cifar_resnet18
from src.utils.seed import seed_everything

FAKE_DATA_CFG = {
    "dataset": {
        "_target_": "src.data.datasets.fake_cifar10",
        "train_size": 64,
        "eval_size": 32,
        "num_classes": 10,
    },
    "num_classes": 10,
    "image_size": None,
    "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
    "loader": {"batch_size": 16, "num_workers": 0, "pin_memory": False, "persistent_workers": False},
}


@pytest.fixture(autouse=True)
def _restore_global_state():
    """Тесты дёргают глобальные рубильники PyTorch — возвращаем их на место."""
    yield
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def test_same_seed_same_tensors():
    seed_everything(42)
    a = torch.randn(64)
    seed_everything(42)
    b = torch.randn(64)
    assert torch.equal(a, b)


def test_same_seed_same_model_init():
    seed_everything(42)
    model_a = cifar_resnet18()
    seed_everything(42)
    model_b = cifar_resnet18()
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)


def test_different_seed_different_model_init():
    seed_everything(1)
    model_a = cifar_resnet18()
    seed_everything(2)
    model_b = cifar_resnet18()
    assert any(
        not torch.equal(p_a, p_b) for p_a, p_b in zip(model_a.parameters(), model_b.parameters())
    )


def test_deterministic_flags():
    seed_everything(42, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled() is True

    seed_everything(42, deterministic=False)
    assert torch.are_deterministic_algorithms_enabled() is False


def test_dataloader_order_is_reproducible():
    cfg = OmegaConf.create(FAKE_DATA_CFG)

    seed_everything(42)
    train_a, _ = build_dataloaders(cfg, seed=42)
    batches_a = [labels.clone() for _, labels in train_a]

    seed_everything(42)
    train_b, _ = build_dataloaders(cfg, seed=42)
    batches_b = [labels.clone() for _, labels in train_b]

    assert len(batches_a) == len(batches_b)
    for labels_a, labels_b in zip(batches_a, batches_b):
        assert torch.equal(labels_a, labels_b)
