"""Композиция Hydra-конфигов: каждый experiment собирается и согласован.

Ловит битые overrides, опечатки в ключах и рассинхрон "лосс требует учителя,
а учителя в конфиге нет" ещё до запуска обучения.
"""

import pytest
from hydra import compose, initialize
from hydra.utils import instantiate

from src.losses import DistillationLoss

EXPERIMENTS = ["b1_vanilla_kd", "b1_scratch", "b2_feature_kd", "b2_scratch"]


def compose_config(overrides: list[str]):
    with initialize(version_base="1.3", config_path="../configs"):
        return compose(config_name="config", overrides=overrides)


def test_default_config_composes():
    cfg = compose_config([])
    assert cfg.seed == 42
    assert cfg.model.teacher is not None
    assert cfg.trainer.epochs > 0


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_composes_and_is_consistent(experiment):
    cfg = compose_config([f"experiment={experiment}"])

    assert cfg.name != "default", "эксперимент обязан задавать своё имя запуска"
    assert cfg.trainer.epochs > 0

    criterion = instantiate(cfg.loss)
    assert isinstance(criterion, DistillationLoss)

    has_teacher = cfg.model.get("teacher") is not None
    assert criterion.requires_teacher == has_teacher, (
        "требование учителя у лосса должно совпадать с наличием model/teacher в конфиге"
    )


def test_feature_kd_declares_layers():
    cfg = compose_config(["experiment=b2_feature_kd"])
    criterion = instantiate(cfg.loss)
    assert criterion.required_features == ("layer1", "layer2", "layer3", "layer4")
    assert len(list(criterion.parameters())) == 4  # по одной 1x1-свёртке на слой


def test_scratch_experiments_have_no_teacher():
    for experiment in ["b1_scratch", "b2_scratch"]:
        cfg = compose_config([f"experiment={experiment}"])
        assert cfg.model.get("teacher") is None


def test_scheduler_t_max_follows_epochs():
    cfg = compose_config(["experiment=b1_vanilla_kd", "trainer.epochs=7"])
    assert cfg.scheduler.T_max == 7
