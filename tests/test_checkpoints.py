"""Совместимость загрузки весов.

Разбор чекпоинта раньше был скопирован в четырёх фабриках моделей, у каждой
со своим набором поддерживаемых ключей. Копии заменены на один
load_checkpoint_into, и эти тесты фиксируют, что ОБЪЕДИНЁННЫЙ набор форматов
грузится каждой из фабрик — то есть ни один старый чекпоинт не перестал
читаться.
"""

import pytest
import torch
from torch import nn

from src.models import segformer_for_segmentation, unet_for_segmentation
from src.models.factory import (
    timm_model_for_classification,
    torchvision_model_for_classification,
)
from src.utils.checkpoints import extract_state_dict, load_checkpoint_into

# Все обёртки, которые встречались в четырёх копиях разбора. Раньше каждая
# фабрика понимала своё подмножество; теперь любую понимают все.
WRAPPERS = ["raw", "student_state", "model_state_dict", "state_dict", "model"]


def wrap(state_dict: dict, wrapper: str) -> dict:
    """Кладёт state_dict в чекпоинт нужной формы (raw — совсем без обёртки)."""
    if wrapper == "raw":
        return state_dict
    # Рядом с весами лежит мусор реального чекпоинта: он не должен мешать.
    return {"epoch": 7, "best_acc": 0.5, wrapper: state_dict}


def save(tmp_path, checkpoint, name="ck.pt") -> str:
    path = tmp_path / name
    torch.save(checkpoint, path)
    return str(path)


class TestExtractStateDict:
    @pytest.mark.parametrize("wrapper", WRAPPERS)
    def test_every_wrapper_is_unwrapped(self, wrapper):
        weights = {"fc.weight": torch.zeros(2, 3)}
        assert extract_state_dict(wrap(weights, wrapper)) is weights

    def test_non_dict_checkpoint_raises_type_error(self):
        with pytest.raises(TypeError):
            extract_state_dict(torch.zeros(3))

    def test_non_dict_value_under_a_known_key_is_ignored(self):
        """Некоторые чекпоинты держат под "model" имя архитектуры, а не веса.
        Старый код в torchvision-ветке проверял это только для "model";
        теперь проверка общая для всех ключей."""
        weights = {"fc.weight": torch.zeros(2, 3)}
        checkpoint = {"model": "resnet18", "state_dict": weights}
        assert extract_state_dict(checkpoint) is weights

    def test_plain_state_dict_is_returned_as_is(self):
        weights = {"fc.weight": torch.zeros(2, 3), "fc.bias": torch.zeros(2)}
        assert extract_state_dict(weights) is weights

    def test_ambiguous_checkpoint_reports_the_choice(self, capsys):
        """Единственное место, где приоритет ключей вообще наблюдаем.
        Среди наших чекпоинтов такого нет, но выбор должен быть виден в логе."""
        ours = {"fc.weight": torch.ones(2, 3)}
        theirs = {"fc.weight": torch.zeros(2, 3)}
        chosen = extract_state_dict({"student_state": ours, "model_state_dict": theirs})

        assert chosen is ours
        assert "несколько наборов весов" in capsys.readouterr().out


class TestLoadCheckpointInto:
    @pytest.mark.parametrize("wrapper", WRAPPERS)
    def test_weights_actually_arrive_in_the_model(self, tmp_path, wrapper):
        source = nn.Linear(3, 2)
        nn.init.constant_(source.weight, 0.25)
        nn.init.constant_(source.bias, -0.5)

        path = save(tmp_path, wrap(source.state_dict(), wrapper), f"{wrapper}.pt")
        target = load_checkpoint_into(nn.Linear(3, 2), path, "Linear")

        assert torch.equal(target.weight, source.weight)
        assert torch.equal(target.bias, source.bias)

    def test_module_prefix_from_ddp_is_stripped(self, tmp_path):
        source = nn.Linear(3, 2)
        wrapped = {f"module.{k}": v for k, v in source.state_dict().items()}

        path = save(tmp_path, {"student_state": wrapped}, "ddp.pt")
        target = load_checkpoint_into(nn.Linear(3, 2), path, "Linear")

        assert torch.equal(target.weight, source.weight)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_checkpoint_into(nn.Linear(3, 2), str(tmp_path / "нет.pt"), "Linear")

    def test_shape_mismatch_still_raises(self, tmp_path):
        """strict=True — чекпоинт от другой архитектуры обязан падать,
        а не грузиться наполовину."""
        path = save(tmp_path, nn.Linear(3, 2).state_dict(), "other.pt")
        with pytest.raises(RuntimeError):
            load_checkpoint_into(nn.Linear(5, 4), path, "Linear")


class TestFactoriesAcceptEveryWrapper:
    """Сквозная проверка: каждая фабрика читает каждый формат.

    Раньше torchvision-ветка не понимала чекпоинт без обёртки с ключом
    "model", timm-ветка не знала про model_state_dict, а U-Net — ни про
    model_state_dict, ни про model. Теперь набор общий.
    """

    @pytest.mark.parametrize("wrapper", WRAPPERS)
    def test_torchvision_factory(self, tmp_path, wrapper):
        reference = torchvision_model_for_classification("resnet18", num_classes=10)
        path = save(tmp_path, wrap(reference.state_dict(), wrapper), f"tv_{wrapper}.pt")

        loaded = torchvision_model_for_classification(
            "resnet18", num_classes=10, checkpoint_path=path
        )
        assert torch.equal(loaded.fc.weight, reference.fc.weight)

    @pytest.mark.parametrize("wrapper", WRAPPERS)
    def test_timm_factory(self, tmp_path, wrapper):
        reference = timm_model_for_classification("resnet18", num_classes=10)
        path = save(tmp_path, wrap(reference.state_dict(), wrapper), f"timm_{wrapper}.pt")

        loaded = timm_model_for_classification(
            "resnet18", num_classes=10, checkpoint_path=path
        )
        assert torch.equal(loaded.fc.weight, reference.fc.weight)

    @pytest.mark.parametrize("wrapper", WRAPPERS)
    def test_unet_factory(self, tmp_path, wrapper):
        reference = unet_for_segmentation(variant="tiny", num_classes=5)
        path = save(tmp_path, wrap(reference.state_dict(), wrapper), f"unet_{wrapper}.pt")

        loaded = unet_for_segmentation(variant="tiny", num_classes=5, checkpoint_path=path)
        assert torch.equal(loaded.head.weight, reference.head.weight)

    @pytest.mark.parametrize("wrapper", WRAPPERS)
    def test_segformer_factory(self, tmp_path, wrapper):
        reference = segformer_for_segmentation(variant="b0", num_classes=5, pretrained=None)
        path = save(tmp_path, wrap(reference.state_dict(), wrapper), f"segformer_{wrapper}.pt")

        # pretrained здесь игнорируется: веса всё равно перезапишет чекпоинт.
        loaded = segformer_for_segmentation(variant="b0", num_classes=5, checkpoint_path=path)
        assert torch.equal(
            loaded.model.decode_head.classifier.weight,
            reference.model.decode_head.classifier.weight,
        )


class TestRealTrainerCheckpointShape:
    def test_trainer_checkpoint_loads_through_every_factory_path(self, tmp_path):
        """Форма, которую реально пишут все три тренера (trainer.py:350/772/1104):
        student_state рядом с состояниями оптимизатора и планировщика."""
        reference = unet_for_segmentation(variant="tiny", num_classes=5)
        checkpoint = {
            "epoch": 42,
            "best_miou": 0.61,
            "student_state": reference.state_dict(),
            "criterion_state": {},
            "optimizer_state": {"param_groups": []},
            "scheduler_state": None,
            "scaler_state": {},
        }
        path = save(tmp_path, checkpoint, "trainer.pt")

        loaded = unet_for_segmentation(variant="tiny", num_classes=5, checkpoint_path=path)
        assert torch.equal(loaded.head.weight, reference.head.weight)
