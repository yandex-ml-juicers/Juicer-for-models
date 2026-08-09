"""Композиция Hydra-конфигов: каждый experiment собирается и согласован.

Ловит битые overrides, опечатки в ключах и рассинхрон "лосс требует учителя,
а учителя в конфиге нет" ещё до запуска обучения.
"""

import pytest
from hydra import compose, initialize
from hydra.utils import instantiate

from src.losses import DistillationLoss

# Пути от configs/experiment/ — эксперименты разложены по подкаталогам.
EXPERIMENTS = [
    "baseline/b1_resnet56_to_resnet20_kd",
    "baseline/b1_scratch",
    "baseline/b2_feature_kd",
    "baseline/b2_scratch",
    "scratch/imagenet1k_scratch_resnet18",
]

# Сегментация: бейзлайны без дистилляции + 4 метода дистилляции
# SegFormer-B2 -> U-Net-base.
SEGMENTATION_EXPERIMENTS = [
    "scratch/cityscapes_scratch_segformer_b2",
    "scratch/cityscapes_scratch_segformer_b5",
    "scratch/cityscapes_scratch_unet",
    "scratch/cityscapes_scratch_unet_base",
    "scratch/cityscapes_scratch_unet_base_strong_aug",
    "segmentation/vanilla-KD/cityscapes_pixel-KD_segformer_b2_to_unet_base",
    "segmentation/CWD/cityscapes_CWD_segformer_b2_to_unet_base",
    "segmentation/FitNets/cityscapes_FitNets_segformer_b2_to_unet_base",
    "segmentation/DIST/cityscapes_DIST_segformer_b2_to_unet_base",
]


def compose_config(overrides: list[str]):
    with initialize(version_base="1.3", config_path="../configs"):
        return compose(config_name="config", overrides=overrides)


def test_default_config_composes():
    cfg = compose_config([])
    assert cfg.seed == 42
    assert cfg.model.teacher is not None
    assert cfg.trainer.epochs > 0


@pytest.mark.parametrize("experiment", EXPERIMENTS + SEGMENTATION_EXPERIMENTS)
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
    cfg = compose_config(["experiment=baseline/b2_feature_kd"])
    criterion = instantiate(cfg.loss)
    assert criterion.required_features == ("layer1", "layer2", "layer3", "layer4")
    assert len(list(criterion.parameters())) == 4  # по одной 1x1-свёртке на слой


def test_scratch_experiments_have_no_teacher():
    for experiment in ["baseline/b1_scratch", "baseline/b2_scratch"]:
        cfg = compose_config([f"experiment={experiment}"])
        assert cfg.model.get("teacher") is None


def test_scheduler_t_max_follows_epochs():
    cfg = compose_config(["experiment=baseline/b1_resnet56_to_resnet20_kd", "trainer.epochs=7"])
    assert cfg.scheduler.T_max == 7


@pytest.mark.parametrize("experiment", SEGMENTATION_EXPERIMENTS)
def test_segmentation_experiments_are_wired_for_segmentation(experiment):
    cfg = compose_config([f"experiment={experiment}"])

    assert cfg.task_type == "segmentation"
    # ignore_index должен доехать до лосса, иначе void-пиксели Cityscapes
    # станут двадцатым «классом» и испортят обучение.
    assert cfg.loss.ignore_index == cfg.data.dataset.ignore_index
    assert cfg.data.dataset.num_classes == 19


def test_segmentation_scratch_experiments_have_no_teacher():
    for experiment in SEGMENTATION_EXPERIMENTS:
        if "scratch" not in experiment:
            continue
        cfg = compose_config([f"experiment={experiment}"])
        assert cfg.model.get("teacher") is None


class TestBatchAugmentGroup:
    """Группа augment (Mixup/CutMix) подключается к любому эксперименту
    одним override'ом и по умолчанию выключена."""

    def test_disabled_by_default(self):
        assert compose_config([]).get("augment") is None

    @pytest.mark.parametrize("augment", ["mixup_cutmix", "cutmix_segmentation"])
    def test_augment_configs_instantiate(self, augment):
        from src.data import MixupCutmix

        cfg = compose_config([f"augment={augment}"])
        assert isinstance(instantiate(cfg.augment), MixupCutmix)

    def test_segmentation_preset_is_cutmix_only(self):
        """Линейная смесь двух городских сцен даёт кадр, которого не бывает,
        и размазанный таргет. Для плотных задач это не рабочий вариант."""
        cfg = compose_config(["augment=cutmix_segmentation"])
        assert cfg.augment.mixup_alpha == 0.0
        assert cfg.augment.cutmix_alpha > 0.0

    def test_can_be_attached_to_an_existing_experiment(self):
        cfg = compose_config(
            [
                "experiment=segmentation/CWD/cityscapes_CWD_segformer_b2_to_unet_base",
                "augment=cutmix_segmentation",
            ]
        )
        assert cfg.augment is not None
        assert cfg.task_type == "segmentation"


class TestStrongAugmentationExperiment:
    EXPERIMENT = "scratch/cityscapes_scratch_unet_base_strong_aug"

    def test_differs_from_the_baseline_only_in_augmentation(self):
        """Пара «базовый / усиленный» имеет смысл, только если всё остальное
        совпадает: иначе разница в mIoU объясняется не аугментациями."""
        baseline = compose_config(["experiment=scratch/cityscapes_scratch_unet_base"])
        strong = compose_config([f"experiment={self.EXPERIMENT}"])

        assert strong.optimizer.lr == baseline.optimizer.lr
        assert strong.data.loader.batch_size == baseline.data.loader.batch_size
        assert strong.model.student.variant == baseline.model.student.variant
        assert strong.loss.label_smoothing == baseline.loss.label_smoothing

    def test_augmentations_are_actually_on(self):
        cfg = compose_config([f"experiment={self.EXPERIMENT}"])

        assert cfg.augment is not None
        assert cfg.data.transform.train.cat_max_ratio == 0.75
        assert cfg.data.transform.train.random_erasing_p > 0
        assert cfg.model.student.dropout > 0

    def test_evaluation_transform_has_no_augmentation(self):
        """Аугментации на валидации сделали бы метрику несравнимой
        с публичными числами."""
        cfg = compose_config([f"experiment={self.EXPERIMENT}"])
        assert cfg.data.transform.eval._target_.endswith("build_segmentation_transform_eval")


def test_base_segmentation_transform_stays_unchanged():
    """Базовый рецепт — точка отсчёта для уже посчитанных бейзлайнов.
    Новые аугментации в нём обязаны быть выключены."""
    cfg = compose_config(["experiment=scratch/cityscapes_scratch_unet_base"])
    train_transform = cfg.data.transform.train

    assert train_transform.color_jitter == 0.4
    assert train_transform.hflip_p == 0.5
    assert train_transform.hue == 0.0
    assert train_transform.cat_max_ratio is None
    assert train_transform.blur_p == 0.0
    assert train_transform.random_erasing_p == 0.0


def test_fitnets_channels_match_the_configured_pair():
    """Каналы регрессора обязаны совпадать с реальными ширинами стадий
    SegFormer-B2 и U-Net-base — иначе адаптер соберётся, а форма не сойдётся
    уже в первом батче.

    Ширины ученика берём не из таблицы, а из собранной модели: у U-Net они
    зависят и от base_channels, и от depth, и держать их в голове бесполезно.
    """
    from src.models import SEGFORMER_VARIANTS, UNET_VARIANTS, UNet

    cfg = compose_config(
        ["experiment=segmentation/FitNets/cityscapes_FitNets_segformer_b2_to_unet_base"]
    )
    spec = cfg.loss.layers["taps.stage3"]

    student = UNet(num_classes=19, **UNET_VARIANTS[cfg.model.student.variant])
    assert spec.student_channels == student.tap_channels["stage3"]

    # stage3 — третья стадия (индекс 2) в спецификации MiT.
    assert (
        spec.teacher_channels
        == SEGFORMER_VARIANTS[cfg.model.teacher.variant]["hidden_sizes"][2]
    )


def test_fitnets_requests_only_taps_the_student_actually_has():
    """У U-Net с depth=4 нет стадии на страйде 32. Если конфиг попросит
    taps.stage4, FeatureExtractor упадёт только на запуске обучения —
    ловим здесь."""
    from src.models import UNET_VARIANTS, UNet

    cfg = compose_config(
        ["experiment=segmentation/FitNets/cityscapes_FitNets_segformer_b2_to_unet_base"]
    )
    student = UNet(num_classes=19, **UNET_VARIANTS[cfg.model.student.variant])
    available = {f"taps.{name}" for name in student.tap_channels}

    assert set(cfg.loss.layers) <= available, (
        f"конфиг просит {set(cfg.loss.layers) - available}, "
        f"а у ученика есть только {sorted(available)}"
    )
