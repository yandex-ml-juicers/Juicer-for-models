"""Композиция Hydra-конфигов: каждый experiment собирается и согласован.

Ловит битые overrides, опечатки в ключах и рассинхрон "лосс требует учителя,
а учителя в конфиге нет" ещё до запуска обучения.
"""

import pytest
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.losses import DistillationLoss

# Пути от configs/experiment/ — эксперименты разложены по подкаталогам.
EXPERIMENTS = [
    "baseline/b1_resnet56_to_resnet20_kd",
    "baseline/b1_scratch",
    "baseline/b2_feature_kd",
    "baseline/b2_scratch",
    "scratch/imagenet1k_scratch_resnet18",
]

# Сегментация: бейзлайны без дистилляции + методы дистилляции
# SegFormer-B2 -> U-Net.
# Абляция лоссов на прогонах без дистилляции: каждый отличается от своего
# бейзлайна (..._aug) ровно блоком loss.
LOSS_ABLATIONS = [
    f"scratch/cityscapes_scratch_timm_unet_{size}_aug_{loss}"
    for size in ("small", "base")
    for loss in ("dice", "ohem", "lovasz")
]

SEGMENTATION_EXPERIMENTS = [
    "scratch/cityscapes_scratch_segformer_b2",
    "scratch/cityscapes_scratch_segformer_b5",
    "scratch/cityscapes_scratch_unet",
    "scratch/cityscapes_scratch_timm_unet",
    "scratch/cityscapes_scratch_timm_unet_small_aug",
    "scratch/cityscapes_scratch_timm_unet_base_aug",
    "segmentation/vanilla-KD/cityscapes_pixel-KD_segformer_b2_to_unet_base",
    "segmentation/vanilla-KD/cityscapes_pixel-KD_segformer_b5_to_unet_small",
    "segmentation/CWD/cityscapes_CWD_segformer_b2_to_unet_base",
    "segmentation/CWD/cityscapes_CWD_segformer_b5_to_unet_small",
    "segmentation/FitNets/cityscapes_FitNets_segformer_b2_to_unet_base",
    "segmentation/FitNets/cityscapes_FitNets_segformer_b2_to_unet_small",
    "segmentation/FitNets/cityscapes_FitNets_dice_ohem_segformer_b5_to_unet_small",
    "segmentation/DIST/cityscapes_DIST_segformer_b2_to_timm_unet_base",
    "segmentation/DIST/cityscapes_DIST_segformer_b5_to_timm_unet_small",
    "segmentation/BPKD/cityscapes_BPKD_segformer_b5_to_unet_small",
    "segmentation/HeteroAKD/cityscapes_HeteroAKD_segformer_b5_to_unet_small",
    *LOSS_ABLATIONS,
    # Новые архитектуры/диагностика (см. outputs/claude-analis/analysis.md):
    # SegNeXt-ученик, Mask2Former-учитель, наносайз timm-U-Net (0.5-2M).
    "segmentation/diagnostics/cityscapes_teacher_miou_probe_segformer_b5_to_unet_small",
    "scratch/cityscapes_scratch_segnext_s",
    "segmentation/BPKD/cityscapes_BPKD_segformer_b5_to_segnext_s",
    "segmentation/BPKD/cityscapes_BPKD_mask2former_tiny_to_unet_small",
    "segmentation/BPKD/cityscapes_BPKD_mask2former_small_to_unet_small",
    "scratch/cityscapes_scratch_timm_unet_mobilenetv3_small_aug_lovasz",
    "segmentation/BPKD/cityscapes_BPKD_segformer_b5_to_timm_unet_mobilenetv3_small",
    "segmentation/BPKD/cityscapes_BPKD_mask2former_tiny_to_segnext_t",
    "segmentation/DIST/cityscapes_DIST_mask2former_tiny_to_segnext_t",
    # Абляции capacity gap (см. outputs/claude-analis/analysis.md, §"Что бы
    # я проверил дальше"): чистая KD без GT-членов, учитель поменьше/побольше
    # студент, оба сразу поменьше, студент побольше.
    "segmentation/diagnostics/cityscapes_BPKD_pure_segformer_b5_to_unet_small",
    "segmentation/diagnostics/cityscapes_BPKD_segformer_b2_to_unet_small",
    "segmentation/diagnostics/cityscapes_BPKD_segformer_b1_to_unet_tiny",
    "segmentation/diagnostics/cityscapes_BPKD_segformer_b5_to_unet_base",
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


class TestTeacherViewExperiment:
    """Эксперименты, где ученик видит сильные аугментации, а учитель — нет."""

    EXPERIMENT = "segmentation/FitNets/cityscapes_FitNets_segformer_b2_to_unet_small"

    def test_regularization_is_on_and_hidden_from_the_teacher(self):
        cfg = compose_config([f"experiment={self.EXPERIMENT}"])
        train_transform = cfg.data.transform.train

        assert train_transform.blur_p > 0
        assert train_transform.random_erasing_p > 0
        assert set(train_transform.teacher_skips) == {"jitter", "blur", "erasing"}

    def test_transform_yields_two_views(self):
        """Трансформ обязан отдавать тройку (ученик, учитель, маска):
        именно по ней тренер понимает, что учителю нужен свой кадр."""
        from src.utils.segmentation_transforms import SegmentationTeacherViewCompose

        cfg = compose_config([f"experiment={self.EXPERIMENT}"])
        transform = instantiate(cfg.data.transform.train)
        assert isinstance(transform, SegmentationTeacherViewCompose)

    def test_distillation_terms_fade_out(self):
        """Расписание обязано гасить дистилляцию, а не наоборот."""
        cfg = compose_config([f"experiment={self.EXPERIMENT}"])
        assert cfg.loss_schedule.hint_weight.end < cfg.loss_schedule.hint_weight.start
        assert cfg.loss_schedule.ce_weight.end > cfg.loss_schedule.ce_weight.start


class TestLossAblation:
    """Прогоны «тот же ученик, другой лосс» на scratch-бейзлайнах."""

    @pytest.mark.parametrize("experiment", LOSS_ABLATIONS)
    def test_differs_from_the_baseline_only_in_the_loss(self, experiment):
        """Если разойдётся хоть что-то ещё — lr, число эпох, аугментации, —
        разницу в mIoU нельзя будет отнести к лоссу, и весь прогон
        превращается в трату GPU-часов."""
        baseline_name = experiment.rsplit("_", 1)[0]
        baseline = compose_config([f"experiment={baseline_name}"])
        ablation = compose_config([f"experiment={experiment}"])

        for section in ("trainer", "optimizer", "scheduler", "data", "model", "augment"):
            assert ablation[section] == baseline[section], section
        assert ablation.data.transform == baseline.data.transform
        assert ablation.loss._target_ != baseline.loss._target_

    @pytest.mark.parametrize("experiment", LOSS_ABLATIONS)
    def test_names_do_not_collide(self, experiment):
        """reuse_last_task_id: True — значит совпадение имён затрёт чужой
        прогон в ClearML и сложит артефакты в один каталог outputs/."""
        cfg = compose_config([f"experiment={experiment}"])
        assert cfg.name == experiment.rsplit("/", 1)[-1]

    def test_ohem_turns_label_smoothing_off(self):
        """Сглаживание держит per-pixel лосс выше нуля даже на угаданном
        пикселе, порог -log(thresh) перестаёт отсекать, и OHEM вырождается
        в обычную CE. Вместе эти два приёма не работают."""
        for experiment in LOSS_ABLATIONS:
            if experiment.endswith("_ohem"):
                assert compose_config([f"experiment={experiment}"]).loss.label_smoothing == 0.0

    def test_others_keep_the_baseline_smoothing(self):
        """У Dice и Lovász CE-половина обязана совпасть с бейзлайном
        до последнего параметра, иначе сравнивается не только лосс."""
        baseline = compose_config(["experiment=scratch/cityscapes_scratch_timm_unet_small_aug"])
        for experiment in LOSS_ABLATIONS:
            if experiment.endswith(("_dice", "_lovasz")):
                cfg = compose_config([f"experiment={experiment}"])
                assert cfg.loss.label_smoothing == baseline.loss.label_smoothing
                assert cfg.loss.ce_weight == 1.0


def test_loss_schedule_paths_exist_in_the_loss():
    """Опечатка в пути расписания иначе всплыла бы через час обучения."""
    from src.training import LossWeightScheduler

    for experiment in SEGMENTATION_EXPERIMENTS:
        cfg = compose_config([f"experiment={experiment}"])
        if not cfg.get("loss_schedule"):
            continue
        criterion = instantiate(cfg.loss)
        LossWeightScheduler(
            criterion,
            OmegaConf.to_container(cfg.loss_schedule, resolve=True),
            total_epochs=cfg.trainer.epochs,
        )


def test_composite_counts_cross_entropy_once():
    """CE есть почти в каждом лоссе проекта, и в композиции её легко
    посчитать дважды с непонятным итоговым весом. Здесь попиксельную
    классификацию берёт на себя только OHEM.

    Проверяется свойство, а не имя слагаемого: имена в композициях меняются
    от эксперимента к эксперименту, а инвариант «CE ровно в одном члене»
    обязан держаться в любом.
    """
    cfg = compose_config(
        ["experiment=segmentation/FitNets/cityscapes_FitNets_dice_ohem_segformer_b5_to_unet_small"]
    )
    with_ce = {
        name: loss_cfg._target_.split(".")[-1]
        for name, loss_cfg in cfg.loss.losses.items()
        if loss_cfg.get("ce_weight", 0.0) > 0 or loss_cfg._target_.endswith("OhemCrossEntropy")
    }
    assert len(with_ce) == 1, with_ce
    assert "OhemCrossEntropy" in with_ce.values(), with_ce


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
    assert cfg.data.dataset.num_classes == 19
    # ignore_index должен доехать до лосса, иначе void-пиксели Cityscapes
    # станут двадцатым «классом» и испортят обучение. У композиции своего
    # ignore_index нет — он задан в слагаемых; те, что его не объявляют,
    # полагаются на дефолт 255, и он обязан совпасть с датасетом.
    losses = cfg.loss.get("losses")
    for loss_cfg in losses.values() if losses is not None else [cfg.loss]:
        assert loss_cfg.get("ignore_index", 255) == cfg.data.dataset.ignore_index


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
    EXPERIMENT = "scratch/cityscapes_scratch_timm_unet_small_aug"

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
    cfg = compose_config(["experiment=scratch/cityscapes_scratch_timm_unet"])
    train_transform = cfg.data.transform.train

    assert train_transform.color_jitter == 0.4
    assert train_transform.hflip_p == 0.5
    assert train_transform.hue == 0.0
    assert train_transform.cat_max_ratio is None
    assert train_transform.blur_p == 0.0
    assert train_transform.random_erasing_p == 0.0


FITNETS_EXPERIMENTS = [
    "segmentation/FitNets/cityscapes_FitNets_segformer_b2_to_unet_base",
    "segmentation/FitNets/cityscapes_FitNets_segformer_b2_to_unet_small",
    "segmentation/FitNets/cityscapes_FitNets_dice_ohem_segformer_b5_to_unet_small",
]


def fitnets_layers(cfg):
    """Спецификация тапов лосса — из самого лосса или из слагаемого композиции."""
    losses = cfg.loss.get("losses")
    return cfg.loss.layers if losses is None else losses.kd.layers


def build_student(cfg):
    from src.models import TIMM_UNET_VARIANTS, TimmUNet

    return TimmUNet(
        encoder_name=TIMM_UNET_VARIANTS[cfg.model.student.variant]["encoder_name"],
        num_classes=19,
        pretrained=False,
    )


@pytest.mark.parametrize("experiment", FITNETS_EXPERIMENTS)
def test_fitnets_channels_match_the_configured_pair(experiment):
    """Каналы регрессора обязаны совпадать с реальными ширинами стадий
    SegFormer-B2 и энкодера ученика — иначе адаптер соберётся, а форма
    не сойдётся уже в первом батче.

    Ширины ученика берём не из таблицы, а из собранной модели: у timm-U-Net
    они целиком определяются энкодером.
    """
    from src.models import SEGFORMER_VARIANTS

    cfg = compose_config([f"experiment={experiment}"])
    student = build_student(cfg)
    hidden_sizes = SEGFORMER_VARIANTS[cfg.model.teacher.variant]["hidden_sizes"]

    for name, spec in fitnets_layers(cfg).items():
        stage = name.removeprefix("taps.")
        assert spec.student_channels == student.tap_channels[stage]
        # stageN — N-я стадия в спецификации MiT.
        assert spec.teacher_channels == hidden_sizes[int(stage[-1]) - 1]


@pytest.mark.parametrize("experiment", FITNETS_EXPERIMENTS)
def test_fitnets_requests_only_taps_the_student_actually_has(experiment):
    """Тап, которого у ученика нет, уронил бы FeatureExtractor только
    на запуске обучения — ловим здесь."""
    cfg = compose_config([f"experiment={experiment}"])
    student = build_student(cfg)

    available = {f"taps.{name}" for name in student.tap_channels}
    requested = set(fitnets_layers(cfg))

    assert requested <= available, (
        f"конфиг просит {requested - available}, "
        f"а у ученика есть только {sorted(available)}"
    )
