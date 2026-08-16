"""Насколько каждая аугментация ломает УЧИТЕЛЯ.

Вопрос, на который отвечает скрипт: какие аугментации ученика учитель ещё
переносит (и тогда ему можно показывать тот же кадр), а на каких он
разваливается и начинает дистиллировать шум (и тогда их надо перечислить
в teacher_skips train-трансформа).

Как меряется. Берётся подвыборка val, к ней применяется ОДНА аугментация
за раз, и считается mIoU учителя относительно настоящей разметки. Никакого
обучения — только forward'ы, поэтому 50 кадров считаются за пару минут.

Геометрические аугментации сюда не входят намеренно: масштаб, кроп и
отражение применяются к обоим видам одинаково и вопроса «показывать ли их
учителю» не создают.

    python scripts/probe_teacher_augmentations.py \\
        experiment=segmentation/FitNets/cityscapes_FitNets_segformer_b2_to_unet_small \\
        +probe.samples=50 +probe.repeats=3

Читается тот же конфиг, что и у обучения, поэтому учитель, датасет и
нормировка — ровно те, с которыми пойдёт прогон.
"""

import logging
import random

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.utils import seed_everything
from src.utils.device import resolve_device
from src.utils.metrics import IoUAccumulator
from src.utils.segmentation_transforms import (
    SegmentationColorJitter,
    SegmentationCompose,
    SegmentationGaussianBlur,
    SegmentationNormalize,
    SegmentationRandomCrop,
    SegmentationRandomErasing,
    SegmentationRandomScale,
    SegmentationToTensor,
)

log = logging.getLogger(__name__)


def probe_transforms(cfg: DictConfig) -> dict[str, SegmentationCompose]:
    """{имя режима: трансформ}. Все режимы отличаются ровно одной операцией.

    Аугментации взяты с теми же параметрами, что в train-конфиге запуска:
    вопрос ведь не «вредит ли размытие вообще», а «вредит ли ТО размытие,
    которое реально стоит в рецепте».

    Геометрия (случайный масштаб и кроп) тоже берётся из train-рецепта, и
    это важно: сила фотометрической аугментации зависит от разрешения —
    размытие с sigma=2 на кропе 512x1024 съедает куда больше деталей, чем
    на полном кадре. Одинаковые кропы во всех режимах обеспечивает общий
    seed (см. measure).
    """
    train_cfg = cfg.data.transform.train
    dataset_cfg = cfg.data.dataset
    normalize = dataset_cfg.normalize

    def geometry() -> list:
        return [
            SegmentationRandomScale(
                scale_range=(float(dataset_cfg.scale_range[0]), float(dataset_cfg.scale_range[1])),
            ),
            SegmentationRandomCrop(
                crop_size=(int(dataset_cfg.crop_size[0]), int(dataset_cfg.crop_size[1])),
                ignore_index=dataset_cfg.ignore_index,
                cat_max_ratio=train_cfg.get("cat_max_ratio"),
            ),
        ]

    def compose(*augmentations) -> SegmentationCompose:
        return SegmentationCompose(
            [
                *geometry(),
                *augmentations,
                SegmentationToTensor(),
                SegmentationNormalize(normalize.mean, normalize.std),
            ]
        )

    modes = {"clean": compose()}

    jitter = float(train_cfg.get("color_jitter", 0.0))
    hue = float(train_cfg.get("hue", 0.0))
    if jitter > 0 or hue > 0:
        modes["jitter"] = compose(
            SegmentationColorJitter(
                brightness=jitter,
                contrast=jitter,
                saturation=jitter,
                hue=hue,
                p=1.0,
            )
        )

    if float(train_cfg.get("blur_p", 0.0)) > 0:
        modes["blur"] = compose(
            SegmentationGaussianBlur(
                p=1.0,
                kernel_size=int(train_cfg.get("blur_kernel_size", 5)),
                sigma=train_cfg.get("blur_sigma", (0.1, 2.0)),
            )
        )

    if float(train_cfg.get("random_erasing_p", 0.0)) > 0:
        # Стирание работает по нормализованному тензору, поэтому идёт после
        # ToTensor/Normalize — как и в самом train-рецепте.
        modes["erasing"] = SegmentationCompose(
            [
                *geometry(),
                SegmentationToTensor(),
                SegmentationNormalize(normalize.mean, normalize.std),
                SegmentationRandomErasing(
                    p=1.0,
                    scale=train_cfg.get("random_erasing_scale", (0.02, 0.2)),
                    ratio=train_cfg.get("random_erasing_ratio", (0.3, 3.3)),
                    value=train_cfg.get("random_erasing_value", 0.0),
                    erase_labels=False,
                    ignore_index=cfg.data.dataset.ignore_index,
                ),
            ]
        )

    return modes


@torch.no_grad()
def measure(
    teacher: torch.nn.Module,
    dataset,
    indices: list[int],
    transform: SegmentationCompose,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    amp: bool,
    seed: int,
) -> float:
    """mIoU учителя на подвыборке под заданным трансформом.

    seed фиксирует геометрию: кропы обязаны быть одними и теми же во всех
    режимах, иначе разница в mIoU окажется разницей между кадрами, а не
    между аугментациями.
    """
    random.seed(seed)
    accumulator = IoUAccumulator(num_classes, device, ignore_index)
    original_transform = dataset.transform
    dataset.transform = transform

    try:
        for index in indices:
            image, mask = dataset[index]
            image = image.to(device, non_blocking=True)[None]
            mask = mask.to(device, non_blocking=True)[None]

            with torch.autocast(device.type, enabled=amp):
                logits = teacher(image)

            accumulator.update(logits.float().argmax(dim=1), mask)
    finally:
        dataset.transform = original_transform

    return accumulator.compute()["miou"]


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, deterministic=False, warn_only=True)
    device = resolve_device(cfg.device)

    if cfg.model.get("teacher") is None:
        raise ValueError("В конфиге нет учителя — мерить нечего")

    probe = cfg.get("probe") or {}
    samples = int(probe.get("samples", 50))
    repeats = int(probe.get("repeats", 3))

    dataset = instantiate(cfg.data.dataset.build, train=False, transform=None)
    # Равномерно по валидации: Cityscapes отсортирован по городам, и первые N
    # подряд — это N кадров одной улицы.
    step = max(1, len(dataset) // samples)
    indices = list(range(0, len(dataset), step))[:samples]

    teacher = instantiate(cfg.model.teacher).to(device).eval()

    log.info("Учитель: %s | кадров: %d | повторов: %d", cfg.model.teacher._target_, len(indices), repeats)

    results: dict[str, float] = {}
    for name, transform in probe_transforms(cfg).items():
        # Кроп случаен, поэтому один проход мал даже для clean; repeats
        # усредняет разброс, а seed прохода общий для всех режимов.
        values = [
            measure(
                teacher,
                dataset,
                indices,
                transform,
                device,
                cfg.data.dataset.num_classes,
                cfg.data.dataset.ignore_index,
                amp=bool(cfg.trainer.amp) and device.type == "cuda",
                seed=cfg.seed + repeat,
            )
            for repeat in range(repeats)
        ]
        results[name] = sum(values) / len(values)

    baseline = results["clean"]
    log.info("mIoU учителя по режимам (все аугментации применены с p=1):")
    for name, value in results.items():
        log.info(
            "  %-10s mIoU=%.4f  (%+.4f к чистому кадру)",
            name,
            value,
            value - baseline,
        )
    log.info(
        "Правило чтения: если просадка заметно больше 0.01 mIoU, аугментацию "
        "стоит добавить в teacher_skips train-трансформа — ученику она полезна, "
        "а таргеты учителя портит."
    )


if __name__ == "__main__":
    main()
