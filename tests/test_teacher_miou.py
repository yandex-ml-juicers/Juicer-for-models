"""Диагностика "собственного качества учителя" в SegmentationTrainer.

Отличие от TeacherSimilarity (KL/agreement — насколько ученик похож на
учителя): здесь проверяется train_teacher_miou/teacher_native_miou —
насколько сам учитель прав относительно разметки. Флаг 'teacher_miou' в
clearml.scalars (см. docs/clearML.md).

Однопроцессный DistInfo строится вручную (world_size=1, без реального
process group) — тем же приёмом, что и fake_dist() в tests/test_distributed.py:
all_reduce_* внутри IoUAccumulator.synchronize() безопасно no-op'ают, когда
torch.distributed не инициализирован (см. src.utils.distributed._is_active).
"""

import csv
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.losses.cross_entropy import CrossEntropy
from src.training import SegmentationTrainer
from src.training.trainer import segmentation_evaluate
from src.utils.distributed import DistInfo

SEG_CLASSES = 4
SEG_IGNORE = 255


def fake_dist() -> DistInfo:
    return DistInfo(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"))


def _fixed_seg_dataset(size: int = 12, seed: int = 0) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(size, 3, 8, 8, generator=generator)
    masks = torch.randint(0, SEG_CLASSES, (size, 8, 8), generator=generator)
    return TensorDataset(images, masks)


def _read_history(output_dir: Path) -> list[dict]:
    with (output_dir / "history.csv").open() as f:
        return list(csv.DictReader(f))


def _make_trainer(tmp_path: Path, scalars: list[str], epochs: int = 2) -> SegmentationTrainer:
    dataset = _fixed_seg_dataset()
    train_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    eval_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    torch.manual_seed(1)
    student = nn.Conv2d(3, SEG_CLASSES, kernel_size=3, padding=1)
    torch.manual_seed(2)
    teacher = nn.Conv2d(3, SEG_CLASSES, kernel_size=3, padding=1)

    return SegmentationTrainer(
        student=student,
        teacher=teacher,
        criterion=CrossEntropy(ignore_index=SEG_IGNORE),
        optimizer=torch.optim.SGD(student.parameters(), lr=0.01),
        scheduler=None,
        train_loader=train_loader,
        eval_loader=eval_loader,
        num_classes=SEG_CLASSES,
        dist=fake_dist(),
        output_dir=tmp_path,
        epochs=epochs,
        save_best=False,
        save_last=False,
        progress_bar=False,
        scalars=scalars,
        ignore_index=SEG_IGNORE,
        plots={"enabled": False, "debug_samples": 0},
    )


def test_flag_off_adds_no_columns(tmp_path):
    """Без 'teacher_miou' в scalars поведение обучения не должно меняться."""
    trainer = _make_trainer(tmp_path, scalars=[])
    assert trainer.teacher_iou is None

    trainer.fit()
    rows = _read_history(tmp_path)
    assert "train_teacher_miou" not in rows[0]
    assert "teacher_native_miou" not in rows[0]


def test_teacher_miou_alone_does_not_enable_the_native_probe(tmp_path):
    """'teacher_miou' и 'teacher_native_miou' — независимые флаги (иначе
    любой прогон с дешёвой train_teacher_miou платил бы ещё и за разовый
    прогон по eval_loader, даже если он не нужен)."""
    trainer = _make_trainer(tmp_path, scalars=["teacher_miou"])
    assert trainer.teacher_iou is not None
    assert not trainer.teacher_native_miou_enabled

    trainer.fit()
    rows = _read_history(tmp_path)
    assert "train_teacher_miou" in rows[0]
    assert "teacher_native_miou" not in rows[0]


def test_flag_on_matches_independent_computation(tmp_path):
    """train_teacher_miou/teacher_native_miou логируются и совпадают с тем,
    что даёт прямой вызов IoUAccumulator/segmentation_evaluate на тех же
    данных — то есть подключение в трейнере ничего не портит и не дублирует.

    Оба флага заданы явно: они независимы (см. SegmentationTrainer.__init__),
    и по умолчанию teacher_native_miou не включается вместе с teacher_miou."""
    trainer = _make_trainer(tmp_path, scalars=["teacher_miou", "teacher_native_miou"], epochs=2)
    assert trainer.teacher_iou is not None
    assert trainer.teacher_native_miou_enabled

    trainer.fit()
    rows = _read_history(tmp_path)
    assert len(rows) == 2

    # teacher_native_miou — независимый прогон segmentation_evaluate учителя
    # по eval_loader (тот же путь, что scripts/eval.py eval_slot=teacher).
    _, expected_native_iou = segmentation_evaluate(
        model=trainer.teacher,
        loader=trainer.eval_loader,
        device=trainer.device,
        num_classes=SEG_CLASSES,
        ignore_index=SEG_IGNORE,
    )
    expected_native_miou = expected_native_iou.compute()["miou"]

    for row in rows:
        # Плоская линия-ориентир: одно и то же число на каждой эпохе, а не
        # пересчитанное заново (учитель заморожен, но проверяем именно то,
        # что реализация не гоняет лишний forward каждую эпоху).
        assert float(row["teacher_native_miou"]) == pytest.approx(expected_native_miou, abs=1e-6)

    # train_teacher_miou из первой и второй эпохи совпадают: те же
    # train_loader/teacher без shuffle дают те же кропы каждый раз.
    assert float(rows[0]["train_teacher_miou"]) == pytest.approx(
        float(rows[1]["train_teacher_miou"]), abs=1e-6
    )


def test_teacher_miou_is_free_of_extra_forward(tmp_path, monkeypatch):
    """train_teacher_miou не должен стоить дополнительного forward'а учителя:
    он использует teacher_logits, уже посчитанные ради KD-лосса.
    teacher_native_miou включён явно — это его единственный (разовый) forward."""
    trainer = _make_trainer(tmp_path, scalars=["teacher_miou", "teacher_native_miou"], epochs=1)

    calls = {"n": 0}
    original_forward = trainer.teacher.forward

    def counting_forward(*args, **kwargs):
        calls["n"] += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(trainer.teacher, "forward", counting_forward)
    trainer.fit()

    n_train_batches = len(trainer.train_loader)
    n_eval_batches = len(trainer.eval_loader)
    # Один forward на train-шаг (KD) + прогоны eval_loader для разового
    # teacher_native_miou — ни одного лишнего сверх этого.
    assert calls["n"] == n_train_batches + n_eval_batches
