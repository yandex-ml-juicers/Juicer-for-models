"""Продолжение упавшего обучения с последнего чекпоинта (SegmentationTrainer).

Адаптация DetectionTrainer.load_checkpoint (см. src/training/trainer.py) —
эти тесты фиксируют тот же контракт для сегментации: чекпоинт содержит
epoch/best_miou/best_epoch/student_state/criterion_state/optimizer_state/
scheduler_state/scaler_state, load_checkpoint() восстанавливает состояние и
двигает start_epoch, а второй fit() не повторяет уже пройденные эпохи и не
теряет уже накопленную history.csv (см. MetricsHistory(resume=...)).

Фикстуры/приём с fake_dist() — тот же, что в tests/test_teacher_miou.py.
"""

import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.losses.cross_entropy import CrossEntropy
from src.training import SegmentationTrainer
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


def _make_trainer(
    output_dir: Path,
    epochs: int,
    *,
    seed: int = 1,
    metrics_callback=None,
) -> SegmentationTrainer:
    dataset = _fixed_seg_dataset()
    train_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    eval_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    torch.manual_seed(seed)
    student = nn.Conv2d(3, SEG_CLASSES, kernel_size=3, padding=1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    return SegmentationTrainer(
        student=student,
        teacher=None,
        criterion=CrossEntropy(ignore_index=SEG_IGNORE),
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        eval_loader=eval_loader,
        num_classes=SEG_CLASSES,
        dist=fake_dist(),
        output_dir=output_dir,
        epochs=epochs,
        save_best=True,
        save_last=True,
        progress_bar=False,
        metrics_callback=metrics_callback,
        scalars=[],
        ignore_index=SEG_IGNORE,
        plots={"enabled": False, "debug_samples": 0},
    )


def test_fresh_trainer_starts_at_epoch_one(tmp_path):
    trainer = _make_trainer(tmp_path, epochs=3)
    assert trainer.start_epoch == 1
    assert trainer.best_miou == 0.0
    assert trainer.best_epoch == 0


def test_load_checkpoint_restores_weights_optimizer_and_epoch(tmp_path):
    """Свежая модель после load_checkpoint неотличима от той, что реально
    отучилась 2 эпохи — не только по метаданным (start_epoch), но и по
    фактическим весам (доказывает, что state_dict реально применился, а не
    только прочитан)."""
    trainer1 = _make_trainer(tmp_path, epochs=2, seed=1)
    trainer1.fit()

    trained_weight = next(trainer1.student.parameters()).clone()

    # Другой seed -> другая инициализация: если load_checkpoint промолчал бы
    # (например из-за опечатки в ключе), веса остались бы ИМЕННО этими.
    trainer2 = _make_trainer(tmp_path, epochs=4, seed=999)
    fresh_weight = next(trainer2.student.parameters()).clone()
    assert not torch.allclose(fresh_weight, trained_weight)

    trainer2.load_checkpoint(tmp_path / "last.pt")

    assert trainer2.start_epoch == 3  # epoch=2 в чекпоинте -> продолжаем с 3
    assert trainer2.best_epoch in (1, 2)  # обе эпохи могли стать лучшей
    assert torch.allclose(next(trainer2.student.parameters()), trained_weight)
    # momentum=0.9: optimizer.state непустой после эпохи обучения, и он должен
    # переехать вместе с чекпоинтом, а не начаться с нуля.
    assert len(trainer2.optimizer.state) == len(trainer1.optimizer.state) > 0


def test_load_checkpoint_tolerates_missing_best_epoch_key(tmp_path):
    """Чекпоинты, сохранённые ДО появления resume (в т.ч. три упавших на
    другом сервере прогона), не содержат ключ best_epoch — load_checkpoint
    обязан подставить разумное приближение, а не падать с KeyError."""
    trainer = _make_trainer(tmp_path, epochs=1)
    trainer.fit()

    checkpoint = torch.load(tmp_path / "last.pt", weights_only=True)
    assert "best_epoch" in checkpoint
    del checkpoint["best_epoch"]
    torch.save(checkpoint, tmp_path / "legacy_last.pt")

    trainer2 = _make_trainer(tmp_path, epochs=2, seed=42)
    trainer2.load_checkpoint(tmp_path / "legacy_last.pt")
    assert trainer2.best_epoch == checkpoint["epoch"]


def test_resumed_fit_does_not_repeat_finished_epochs_and_keeps_old_history(tmp_path):
    """История эпох 1..2 не должна пропасть или задвоиться, когда обучение
    продолжается эпохами 3..4 в тот же output_dir."""
    trainer1 = _make_trainer(tmp_path, epochs=2, seed=1)
    trainer1.fit()
    rows_before = _read_history(tmp_path)
    assert [row["epoch"] for row in rows_before] == ["1", "2"]

    trainer2 = _make_trainer(tmp_path, epochs=4, seed=1)
    trainer2.load_checkpoint(tmp_path / "last.pt")

    ran_epochs = []
    original_train_epoch = trainer2._train_epoch

    def counting_train_epoch(epoch):
        ran_epochs.append(epoch)
        return original_train_epoch(epoch)

    trainer2._train_epoch = counting_train_epoch
    trainer2.fit()

    assert ran_epochs == [3, 4]  # не переобучаем 1 и 2 заново

    rows_after = _read_history(tmp_path)
    assert [row["epoch"] for row in rows_after] == ["1", "2", "3", "4"]


def test_resume_replays_history_into_metrics_callback(tmp_path):
    """ClearML-репортер асинхронный и мог не долететь до крэша — при resume
    все локально сохранённые эпохи должны уйти в metrics_callback заново."""
    recorded: list[dict] = []
    callback = (lambda row: recorded.append(row), lambda *a, **k: None, lambda *a, **k: None, None)

    trainer1 = _make_trainer(tmp_path, epochs=2, seed=1, metrics_callback=callback)
    trainer1.fit()
    assert len(recorded) == 2  # обе эпохи ушли живьём при первом прогоне
    recorded.clear()

    trainer2 = _make_trainer(tmp_path, epochs=3, seed=1, metrics_callback=callback)
    trainer2.load_checkpoint(tmp_path / "last.pt")
    trainer2.fit()

    # Реплей истории (эпохи 1-2) + живой репорт новой эпохи (3) = 3 вызова.
    # Реплеенные строки идут из CSV (str), живая — из all_values (int).
    assert len(recorded) == 3
    assert [int(row["epoch"]) for row in recorded] == [1, 2, 3]
