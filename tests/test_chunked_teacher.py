"""ChunkedTeacher: forward учителя кусками батча, результат тот же, что и
без чанкинга (учитель без градиентов — не приближение).

См. src/models/chunked_teacher.py.
"""

import pytest
import torch

from src.models import ChunkedTeacher


class _RecordingStub(torch.nn.Module):
    """Сегментатор-заглушка: логиты в разрешении входа, никакой случайности
    (BatchNorm бы сделал результат зависимым от состава батча — здесь его
    нет специально), и запоминает размеры всех входов, которые видел."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.head = torch.nn.Conv2d(3, num_classes, kernel_size=3, padding=1)
        self.seen_batch_sizes: list[int] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.seen_batch_sizes.append(images.size(0))
        return self.head(images)


def test_output_matches_a_plain_forward():
    """Не torch.equal: свёртка без BatchNorm математически не зависит от
    состава батча, но конкретный алгоритм (cuDNN/MKL) вправе отличаться при
    разном batch_size на входе, и он даёт чуть другое округление FP — это
    не приближение чанкинга, а обычный шум плавающей точки."""
    model = _RecordingStub()
    images = torch.randn(7, 3, 16, 24)

    with torch.no_grad():
        plain = model(images)

    model.seen_batch_sizes.clear()
    wrapped = ChunkedTeacher(model, micro_batch_size=3)
    with torch.no_grad():
        chunked = wrapped(images)

    assert torch.allclose(plain, chunked, atol=1e-5)


def test_model_actually_receives_chunks_not_the_whole_batch():
    model = _RecordingStub()
    wrapped = ChunkedTeacher(model, micro_batch_size=3)

    with torch.no_grad():
        wrapped(torch.randn(7, 3, 16, 24))

    # 7 при кусках по 3 -> 3, 3, 1: ни один forward не видел весь батч.
    assert model.seen_batch_sizes == [3, 3, 1]


def test_batch_not_larger_than_micro_batch_size_is_a_single_call():
    model = _RecordingStub()
    wrapped = ChunkedTeacher(model, micro_batch_size=8)

    with torch.no_grad():
        wrapped(torch.randn(5, 3, 16, 24))

    assert model.seen_batch_sizes == [5]


def test_module_attribute_holds_the_wrapped_model():
    model = _RecordingStub()
    wrapped = ChunkedTeacher(model, micro_batch_size=3)

    assert wrapped.module is model


def test_rejects_non_positive_micro_batch_size():
    with pytest.raises(ValueError):
        ChunkedTeacher(_RecordingStub(), micro_batch_size=0)
