"""NativeResolutionTeacher: апсемпл входа учителю, даунсемпл логитов обратно.

См. src/models/native_resolution_teacher.py — контекст и почему это нужно.
"""

import pytest
import torch

from src.models import NativeResolutionTeacher


class _RecordingStub(torch.nn.Module):
    """Сегментатор-заглушка: логиты в разрешении входа, никакой случайности,
    и запоминает форму последнего входа — так видно, апсемплился ли вход
    на самом деле, а не только выход обратно даунсемплился."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.head = torch.nn.Conv2d(3, num_classes, kernel_size=3, padding=1)
        self.last_input_shape: torch.Size | None = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.last_input_shape = images.shape
        return self.head(images)


def test_output_shape_matches_input_not_the_upscaled_size():
    model = _RecordingStub()
    wrapped = NativeResolutionTeacher(model, scale_factor=2.0).eval()

    with torch.no_grad():
        logits = wrapped(torch.randn(2, 3, 32, 64))

    assert logits.shape == (2, 5, 32, 64)


def test_model_actually_receives_the_upscaled_input():
    model = _RecordingStub()
    wrapped = NativeResolutionTeacher(model, scale_factor=2.0).eval()

    with torch.no_grad():
        wrapped(torch.randn(1, 3, 32, 64))

    assert model.last_input_shape == (1, 3, 64, 128)


def test_scale_factor_one_is_close_to_a_plain_forward():
    """scale_factor=1.0 — апсемпл входа тождественный, поэтому логиты почти
    совпадают с обычным forward'ом (не бит-в-бит: два bilinear-интерполятора
    туда-обратно на своей сетке, но расхождение исчезающе мало)."""
    model = _RecordingStub()
    images = torch.randn(1, 3, 32, 64)

    with torch.no_grad():
        plain = model(images)
        wrapped = NativeResolutionTeacher(model, scale_factor=1.0)(images)

    assert torch.allclose(plain, wrapped, atol=1e-5)


def test_module_attribute_holds_the_wrapped_model():
    """unwrap_model снимает обёртку по атрибуту .module (см.
    src/models/feature_extractor.py) — то же соглашение, что у
    MultiScaleInference."""
    model = _RecordingStub()
    wrapped = NativeResolutionTeacher(model)

    assert wrapped.module is model


def test_rejects_non_positive_scale_factor():
    with pytest.raises(ValueError):
        NativeResolutionTeacher(_RecordingStub(), scale_factor=0.0)
