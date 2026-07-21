import numpy as np
import torch
from PIL import Image

from src.data.transforms import base_transform


def solid_image(value: int = 128, size: int = 32) -> Image.Image:
    return Image.fromarray(np.full((size, size, 3), value, dtype=np.uint8))


def test_normalization_values():
    mean, std = [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]
    transform = base_transform(mean=mean, std=std)
    tensor = transform(solid_image(128))

    assert tensor.shape == (3, 32, 32)
    expected = (128 / 255 - 0.5) / 0.25
    assert torch.allclose(tensor, torch.full_like(tensor, expected), atol=1e-6)


def test_resize_applied_when_image_size_set():
    transform = base_transform(mean=[0.5] * 3, std=[0.5] * 3, image_size=224)
    tensor = transform(solid_image(64, size=32))
    assert tensor.shape == (3, 224, 224)


def test_no_resize_by_default():
    transform = base_transform(mean=[0.5] * 3, std=[0.5] * 3, image_size=None)
    assert transform(solid_image()).shape == (3, 32, 32)
