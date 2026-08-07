import random
from collections.abc import Callable, Sequence

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as F

class DetectionCompose:
    """Applies transforms to image and target sequentially."""

    def __init__(
        self, 
        transforms: Sequence[Callable]
    ) -> None:
        self.transforms = transforms

    def __call__(
        self,
        image: Image.Image | Tensor,
        target: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        for transform in self.transforms:
            image, target = transform(image, target)

        return image, target

class DetectionResize:
    """Resizes the image and scales the bounding boxes."""

    def __init__(
        self,
        size: tuple[int, int],
    ) -> None:
        self.size = size

    def __call__(
        self,
        image: Image.Image | Tensor,
        target: dict[str, Tensor],
    ) -> tuple[Image.Image | Tensor, dict[str, Tensor]]:
        new_height, new_width = self.size

        if isinstance(image, Image.Image):
            old_width, old_height = image.size
        else:
            old_height, old_width = image.shape[-2:]

        image = F.resize(image, size=[new_height, new_width], antialias=True)

        scale_x = new_width / old_width
        scale_y = new_height / old_height

        target = target.copy()
        boxes = target["boxes"].clone()

        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

            boxes[:, [0, 2]].clamp_(0, new_width)
            boxes[:, [1, 3]].clamp_(0, new_height)

        target["boxes"] = boxes
        target["size"] = torch.tensor([new_height, new_width], dtype=torch.int64)

        if "area" in target:
            target["area"] = (target["area"] * scale_x * scale_y)

        return image, target

class DetectionRandomHorizontalFlip:
    """Randomly flips the image and bounding boxes horizontally."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self,
        image: Image.Image | Tensor,
        target: dict[str, Tensor],
    ) -> tuple[Image.Image | Tensor, dict[str, Tensor]]:
        if random.random() >= self.p:
            return image, target

        if isinstance(image, Image.Image):
            width, _ = image.size
        else:
            _, width = image.shape[-2:]

        image = F.hflip(image)

        target = target.copy()
        boxes = target["boxes"].clone()

        if boxes.numel() > 0:
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]

        target["boxes"] = boxes

        return image, target

class DetectionColorJitter:
    """Randomly changes image brightness, contrast, saturation and hue."""

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ) -> None:
        self.transform = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(
        self,
        image: Image.Image | Tensor,
        target: dict[str, Tensor],
    ) -> tuple[Image.Image | Tensor, dict[str, Tensor]]:
        image = self.transform(image)

        return image, target

class DetectionRandomResize:
    """Randomly chooses one image size and resizes image and bounding boxes."""

    def __init__(
        self,
        sizes: Sequence[tuple[int, int]],
    ) -> None:
        self.sizes = tuple(sizes)

    def __call__(
        self,
        image: Image.Image | Tensor,
        target: dict[str, Tensor],
    ) -> tuple[Image.Image | Tensor, dict[str, Tensor]]:
        size = random.choice(self.sizes)

        resize = DetectionResize(size)

        return resize(image, target)

class DetectionToTensor:
    def __call__(
        self,
        image: Image.Image | Tensor,
        target: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not isinstance(image, Tensor):
            image = F.pil_to_tensor(image)
            image = image.float() / 255.0
        else:
            image = image.float()

            if image.max() > 1:
                image = image / 255.0

        return image, target

class DetectionNormalize:
    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        self.mean = tuple(mean)
        self.std = tuple(std)

    def __call__(
        self,
        image: Tensor,
        target: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        image = F.normalize(image, mean=self.mean, std=self.std)

        return image, target