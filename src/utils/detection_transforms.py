import random
from collections.abc import Callable, Sequence
import random
import torch
from PIL import Image
from torch import Tensor
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

class DetectionHorizontalFlip:
    """Случайное горизонтальное отражение изображения и боксов."""
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self,
        image: Image.Image | Tensor,
        target: dict[str, Tensor],
    ) -> tuple[Image.Image | Tensor, dict[str, Tensor]]:
        if random.random() < self.p:
            # 1. Отражаем изображение
            if isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                width, _ = image.size
            else:
                image = F.hflip(image)
                width = image.shape[-1]

            # 2. Отражаем боксы [x_min, y_min, x_max, y_max]
            target = target.copy()
            boxes = target["boxes"].clone()
            
            if boxes.numel() > 0:
                # Новые координаты X:
                # x_min_new = width - x_max_old
                # x_max_new = width - x_min_old
                old_x_min = boxes[:, 0].clone()
                boxes[:, 0] = width - boxes[:, 2]
                boxes[:, 2] = width - old_x_min
                
                # Проверка корректности (на всякий случай)
                target["boxes"] = boxes

        return image, target
