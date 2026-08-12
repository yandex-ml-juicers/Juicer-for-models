import random
from collections.abc import Callable, Sequence

import cv2
import numpy as np

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as F

from ultralytics.data.augment import RandomHSV

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

class DetectionRandomResizedCrop:
    """Random crop followed by resize to the given output size."""

    def __init__(
        self,
        size: tuple[int, int],
        scale: tuple[float, float] = (0.5, 1.0),
        ratio: tuple[float, float] = (0.75, 1.33),
        max_removed_fraction: float = 0.75,
        max_attempts: int = 10,
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.max_removed_fraction = max_removed_fraction
        self.max_attempts = max_attempts

    def __call__(
        self,
        image: Image.Image | Tensor,
        target: dict[str, Tensor],
    ) -> tuple[Image.Image | Tensor, dict[str, Tensor]]:

        original_boxes = target["boxes"]
        num_objects = len(original_boxes)

        # If there are no objects, use regular random crop
        if num_objects == 0:
            top, left, height, width = transforms.RandomResizedCrop.get_params(image, scale=self.scale, ratio=self.ratio)
            image = F.crop(image, top=top, left=left, height=height, width=width)

            target = target.copy()
            target["size"] = torch.tensor([height, width], dtype=torch.int64)

            return DetectionResize(self.size)(image, target)

        for _ in range(self.max_attempts):
            top, left, height, width = transforms.RandomResizedCrop.get_params(image, scale=self.scale, ratio=self.ratio)
            boxes = original_boxes.clone()

            # Move boxes into cropped image coordinates
            boxes[:, [0, 2]] -= left
            boxes[:, [1, 3]] -= top

            # Clip boxes to crop borders
            boxes[:, [0, 2]].clamp_(0, width)
            boxes[:, [1, 3]].clamp_(0, height)

            # Remove boxes that disappeared after crop
            keep = ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))

            num_remaining = keep.sum().item()
            removed_fraction = 1.0 - num_remaining / num_objects

            # Reject crop if more than 75% of objects disappeared
            if removed_fraction > self.max_removed_fraction:
                continue

            # Apply accepted crop
            image = F.crop(image, top=top, left=left, height=height, width=width)
            target = target.copy()
            boxes = boxes[keep]

            target["boxes"] = boxes
            target["labels"] = target["labels"][keep]

            if "iscrowd" in target:
                target["iscrowd"] = target["iscrowd"][keep]

            if "area" in target:
                target["area"] = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))

            target["size"] = torch.tensor([height, width], dtype=torch.int64)

            return DetectionResize(self.size)(image, target)

        # If suitable crop was not found, resize original image
        return DetectionResize(self.size)(image, target)

class DetectionFilterBoxes:
    """Выбрасывает боксы тоньше min_size пикселей по любой из сторон.

    После кропа рамка объекта на краю кадра обрезается по границе, и от неё
    остаётся полоска в доли пикселя: предсказать её нельзя, а ассайнер YOLO
    считает её обычным положительным примером.
    """

    def __init__(self, min_size: float = 2.0) -> None:
        self.min_size = min_size

    def __call__(
        self,
        image: Image.Image | Tensor,
        target: dict[str, Tensor],
    ) -> tuple[Image.Image | Tensor, dict[str, Tensor]]:
        boxes = target["boxes"]

        if boxes.numel() == 0:
            return image, target

        keep = ((boxes[:, 2] - boxes[:, 0]) >= self.min_size) & (
            (boxes[:, 3] - boxes[:, 1]) >= self.min_size
        )

        target = target.copy()
        for key in ("boxes", "labels", "area", "iscrowd"):
            if key in target:
                target[key] = target[key][keep]

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

class DetectionRandomHSV:
    """Random HSV adjustment using Ultralytics implementation."""

    def __init__(
        self,
        hgain: float = 0.015,
        sgain: float = 0.7,
        vgain: float = 0.4,
        p: float = 1.0,
    ) -> None:
        self.transform = RandomHSV(hgain=hgain, sgain=sgain, vgain=vgain)
        self.p = p

    def __call__(
        self,
        image: Image.Image | Tensor,
        target: dict[str, Tensor],
    ) -> tuple[Image.Image | Tensor, dict[str, Tensor]]:
        if random.random() >= self.p:
            return image, target

        is_pil = isinstance(image, Image.Image)

        if is_pil:
            image_np = np.asarray(image.convert("RGB"))
        else:
            image_tensor = image.detach().cpu()

            if image_tensor.dtype.is_floating_point:
                image_np = (image_tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            else:
                image_np = image_tensor.permute(1, 2, 0).numpy()

        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_bgr = self.transform({"img": image_bgr})["img"]
        image_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if is_pil:
            image = Image.fromarray(image_np)
        else:
            image = torch.from_numpy(image_np).permute(2, 0, 1)

            if image_tensor.dtype.is_floating_point:
                image = image.to(dtype=image_tensor.dtype) / 255.0
            else:
                image = image.to(dtype=image_tensor.dtype)

        return image, target

class DetectionGaussianBlur:
    def __init__(
        self,
        kernel_size: int = 5,
        sigma: tuple[float, float] = (0.1, 2.0),
        p: float = 0.2,
    ) -> None:
        self.transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.p = p

    def __call__(
        self,
        image: Image.Image | Tensor,
        target: dict[str, Tensor],
    ):
        if random.random() < self.p:
            image = self.transform(image)

        return image, target

class DetectionRandomTranslation:
    """Randomly translates image and bounding boxes."""

    def __init__(
        self,
        translate: float = 0.1,
        p: float = 1.0,
    ) -> None:
        self.translate = translate
        self.p = p

    def __call__(
        self,
        image: Image.Image | Tensor,
        target: dict[str, Tensor],
    ) -> tuple[Image.Image | Tensor, dict[str, Tensor]]:
        if random.random() >= self.p:
            return image, target

        if isinstance(image, Image.Image):
            width, height = image.size
            fill = [114, 114, 114]
        else:
            height, width = image.shape[-2:]
            fill_value = 114 / 255 if image.dtype.is_floating_point else 114
            fill = [fill_value] * image.shape[-3]

        translate_x = int(random.uniform(-self.translate, self.translate) * width)
        translate_y = int(random.uniform(-self.translate, self.translate) * height)

        image = F.affine(
            image,
            angle=0.0,
            translate=[translate_x, translate_y],
            scale=1.0,
            shear=[0.0, 0.0],
            fill=fill,
        )

        target = target.copy()
        boxes = target["boxes"].clone()

        if boxes.numel() > 0:
            boxes[:, [0, 2]] += translate_x
            boxes[:, [1, 3]] += translate_y

            boxes[:, [0, 2]].clamp_(0, width)
            boxes[:, [1, 3]].clamp_(0, height)

            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

            boxes = boxes[keep]
            target["labels"] = target["labels"][keep]

            if "iscrowd" in target:
                target["iscrowd"] = target["iscrowd"][keep]

            if "area" in target:
                target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        target["boxes"] = boxes

        return image, target

class DetectionRandomMosaic:
    """Combines four detection samples into one image."""

    def __init__(
        self,
        sample_getter: Callable,
        dataset_size: int,
        p: float = 1.0,
        center_range: tuple[float, float] = (0.25, 0.75),
    ) -> None:
        self.sample_getter = sample_getter
        self.dataset_size = dataset_size
        self.p = p
        self.center_range = center_range

    def __call__(
        self,
        image: Image.Image | Tensor,
        target: dict[str, Tensor],
    ) -> tuple[Image.Image | Tensor, dict[str, Tensor]]:
        if random.random() >= self.p:
            return image, target

        is_pil = isinstance(image, Image.Image)

        if is_pil:
            width, height = image.size
        else:
            height, width = image.shape[-2:]

        samples = [(image, target)]

        for _ in range(3):
            index = random.randrange(self.dataset_size)
            samples.append(self.sample_getter(index))

        xc = int(random.uniform(*self.center_range) * width)
        yc = int(random.uniform(*self.center_range) * height)

        tensor_samples = []

        for sample_image, sample_target in samples:
            if isinstance(sample_image, Image.Image):
                sample_width, sample_height = sample_image.size
                sample_image = F.pil_to_tensor(sample_image)
            else:
                sample_height, sample_width = sample_image.shape[-2:]

            if sample_height != height or sample_width != width:
                raise ValueError(
                    f"Mosaic expects equal image sizes, got {(sample_height, sample_width)} and {(height, width)}"
                )

            tensor_samples.append((sample_image, sample_target))

        canvas = torch.empty_like(tensor_samples[0][0])

        placements = [
            (0, 0, xc, yc, width - xc, height - yc, width, height),
            (xc, 0, width, yc, 0, height - yc, width - xc, height),
            (0, yc, xc, height, width - xc, 0, width, height - yc),
            (xc, yc, width, height, 0, 0, width - xc, height - yc),
        ]

        all_boxes = []
        all_labels = []
        all_iscrowd = []

        for (sample_image, sample_target), placement in zip(tensor_samples, placements):
            dx1, dy1, dx2, dy2, sx1, sy1, sx2, sy2 = placement

            canvas[:, dy1:dy2, dx1:dx2] = sample_image[:, sy1:sy2, sx1:sx2]

            boxes = sample_target["boxes"].clone()
            labels = sample_target["labels"].clone()

            if boxes.numel() == 0:
                continue

            boxes[:, [0, 2]] += dx1 - sx1
            boxes[:, [1, 3]] += dy1 - sy1

            boxes[:, [0, 2]].clamp_(dx1, dx2)
            boxes[:, [1, 3]].clamp_(dy1, dy2)

            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

            boxes = boxes[keep]
            labels = labels[keep]

            all_boxes.append(boxes)
            all_labels.append(labels)

            if "iscrowd" in sample_target:
                all_iscrowd.append(sample_target["iscrowd"][keep])

        target = target.copy()

        if all_boxes:
            boxes = torch.cat(all_boxes, dim=0)
            labels = torch.cat(all_labels, dim=0)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.long)

        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target["size"] = torch.tensor([height, width], dtype=torch.int64)

        if all_iscrowd:
            target["iscrowd"] = torch.cat(all_iscrowd, dim=0)
        elif "iscrowd" in target:
            target["iscrowd"] = torch.zeros(len(boxes), dtype=torch.int64)

        if is_pil:
            image = F.to_pil_image(canvas)
        else:
            image = canvas

        return image, target

class DetectionRandomScale:
    def __init__(self, scale: float = 0.5, p: float = 1.0) -> None:
        self.scale = scale
        self.p = p

    def __call__(self, image: Image.Image | Tensor, target: dict[str, Tensor]) -> tuple[Image.Image | Tensor, dict[str, Tensor]]:
        if random.random() >= self.p:
            return image, target

        if isinstance(image, Image.Image):
            width, height = image.size
        else:
            height, width = image.shape[-2:]

        scale = random.uniform(1.0 - self.scale, 1.0 + self.scale)

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        image = F.resize(image, [new_height, new_width])

        target = target.copy()
        boxes = target["boxes"].clone()

        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= new_width / width
            boxes[:, [1, 3]] *= new_height / height

        target["boxes"] = boxes
        target["size"] = torch.tensor([new_height, new_width], dtype=torch.int64)

        if "area" in target:
            target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

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