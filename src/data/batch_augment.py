"""Батч-уровневые аугментации: Mixup (arXiv:1710.09412) и CutMix (arXiv:1905.04899).

Почему отдельный модуль, а не ещё один трансформ в src/data/transforms.py:
трансформ внутри датасета видит РОВНО ОДИН пример, а Mixup и CutMix смешивают
разные примеры между собой. Поэтому они живут в тренере и применяются к уже
собранному батчу — после .to(device), то есть смешивание идёт на GPU и ничего
не стоит.

Контракт с лоссами проекта: forward принимает целочисленные метки, а не
soft-таргеты (см. src/losses/base.py). Ломать его ради mixup'а не нужно —
достаточно вернуть ДВА набора таргетов и вес lam, а тренер линейно
интерполирует значения лосса:

    loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)

Для кросс-энтропии это тождественно смешиванию one-hot таргетов, потому что
CE линейна по таргету. Дистилляционные слагаемые (KD, CWD, DIST, hint) от
меток вообще не зависят, поэтому в обеих ветках дают одно и то же значение,
и интерполяция их не меняет — смешивание не искажает дистилляцию.

Задача определяется по форме таргета, а не отдельным флагом в конфиге:
  targets [B]       — метки классификации;
  targets [B, H, W] — маски сегментации.
"""

import math
import random
from typing import NamedTuple

import torch
from torch import Tensor


class MixedBatch(NamedTuple):
    """Результат смешивания.

    lam == 1.0 означает "второй набор таргетов не нужен": либо смешивания не
    было вовсе, либо это CutMix по маскам, где таргет получился точным.
    Тренер по этому признаку пропускает второй вызов лосса.

    teacher_images — тот же батч, но со слабыми аугментациями (вид учителя,
    см. SegmentationTeacherViewCompose). Смешивается ровно теми же
    перестановкой и прямоугольником, что и вид ученика: иначе после CutMix
    два вида показывали бы разные сцены. None — вида учителя нет.
    """

    images: Tensor
    targets_a: Tensor
    targets_b: Tensor
    lam: float
    teacher_images: Tensor | None = None


def sample_lambda(alpha: float) -> float:
    """Вес исходного примера в смеси, lam ~ Beta(alpha, alpha), приведённый к [0.5, 1].

    Приведение нужно, чтобы доминирующим всегда оставался ИСХОДНЫЙ пример:
    тогда train-метрики, которые тренер считает по targets_a, остаются
    осмысленными. Распределение самих смесей от этого не меняется —
    Beta(alpha, alpha) симметрична, а пара для смешивания берётся случайной
    перестановкой батча, так что "кто из двоих назван первым" ни на что
    не влияет.
    """
    lam = random.betavariate(alpha, alpha)
    return max(lam, 1.0 - lam)


def random_bbox(height: int, width: int, lam: float) -> tuple[int, int, int, int]:
    """Прямоугольник (top, left, bottom, right) площадью (1 - lam) от кадра.

    Центр выбирается равномерно по всему кадру, поэтому прямоугольник может
    вылезти за границу и после обрезки оказаться меньше запрошенного — это
    поведение исходной статьи, и именно из-за него вызывающий код обязан
    пересчитать lam по фактической площади, а не доверять исходному.
    """
    ratio = math.sqrt(max(0.0, 1.0 - lam))
    box_height = int(height * ratio)
    box_width = int(width * ratio)

    center_y = random.randrange(height)
    center_x = random.randrange(width)

    top = max(center_y - box_height // 2, 0)
    left = max(center_x - box_width // 2, 0)
    bottom = min(center_y + box_height // 2, height)
    right = min(center_x + box_width // 2, width)

    return top, left, bottom, right


class MixupCutmix:
    """Mixup и CutMix над готовым батчем; работает и для классификации,
    и для семантической сегментации.

    Mixup смешивает две картинки линейно и требует от модели вести себя
    линейно между примерами. CutMix вырезает прямоугольник из одной картинки
    и вставляет в другую — «естественнее» mixup'а, потому что каждый пиксель
    результата остаётся настоящим пикселем настоящей картинки.

    Сегментация. CutMix переносит вместе с куском изображения и кусок маски,
    поэтому таргет получается ТОЧНЫМ (lam = 1.0, интерполяция лосса не нужна):
    это и есть стандартный CutMix для плотных задач. Mixup же для сегментации
    даёт полупрозрачное наложение двух сцен — таргет остаётся смешанным,
    и обучение идёт через интерполяцию лосса. Для Cityscapes разумный дефолт —
    только CutMix (mixup_alpha=0), см. configs/augment/.

    Args:
        mixup_alpha: параметр Beta для mixup; 0 — выключить mixup.
        cutmix_alpha: параметр Beta для cutmix; 0 — выключить cutmix.
        prob: вероятность вообще применить смешивание к батчу.
        switch_prob: если включены оба метода — вероятность выбрать cutmix.
    """

    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
        switch_prob: float = 0.5,
    ) -> None:
        if mixup_alpha < 0 or cutmix_alpha < 0:
            raise ValueError(
                f"alpha должны быть >= 0, получено mixup_alpha={mixup_alpha}, "
                f"cutmix_alpha={cutmix_alpha}"
            )
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob должна быть в [0, 1], получено {prob}")
        if not 0.0 <= switch_prob <= 1.0:
            raise ValueError(f"switch_prob должна быть в [0, 1], получено {switch_prob}")
        if mixup_alpha == 0 and cutmix_alpha == 0:
            raise ValueError(
                "Оба alpha нулевые — смешивать нечем. Чтобы отключить аугментацию, "
                "уберите её из конфига (augment=null), а не обнуляйте параметры."
            )

        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.prob = float(prob)
        self.switch_prob = float(switch_prob)

    def __call__(
        self,
        images: Tensor,
        targets: Tensor,
        teacher_images: Tensor | None = None,
    ) -> MixedBatch:
        if targets.ndim not in (1, 3):
            raise ValueError(
                "MixupCutmix ожидает таргет [B] (классификация) или [B, H, W] "
                f"(сегментация), получено {tuple(targets.shape)}"
            )
        # Батч из одного примера смешивать не с чем: перестановка тождественна.
        if images.size(0) < 2 or random.random() >= self.prob:
            return MixedBatch(images, targets, targets, 1.0, teacher_images)

        # Перестановка строится на CPU и переносится на устройство: так порядок
        # пар не зависит от того, на каком девайсе идёт обучение, и запуск
        # воспроизводится по seed'у одинаково на CPU и на GPU.
        permutation = torch.randperm(images.size(0)).to(images.device)

        if self._use_cutmix():
            return self._cutmix(images, targets, permutation, teacher_images)
        return self._mixup(images, targets, permutation, teacher_images)

    def _use_cutmix(self) -> bool:
        if self.mixup_alpha == 0.0:
            return True
        if self.cutmix_alpha == 0.0:
            return False
        return random.random() < self.switch_prob

    def _mixup(
        self,
        images: Tensor,
        targets: Tensor,
        permutation: Tensor,
        teacher_images: Tensor | None,
    ) -> MixedBatch:
        lam = sample_lambda(self.mixup_alpha)
        mixed = lam * images + (1.0 - lam) * images[permutation]
        if teacher_images is not None:
            teacher_images = lam * teacher_images + (1.0 - lam) * teacher_images[permutation]
        return MixedBatch(mixed, targets, targets[permutation], lam, teacher_images)

    def _cutmix(
        self,
        images: Tensor,
        targets: Tensor,
        permutation: Tensor,
        teacher_images: Tensor | None,
    ) -> MixedBatch:
        height, width = images.shape[-2:]
        top, left, bottom, right = random_bbox(height, width, sample_lambda(self.cutmix_alpha))

        mixed = images.clone()
        mixed[..., top:bottom, left:right] = images[permutation][..., top:bottom, left:right]

        if teacher_images is not None:
            teacher_mixed = teacher_images.clone()
            teacher_mixed[..., top:bottom, left:right] = (
                teacher_images[permutation][..., top:bottom, left:right]
            )
            teacher_images = teacher_mixed

        # Маски сегментации переносятся вместе с пикселями — таргет точный.
        if targets.ndim == 3:
            if targets.shape[-2:] != images.shape[-2:]:
                raise ValueError(
                    "CutMix для сегментации требует, чтобы маска и изображение совпадали "
                    f"по размеру: маска {tuple(targets.shape[-2:])}, "
                    f"изображение {tuple(images.shape[-2:])}"
                )
            mixed_targets = targets.clone()
            mixed_targets[:, top:bottom, left:right] = (
                targets[permutation][:, top:bottom, left:right]
            )
            return MixedBatch(mixed, mixed_targets, mixed_targets, 1.0, teacher_images)

        # Классификация: доля пикселей, оставшихся от исходной картинки.
        # Считается по фактической площади прямоугольника, а не по исходному
        # lam, потому что прямоугольник мог быть обрезан границей кадра.
        lam = 1.0 - (bottom - top) * (right - left) / (height * width)
        return MixedBatch(mixed, targets, targets[permutation], lam, teacher_images)


def interpolate_losses(
    losses_a: dict[str, Tensor],
    losses_b: dict[str, Tensor],
    lam: float,
) -> dict[str, Tensor]:
    """lam * losses_a + (1 - lam) * losses_b покомпонентно.

    Оба словаря приходят из одного и того же лосса на одних и тех же логитах,
    поэтому набор ключей совпадает по построению.
    """
    return {key: lam * value + (1.0 - lam) * losses_b[key] for key, value in losses_a.items()}
