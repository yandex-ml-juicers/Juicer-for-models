"""Общие детали дистилляционных лоссов семантической сегментации.

Три вещи повторяются в каждом из них и вынесены сюда, чтобы не разъезжались
между реализациями:

1. Приведение логитов учителя к сетке ученика. Наши сегментаторы (SegFormer,
   U-Net) отдают логиты в разрешении входа, поэтому размеры обычно
   совпадают — но полагаться на это нельзя: любая модель с другим страйдом
   головы молча сломала бы поэлементное сравнение.

2. Перевод в fp32. Под AMP логиты приходят в fp16, а KL и корреляции Пирсона
   считают логарифмы и делят на нормы — в половинной точности это даёт
   NaN на ровном месте.

3. Отсоединение учителя от графа. Учитель заморожен тренером, но detach()
   здесь — это ещё и явная гарантия, что через него не потечёт градиент.


Про ignore_index: он применяется ТОЛЬКО к кросс-энтропии. Дистилляционные
члены считаются по всем пикселям, включая void — на них у учителя есть
осмысленное распределение, и это часть того, зачем дистилляция нужна.
"""

import torch
import torch.nn.functional as F


def align_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    loss_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Возвращает (логиты ученика fp32, логиты учителя fp32 на сетке ученика)."""
    if teacher_logits is None:
        raise TypeError(f"{loss_name} требует логиты учителя (requires_teacher=True)")

    student = student_logits.float()
    teacher = teacher_logits.detach().float()

    if teacher.shape[2:] != student.shape[2:]:
        teacher = F.interpolate(
            teacher, size=student.shape[2:], mode="bilinear", align_corners=False
        )

    if teacher.shape[1] != student.shape[1]:
        raise ValueError(
            f"{loss_name}: у ученика {student.shape[1]} классов, у учителя {teacher.shape[1]}. "
            f"Дистилляция логитов требует одинакового набора классов."
        )

    return student, teacher


def subsample_spatially(tensor: torch.Tensor, stride: int) -> torch.Tensor:
    """Прореживание по H и W с шагом stride (stride=1 — без изменений).

    Нужно там, где лосс материализует несколько тензоров размера логитов:
    на кропе 512x1024 карта [B, 19, H, W] весит сотни мегабайт, и пара
    промежуточных копий съедает память быстрее, чем сама модель. Статистики
    по 130 тысячам пикселей вместо 520 тысяч не меняются.
    """
    if stride <= 1:
        return tensor
    return tensor[..., ::stride, ::stride]


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    return (a * b).sum(dim) / (a.norm(dim=dim) * b.norm(dim=dim) + eps)


def pearson_correlation(
    a: torch.Tensor, b: torch.Tensor, dim: int, eps: float = 1e-8
) -> torch.Tensor:
    """Корреляция Пирсона вдоль оси dim = косинус после центрирования."""
    return cosine_similarity(
        a - a.mean(dim=dim, keepdim=True),
        b - b.mean(dim=dim, keepdim=True),
        dim=dim,
        eps=eps,
    )
