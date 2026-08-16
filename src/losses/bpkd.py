"""BPKD — Boundary Privileged Knowledge Distillation (Liu et al., WACV 2024,
arXiv:2306.08075).
"""

import torch
import torch.nn.functional as F

from src.losses.base import DistillationLoss
from src.losses.segmentation_utils import align_logits, subsample_spatially


def boundary_mask(labels: torch.Tensor, width: int) -> torch.Tensor:
    """Маска границ [B, 1, H, W] по разметке: 1 там, где рядом разные классы.

    Морфологическая разность из статьи, dilation(GT) - erosion(GT), только
    без цикла по классам: расширение метки — это max-pool, сжатие — min-pool,
    и там, где они разошлись, в окно попало больше одной метки. Ширина полосы
    задаётся размером окна.

    Переход «класс / ignore» тоже считается границей, и это правильно:
    void-пиксели Cityscapes размечены как раз по контурам объектов.
    """
    if width < 3 or width % 2 == 0:
        raise ValueError(f"width должна быть нечётной и >= 3, получено {width}")

    values = labels.unsqueeze(1).float()
    padding = width // 2
    dilated = F.max_pool2d(values, width, stride=1, padding=padding)
    eroded = -F.max_pool2d(-values, width, stride=1, padding=padding)

    return (dilated != eroded).to(values.dtype)


class BPKDLoss(DistillationLoss):
    """total = ce_weight*CE + edge_weight*L_edge + body_weight*L_body.

    Идея статьи: границы и «тело» объекта требуют разного знания. В теле
    важна форма — крупная связная область одного класса; на границе важно
    попиксельное различение соседних классов, и именно там компактный ученик
    ошибается чаще всего, потому что у него меньше контекста и грубее
    признаки. Дистиллировать их одним лоссом — значит усреднять два разных
    требования.

    Отсюда две ветви, обе по логитам, но с softmax вдоль РАЗНЫХ осей:

    - L_edge — попиксельная KL по классам, усреднённая только по пикселям
      граничной полосы. Это вопрос «какой из двух соседних классов здесь»,
      заданный в каждой точке границы отдельно;
    - L_body — канальная KL (как в CWD): softmax по ПРОСТРАНСТВУ внутри
      канала, причём граничные пиксели из этого распределения исключены.
      Это вопрос «где именно лежит тело класса», то есть ограничение на форму.

    Отличие от статьи в нормировке: там маска умножается на логиты, а суммы
    берутся по всему кадру; здесь L_edge усредняется по числу граничных
    пикселей, а тело исключается из softmax маскированием. Для бинарной маски
    это то же самое с точностью до множителя, но веса становятся
    интерпретируемыми: они сравнимы с alpha у PixelWiseKD и cwd_weight у CWD,
    а не с 20/50 из статьи, где нормировка другая.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        ce_weight: float = 1.0,
        edge_weight: float = 3.0,
        body_weight: float = 3.0,
        edge_width: int = 7,
        spatial_stride: int = 1,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
    ) -> None:
        """
        Args:
            edge_width: ширина граничной полосы в пикселях (в статье 7).
            spatial_stride: прореживание пикселей перед подсчётом обоих членов.
                BPKD материализует вдвое больше полноразмерных тензоров, чем
                CWD (попиксельная ветка плюс маскированные копии для канальной),
                а на кропе 512x1024 каждый такой тензор — сотни мегабайт.
                stride=2 режет память вчетверо; полоса границы при ширине 7
                переживает прореживание, а статистики каналов не меняются.
        """
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature должна быть > 0, получено {temperature}")
        if spatial_stride < 1:
            raise ValueError(f"spatial_stride должен быть >= 1, получено {spatial_stride}")

        self.temperature = float(temperature)
        self.ce_weight = float(ce_weight)
        self.edge_weight = float(edge_weight)
        self.body_weight = float(body_weight)
        self.edge_width = int(edge_width)
        self.spatial_stride = int(spatial_stride)
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)

    def _edge_loss(
        self, student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        temperature = self.temperature
        log_student = F.log_softmax(student / temperature, dim=1)
        log_teacher = F.log_softmax(teacher / temperature, dim=1)

        per_pixel = (log_teacher.exp() * (log_teacher - log_student)).sum(dim=1, keepdim=True)
        return (per_pixel * mask).sum() / mask.sum().clamp(min=1.0) * temperature**2

    def _body_loss(
        self, student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        temperature = self.temperature
        batch_size, num_channels = student.shape[:2]

        # Граничные пиксели выбиваются из пространственного softmax: после
        # подстановки большого отрицательного числа их вес обращается в ноль,
        # и распределение канала описывает ровно тело класса.
        is_edge = mask > 0
        student = student.masked_fill(is_edge, -1e4).reshape(batch_size, num_channels, -1)
        teacher = teacher.masked_fill(is_edge, -1e4).reshape(batch_size, num_channels, -1)

        log_student = F.log_softmax(student / temperature, dim=2)
        log_teacher = F.log_softmax(teacher / temperature, dim=2)

        per_channel = (log_teacher.exp() * (log_teacher - log_student)).sum(dim=2)
        return per_channel.mean() * temperature**2

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        labels: torch.Tensor,
        student_features: dict | None = None,
        teacher_features: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        student, teacher = align_logits(student_logits, teacher_logits, "BPKDLoss")

        ce = F.cross_entropy(
            student,
            labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        mask = boundary_mask(labels, self.edge_width)
        if mask.shape[2:] != student.shape[2:]:
            mask = F.interpolate(mask, size=student.shape[2:], mode="nearest")

        # Прореживание — после построения маски: полоса границы считается по
        # полной разметке, иначе при stride > 1 она бы истончилась вдвое.
        student = subsample_spatially(student, self.spatial_stride)
        teacher = subsample_spatially(teacher, self.spatial_stride)
        mask = subsample_spatially(mask, self.spatial_stride)

        edge = self._edge_loss(student, teacher, mask)
        body = self._body_loss(student, teacher, mask)

        total = self.ce_weight * ce + self.edge_weight * edge + self.body_weight * body
        return {"total": total, "ce": ce, "edge": edge, "body": body}
