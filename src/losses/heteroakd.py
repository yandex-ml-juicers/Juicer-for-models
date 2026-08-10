"""HeteroAKD — дистилляция между РАЗНОРОДНЫМИ архитектурами
(Wang et al., 2025, arXiv:2504.07691).
"""

import torch
import torch.nn.functional as F
from torch import nn

from src.losses.base import DistillationLoss
from src.losses.segmentation_utils import align_logits


def binary_divergence(target_logits: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Поэлементная KL между двумя бернуллиевскими распределениями.

    Считается через logsigmoid, а не через log(sigmoid(x)): сигмоида на
    больших по модулю логитах насыщается в 0 или 1, и логарифм от неё уходит
    в -inf, тогда как logsigmoid устойчив на всей оси.
    """
    log_target = F.logsigmoid(target_logits)
    log_not_target = F.logsigmoid(-target_logits)
    target = log_target.exp()

    return target * (log_target - F.logsigmoid(logits)) + (1.0 - target) * (
        log_not_target - F.logsigmoid(-logits)
    )


class FeatureProjector(nn.Module):
    """1x1-свёртка + BN: карта признаков -> карта «логитов» по классам.

    Проектор обучаемый и живёт внутри лосса (как регрессор FitNets), поэтому
    попадает в optimizer через criterion.parameters().

    В статье проектор описан как «1x1 conv с BN и ReLU». ReLU на выходе здесь
    нет намеренно: следом идёт sigmoid, и после ReLU он не смог бы опуститься
    ниже 0.5 — то есть проекция физически не могла бы сказать «этого класса
    здесь нет», а именно на таких высказываниях построены и KMM, и KEM.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.norm(self.project(features))


class HeteroAKDLoss(DistillationLoss):
    """total = ce_weight*CE + kd_weight*KD + hakd_weight*L_hakd + proj_weight*L_proj.

    Задача метода — дистилляция там, где учитель и ученик устроены по-разному
    (SegFormer -> U-Net): их промежуточные признаки живут в несопоставимых
    пространствах, и сравнивать их напрямую, как FitNets, значит заставлять
    ученика воспроизводить особенности чужой архитектуры, а не знание о сцене.

    Решение статьи — три шага.

    1. Общее пространство. Признаки обеих моделей проецируются 1x1-свёрткой
       в пространство КЛАССОВ (по каналу на класс). Всё, что специфично для
       архитектуры, при этом отбрасывается, а то, что относится к задаче,
       остаётся сравнимым поэлементно.

    2. KMM (Knowledge Mixing). Учитель не везде прав: там, где его проекция
       ошибается сильнее ученической, копировать её вредно. Надёжность
       меряется бинарной кросс-энтропией H с разметкой, и таргет собирается
       смесью двух источников:

           S_t = 1 - H(Z_t) / (H(Z_t) + H(Z_s)),
           Z_hybrid = S_t * Z_t + (1 - S_t) * Z_s

       То есть в пикселях, где учитель уверенно прав, таргет — его, а где он
       хуже ученика, таргет сдвигается к самому ученику и просто не тянет его
       в сторону ошибки.

    3. KEM (Knowledge Evaluation). Не всякий пиксель одинаково полезен: учить
       имеет смысл там, где ученик хуже собранного таргета. Разрыв
       dH = max(0, H(Z_s) - H(Z_hybrid)) превращается в веса softmax'ом по
       пространству, и дистилляционный член взвешивается ими.

    L_proj — супервизия самих проекторов разметкой (та же H). В статье она
    отдельным членом не выписана, но без неё «пространство логитов» ничем не
    закреплено за классами: обе проекции могли бы сойтись к любому общему
    представлению, и H перестала бы означать надёжность. Ставить proj_weight
    в 0 не стоит ещё и потому, что проектор учителя тогда не получит градиента
    вовсе (таргет отсоединён от графа), и под DDP это уронит шаг.
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        num_classes: int = 19,
        layer: str = "taps.stage4",
        temperature: float = 4.0,
        ce_weight: float = 1.0,
        kd_weight: float = 0.1,
        hakd_weight: float = 1.0,
        proj_weight: float = 1.0,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
    ) -> None:
        """
        Args:
            layer: канонический тап, с которого берутся признаки. В статье это
                последняя стадия бэкбона — taps.stage4 (страйд 32).
            kd_weight: вес обычной попиксельной KD по логитам (в статье 0.1).
        """
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature должна быть > 0, получено {temperature}")

        self.temperature = float(temperature)
        self.ce_weight = float(ce_weight)
        self.kd_weight = float(kd_weight)
        self.hakd_weight = float(hakd_weight)
        self.proj_weight = float(proj_weight)
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)
        self.num_classes = int(num_classes)

        self.layer = layer
        self.required_features = (layer,)

        self.student_projector = FeatureProjector(student_channels, num_classes)
        self.teacher_projector = FeatureProjector(teacher_channels, num_classes)

    def _targets_on(self, labels: torch.Tensor, size: torch.Size) -> tuple[torch.Tensor, torch.Tensor]:
        """One-hot разметка и маска валидности на сетке признаков."""
        labels = F.interpolate(labels.unsqueeze(1).float(), size=size, mode="nearest")
        labels = labels.squeeze(1).long()

        valid = (labels != self.ignore_index).unsqueeze(1).float()
        one_hot = F.one_hot(
            labels.masked_fill(labels == self.ignore_index, 0), self.num_classes
        )
        return one_hot.permute(0, 3, 1, 2).float(), valid

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        labels: torch.Tensor,
        student_features: dict | None = None,
        teacher_features: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        student, teacher = align_logits(student_logits, teacher_logits, "HeteroAKDLoss")
        if student_features is None or teacher_features is None:
            raise TypeError(
                "HeteroAKDLoss требует карты признаков обеих моделей (см. required_features)"
            )
        for name, features in (
            ("student_features", student_features),
            ("teacher_features", teacher_features),
        ):
            if self.layer not in features:
                raise KeyError(f"В {name} нет тапа {self.layer!r}: Trainer не снял его хуками")

        ce = F.cross_entropy(
            student,
            labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        temperature = self.temperature
        log_student = F.log_softmax(student / temperature, dim=1)
        log_teacher = F.log_softmax(teacher / temperature, dim=1)
        kd = (log_teacher.exp() * (log_teacher - log_student)).sum(dim=1).mean() * temperature**2

        # Проекции — в fp32: дальше идут логарифмы и softmax по пространству.
        projected_student = self.student_projector(student_features[self.layer].float())
        projected_teacher = self.teacher_projector(teacher_features[self.layer].detach().float())
        if projected_teacher.shape[2:] != projected_student.shape[2:]:
            projected_teacher = F.interpolate(
                projected_teacher,
                size=projected_student.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        one_hot, valid = self._targets_on(labels, projected_student.shape[2:])

        entropy_student = F.binary_cross_entropy_with_logits(
            projected_student, one_hot, reduction="none"
        )
        entropy_teacher = F.binary_cross_entropy_with_logits(
            projected_teacher, one_hot, reduction="none"
        )

        # KMM. Веса и сам таргет отсоединены от графа: это цель обучения,
        # а не ещё одна ветка, по которой ученик может подстроиться.
        with torch.no_grad():
            reliability = 1.0 - entropy_teacher / (entropy_teacher + entropy_student + 1e-6)
            hybrid = reliability * projected_teacher + (1.0 - reliability) * projected_student
            entropy_hybrid = F.binary_cross_entropy_with_logits(hybrid, one_hot, reduction="none")

            # KEM. Пиксель тем важнее, чем сильнее ученик отстаёт от таргета;
            # где не отстаёт — остаётся только его собственная неуверенность.
            gap = (entropy_student - entropy_hybrid).clamp(min=0.0)
            importance = entropy_student + gap
            # Пиксели вне разметки не участвуют: их вес обнуляется softmax'ом.
            importance = importance.masked_fill(valid.expand_as(importance) == 0, -torch.inf)
            weights = importance.flatten(2).softmax(dim=2).reshape(importance.shape)
            # Кадр целиком без разметки дал бы NaN на всех своих пикселях.
            weights = torch.nan_to_num(weights, nan=0.0)

        # Расхождение считается ПОКЛАССОВО и в той же логике, что и надёжность
        # выше: канал класса — это отдельное «есть/нет», то есть сигмоида, а не
        # softmax по классам. Иначе поклассовый вес W_c ломал бы сам смысл
        # величины: слагаемое одного класса в KL по softmax бывает
        # отрицательным, неотрицательна только их сумма, и взвешенная сумма
        # ушла бы ниже нуля.
        divergence = binary_divergence(hybrid / temperature, projected_student / temperature)
        # Веса суммируются в единицу по пикселям, поэтому sum по пространству —
        # это взвешенное среднее; дальше обычное среднее по классам и батчу.
        hakd = (divergence * weights).sum(dim=(2, 3)).mean() * temperature**2

        count = valid.sum().clamp(min=1.0) * self.num_classes
        projection = ((entropy_student + entropy_teacher) * valid).sum() / count

        total = (
            self.ce_weight * ce
            + self.kd_weight * kd
            + self.hakd_weight * hakd
            + self.proj_weight * projection
        )
        return {"total": total, "ce": ce, "kd": kd, "hakd": hakd, "proj": projection}
