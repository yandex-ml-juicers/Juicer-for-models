"""Feature-based дистилляция: логиты (как у Хинтона) + MSE по нормализованным
картам признаков промежуточных слоёв, с обучаемыми 1x1-адаптерами каналов."""

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from src.losses.base import DistillationLoss
from src.models.adapters import ChannelAdapters


def normalize_feature_map(feature_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Нормализация каждого объекта и канала по пространственным осям HxW.

    Убирает разницу масштабов активаций учителя и ученика: MSE сравнивает
    ФОРМУ карт признаков, а не их амплитуду.
    """
    mean = feature_map.mean(dim=(2, 3), keepdim=True)
    std = feature_map.std(dim=(2, 3), keepdim=True)
    return (feature_map - mean) / (std + eps)


class FeatureKD(DistillationLoss):
    """total = ce_weight * CE + logits_weight * KD + feature_weight * FeatMSE.

    FeatMSE — взвешенное среднее по слоям MSE между нормализованными картами;
    карты ученика прогоняются через обучаемые адаптеры (см. ChannelAdapters).
    Если пространственные размеры не совпадают, карта ученика интерполируется
    к размеру учителя (bilinear).

    ВНИМАНИЕ: у bilinear-интерполяции нет детерминированного CUDA-ядра для
    backward. Если размеры слоёв совпадают (наша пара ResNet50/ResNet18 под
    CIFAR), ветка не исполняется и детерминизм не страдает; если не совпадают —
    под deterministic=true запуск упадёт с RuntimeError, и это осознанное
    поведение (лучше явный отказ, чем тихая невоспроизводимость).
    """

    def __init__(
        self,
        layers: Mapping[str, Mapping],
        temperature: float = 4.0,
        ce_weight: float = 1.0,
        logits_weight: float = 1.0,
        feature_weight: float = 1.0,
    ) -> None:
        """
        Args:
            layers: {имя_слоя: {"student_channels": int, "teacher_channels": int,
                "weight": float}} — какие слои дистиллировать и с каким весом.
        """
        super().__init__()
        self.temperature = temperature
        self.ce_weight = ce_weight
        self.logits_weight = logits_weight
        self.feature_weight = feature_weight

        self.layer_weights = {name: float(spec.get("weight", 1.0)) for name, spec in layers.items()}
        self.adapters = ChannelAdapters(
            {name: (spec["student_channels"], spec["teacher_channels"]) for name, spec in layers.items()}
        )
        self.required_features = tuple(layers)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        labels: torch.Tensor,
        student_features: dict | None = None,
        teacher_features: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        if teacher_logits is None:
            raise TypeError("FeatureKD требует логиты учителя (requires_teacher=True)")
        if student_features is None or teacher_features is None:
            raise TypeError(
                "FeatureKD требует карты признаков обеих моделей (см. required_features)"
            )

        temperature = self.temperature
        ce = F.cross_entropy(student_logits, labels)

        soft_student = F.log_softmax(student_logits / temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits.detach() / temperature, dim=1)
        kd = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * temperature**2

        feature_loss, per_layer = self._feature_loss(student_features, teacher_features)

        total = self.ce_weight * ce + self.logits_weight * kd + self.feature_weight * feature_loss
        result = {"total": total, "ce": ce, "kd": kd, "feature": feature_loss}
        result.update({f"feature_{name}": loss for name, loss in per_layer.items()})
        return result

    def _feature_loss(
        self, student_features: dict, teacher_features: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        missing = [name for name in self.required_features if name not in student_features]
        if missing:
            raise KeyError(
                f"В student_features нет слоёв {missing}: Trainer не снял их хуками "
                f"(проверь required_features и имена слоёв в конфиге лосса)"
            )

        adapted = self.adapters(student_features)
        total = torch.zeros((), device=next(self.adapters.parameters()).device)
        per_layer: dict[str, torch.Tensor] = {}

        for layer_name, student_map in adapted.items():
            teacher_map = teacher_features[layer_name].detach()
            if student_map.shape[2:] != teacher_map.shape[2:]:
                student_map = F.interpolate(
                    student_map, size=teacher_map.shape[2:], mode="bilinear", align_corners=False
                )
            layer_loss = F.mse_loss(
                normalize_feature_map(student_map), normalize_feature_map(teacher_map)
            )
            per_layer[layer_name] = layer_loss
            total = total + self.layer_weights[layer_name] * layer_loss

        return total / sum(self.layer_weights.values()), per_layer
