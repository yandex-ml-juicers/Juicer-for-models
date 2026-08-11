"""FitNets hint loss (Romero et al., ICLR 2015, arXiv:1412.6550)."""

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from src.losses.base import DistillationLoss
from src.losses.segmentation_utils import align_logits
from src.models.adapters import ChannelAdapters


class FitNetsKD(DistillationLoss):
    """total = ce_weight * CE + hint_weight * MSE(regressor(F_student), F_teacher).

    Идея статьи: дистиллировать не только ответ, но и промежуточное
    представление. Слой ученика (guided layer) обязан воспроизвести карту
    признаков слоя учителя (hint layer); поскольку размерности не совпадают,
    между ними ставится обучаемый регрессор — здесь 1x1-свёртка из
    ChannelAdapters, живущая внутри лосса и попадающая в optimizer через
    criterion.parameters().

    В оригинале это первая из двух стадий (сначала прогрев по hint'у, потом
    обычная KD). Здесь hint считается совместно с CE — так метод обычно и
    воспроизводят, и так он сравним с остальными лоссами в одной сетке
    экспериментов. Член KD по логитам выключен по умолчанию
    (logits_weight=0), чтобы конфиг был честной абляцией именно hint-лосса;
    поставьте вес > 0, если нужен вариант "FitNets + Hinton".

    Разные архитектуры (SegFormer -> U-Net) сравниваются через
    канонические тапы taps.stageN (см. src/models/feature_taps.py):
    внутренние имена слоёв у моделей разные, а тапы — общие.
    """

    def __init__(
        self,
        layers: Mapping[str, Mapping],
        ce_weight: float = 1.0,
        hint_weight: float = 1.0,
        logits_weight: float = 0.0,
        temperature: float = 4.0,
        normalize_features: bool = False,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
    ) -> None:
        """
        Args:
            layers: {имя_тапа: {"student_channels": int, "teacher_channels": int,
                "weight": float}} — какие стадии сравнивать и с каким весом.
            normalize_features: нормировать карты по HxW перед MSE. В статье
                нормировки нет (default False), но она спасает, если масштабы
                активаций учителя и ученика разошлись на порядки.
        """
        super().__init__()
        if not layers:
            raise ValueError("FitNetsKD требует хотя бы один слой в layers")

        self.ce_weight = float(ce_weight)
        self.hint_weight = float(hint_weight)
        self.logits_weight = float(logits_weight)
        self.temperature = float(temperature)
        self.normalize_features = bool(normalize_features)
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)

        self.layer_weights = {name: float(spec.get("weight", 1.0)) for name, spec in layers.items()}

        # nn.ModuleDict запрещает точку в ключе, а имена тапов — это пути
        # модулей ("taps.stage3"). Держим отдельное соответствие
        # "имя тапа -> ключ адаптера" вместо того, чтобы менять ChannelAdapters:
        # на нём висит уже работающий FeatureKD.
        self.adapter_keys = {name: name.replace(".", "__") for name in layers}
        self.adapters = ChannelAdapters(
            {
                self.adapter_keys[name]: (spec["student_channels"], spec["teacher_channels"])
                for name, spec in layers.items()
            }
        )
        self.required_features = tuple(layers)

    @staticmethod
    def _normalize(feature_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mean = feature_map.mean(dim=(2, 3), keepdim=True)
        std = feature_map.std(dim=(2, 3), keepdim=True)
        return (feature_map - mean) / (std + eps)

    def _hint_loss(
        self, student_features: dict, teacher_features: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        missing = [name for name in self.required_features if name not in student_features]
        if missing:
            raise KeyError(
                f"В student_features нет тапов {missing}: Trainer не снял их хуками "
                f"(проверь required_features и имена тапов в конфиге лосса)"
            )
        missing = [name for name in self.required_features if name not in teacher_features]
        if missing:
            raise KeyError(f"В teacher_features нет тапов {missing}")

        # Адаптеры считаем в fp32: под AMP карты приходят в fp16, а MSE
        # по ним копит ошибку округления на миллионах элементов.
        adapted = self.adapters(
            {
                self.adapter_keys[name]: student_features[name].float()
                for name in self.required_features
            }
        )

        total = torch.zeros((), device=next(self.adapters.parameters()).device)
        per_layer: dict[str, torch.Tensor] = {}

        for layer_name in self.required_features:
            student_map = adapted[self.adapter_keys[layer_name]]
            teacher_map = teacher_features[layer_name].detach().float()

            if student_map.shape[2:] != teacher_map.shape[2:]:
                student_map = F.interpolate(
                    student_map, size=teacher_map.shape[2:], mode="bilinear", align_corners=False
                )

            if self.normalize_features:
                student_map = self._normalize(student_map)
                teacher_map = self._normalize(teacher_map)

            layer_loss = F.mse_loss(student_map, teacher_map)
            per_layer[layer_name] = layer_loss
            total = total + self.layer_weights[layer_name] * layer_loss

        return total / sum(self.layer_weights.values()), per_layer

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        labels: torch.Tensor,
        student_features: dict | None = None,
        teacher_features: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        student, teacher = align_logits(student_logits, teacher_logits, "FitNetsKD")
        if student_features is None or teacher_features is None:
            raise TypeError(
                "FitNetsKD требует карты признаков обеих моделей (см. required_features)"
            )

        ce = F.cross_entropy(
            student,
            labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        hint, per_layer = self._hint_loss(student_features, teacher_features)

        total = self.ce_weight * ce + self.hint_weight * hint
        result = {"total": total, "ce": ce, "hint": hint}

        if self.logits_weight > 0.0:
            temperature = self.temperature
            log_student = F.log_softmax(student / temperature, dim=1)
            log_teacher = F.log_softmax(teacher / temperature, dim=1)
            kd = (log_teacher.exp() * (log_teacher - log_student)).sum(dim=1).mean()
            kd = kd * temperature**2
            result["kd"] = kd
            result["total"] = total + self.logits_weight * kd

        result.update({f"hint_{name}": value for name, value in per_layer.items()})
        return result
