"""Stochastic Depth / DropPath (Huang et al., ECCV 2016, arXiv:1603.09382).

Во время обучения residual-блок с вероятностью p целиком выключается: его
ветка обнуляется, и от блока остаётся чистый skip-connection. Это dropout
на уровне слоёв, а не нейронов; на инференсе выключений нет вовсе.

Кому это нужно в проекте:
- SegFormer и timm-модели умеют stochastic depth САМИ — им достаточно
  прокинуть drop_path_rate в конструктор (см. src/models/factory.py),
  и этот модуль для них не используется;
- torchvision-ResNet'ы (resnet18/50/152 в роли учеников и учителей
  классификации) такой опции не имеют — вот их и патчит apply_stochastic_depth.

U-Net из src/models/unet.py residual-блоков не содержит вовсе: там нет
сложения ветки со входом, только конкатенация skip-связей в декодере.
Выключать в нём нечего — регуляризовать его нужно dropout'ом (параметр
dropout у unet_for_segmentation), а не stochastic depth.
"""

from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.ops import StochasticDepth


def linear_drop_path_rates(num_blocks: int, drop_path_rate: float) -> list[float]:
    """Линейно растущие вероятности выключения: 0 у первого блока,
    drop_path_rate у последнего.

    Так делают все современные рецепты (ViT, Swin, ConvNeXt), и не случайно:
    ранние блоки строят низкоуровневые признаки, на которых держится вся сеть,
    и выключать их вредно; поздние блоки избыточны, и именно там регуляризация
    даёт эффект.
    """
    if num_blocks <= 0:
        raise ValueError(f"num_blocks должен быть > 0, получено {num_blocks}")
    if num_blocks == 1:
        return [drop_path_rate]

    return [drop_path_rate * index / (num_blocks - 1) for index in range(num_blocks)]


class StochasticDepthBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d, за которым сразу применяется drop-path.

    Почему подменяется именно BatchNorm, а не вставляется отдельный модуль:
    имена параметров обязаны остаться прежними. В torchvision-ResNet блок
    считает out = relu(F(x) + identity(x)), где F заканчивается последним
    BatchNorm'ом (bn2 у BasicBlock, bn3 у Bottleneck). Если обернуть этот BN
    в nn.Sequential(bn, StochasticDepth), ключи state_dict поедут
    (bn2.weight -> bn2.0.weight), и чекпоинт, обученный со stochastic depth,
    перестанет грузиться в модель без него — а именно так в проекте ученик
    первого этапа становится учителем второго (load_checkpoint_into,
    strict=True). Подкласс BatchNorm2d этой проблемы не создаёт: имена те же,
    а StochasticDepth своих параметров и буферов не имеет.

    Умножать на нули НЕ эквивалентно "ничего не делать": StochasticDepth
    в режиме row делит выжившие примеры на (1 - p), чтобы матожидание выхода
    совпадало с инференсом.
    """

    def __init__(self, num_features: int, drop_prob: float, **kwargs) -> None:
        super().__init__(num_features, **kwargs)
        self.drop_path = StochasticDepth(float(drop_prob), mode="row")

    @classmethod
    def from_batch_norm(cls, norm: nn.BatchNorm2d, drop_prob: float) -> "StochasticDepthBatchNorm2d":
        """Копия существующего BatchNorm2d с тем же состоянием и drop-path сверху."""
        replacement = cls(
            norm.num_features,
            drop_prob=drop_prob,
            eps=norm.eps,
            momentum=norm.momentum,
            affine=norm.affine,
            track_running_stats=norm.track_running_stats,
        )
        replacement.load_state_dict(norm.state_dict())

        # Замена должна оказаться на том же устройстве и в том же dtype, что
        # оригинал. У BatchNorm с affine=False и track_running_stats=False нет
        # ни параметров, ни буферов — тогда переносить просто нечего.
        reference = next(iter(norm.parameters()), None)
        if reference is None:
            reference = next(iter(norm.buffers()), None)
        if reference is not None:
            replacement = replacement.to(device=reference.device, dtype=reference.dtype)

        return replacement

    def forward(self, inputs: Tensor) -> Tensor:
        return self.drop_path(super().forward(inputs))


def residual_tail_name(module: nn.Module) -> str | None:
    """Имя последнего BatchNorm'а residual-ветки блока, или None если это не блок."""
    if isinstance(module, Bottleneck):
        return "bn3"
    if isinstance(module, BasicBlock):
        return "bn2"
    return None


def apply_stochastic_depth(
    model: nn.Module,
    drop_path_rate: float,
    mode: str = "linear",
) -> nn.Module:
    """Включает stochastic depth во всех residual-блоках torchvision-ResNet'а.

    Модель правится на месте и возвращается для удобства цепочки вызовов.

    Args:
        drop_path_rate: максимальная вероятность выключения блока. Рабочие
            значения — 0.05..0.2 для ResNet-18/50; 0.3+ обычно уже мешает
            сходимости на коротких расписаниях.
        mode: "linear" — рейт растёт от 0 к drop_path_rate по глубине
            (рекомендуется); "uniform" — один и тот же рейт во всех блоках,
            как в исходной статье.

    Raises:
        ValueError: если в модели нет поддерживаемых блоков. Молча ничего не
            делать нельзя: конфиг с drop_path_rate=0.1 на ShuffleNet выглядел бы
            рабочим, а регуляризации в обучении не было бы вовсе.
    """
    if not 0.0 <= drop_path_rate < 1.0:
        raise ValueError(f"drop_path_rate должен быть в [0, 1), получено {drop_path_rate}")
    if mode not in {"linear", "uniform"}:
        raise ValueError(f"mode должен быть 'linear' или 'uniform', получено {mode!r}")

    blocks = [module for module in model.modules() if residual_tail_name(module) is not None]
    if not blocks:
        raise ValueError(
            "В модели не найдено residual-блоков torchvision (BasicBlock/Bottleneck), "
            "поэтому stochastic depth применить не к чему. У timm-моделей и SegFormer "
            "для этого есть собственный аргумент drop_path_rate."
        )

    rates = (
        linear_drop_path_rates(len(blocks), drop_path_rate)
        if mode == "linear"
        else [drop_path_rate] * len(blocks)
    )

    for block, rate in zip(blocks, rates):
        name = residual_tail_name(block)
        norm = getattr(block, name)
        if not isinstance(norm, nn.BatchNorm2d):
            raise TypeError(
                f"Ожидался BatchNorm2d в {type(block).__name__}.{name}, получено "
                f"{type(norm).__name__}: модель собрана с нестандартным norm_layer, "
                "и точка вставки drop-path для неё неизвестна."
            )
        setattr(block, name, StochasticDepthBatchNorm2d.from_batch_norm(norm, rate))

    return model
