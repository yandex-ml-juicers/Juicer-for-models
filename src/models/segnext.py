"""SegNeXt (Guo et al., NeurIPS 2022, arXiv:2209.08575): энкодер MSCAN +
декодер LightHamHead.

Зачем он в проекте. Это свёрточная сеть, которая на Cityscapes догоняет и
обгоняет трансформеры своего размера (MSCAN-T: 4M параметров, 79.8 mIoU при
одном масштабе), поэтому она интересна с обеих сторон дистилляции: как
сильный компактный УЧЕНИК и как учитель, отличный по устройству от SegFormer.

Почему архитектура написана здесь, а не взята из библиотеки. Оригинальная
реализация живёт в mmsegmentation и тянет за собой mmcv/mmengine с их
реестрами и собственным способом собирать модель из конфига; в timm и
transformers MSCAN нет вовсе. Здесь воспроизведены ровно два модуля, зато
модель подчиняется контракту проекта (forward -> логиты в разрешении входа,
канонические тапы taps.stage1..4).

ИМЕНА ПАРАМЕТРОВ намеренно повторяют mmsegmentation (patch_embed1..4,
block{i}.{j}.attn.spatial_gating_unit.conv0_1, squeeze/hamburger/align,
conv_seg): это и есть способ загружать опубликованные веса — см.
convert_mmseg_state_dict.
"""

import math
from collections import OrderedDict
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.models.feature_taps import STAGE_TAPS, FeatureTaps

# Спецификации из статьи (Table 6) и конфигов mmsegmentation.
SEGNEXT_VARIANTS: dict[str, dict] = {
    "t": {"embed_dims": [32, 64, 160, 256], "depths": [3, 3, 5, 2], "decoder_channels": 256,
          "drop_path_rate": 0.1},
    "s": {"embed_dims": [64, 128, 320, 512], "depths": [2, 2, 4, 2], "decoder_channels": 256,
          "drop_path_rate": 0.1},
    "b": {"embed_dims": [64, 128, 320, 512], "depths": [3, 3, 12, 3], "decoder_channels": 512,
          "drop_path_rate": 0.1},
    "l": {"embed_dims": [64, 128, 320, 512], "depths": [3, 5, 27, 3], "decoder_channels": 1024,
          "drop_path_rate": 0.3},
}

# Энкодеры MSCAN, предобученные на ImageNet (конвертация OpenMMLab).
# Головы декодера в этих файлах нет — она инициализируется случайно, ровно
# как в рецепте самой статьи.
IMAGENET_WEIGHTS: dict[str, str] = {
    "t": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth",
    "s": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_s_20230227-f33ccdf2.pth",
    "b": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_b_20230227-3ab7d230.pth",
    "l": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_l_20230227-cef260d4.pth",
}


class ConvNorm(nn.Sequential):
    """conv + norm (+ act) с именами подмодулей как у mmcv.ConvModule.

    В ConvModule свёртка всегда зовётся conv, а норма — по своему типу: bn
    для BatchNorm, gn для GroupNorm. Имена важны только для загрузки чужих
    весов, но ради неё модуль и написан.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str | None = "gn",
        num_groups: int = 32,
        activation: bool = True,
    ) -> None:
        # Смещение у свёртки есть ровно тогда, когда за ней не идёт норма —
        # то же правило bias='auto', что и в ConvModule.
        layers = OrderedDict(
            conv=nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=norm is None)
        )
        if norm == "gn":
            layers["gn"] = nn.GroupNorm(num_groups, out_channels)
        elif norm == "bn":
            layers["bn"] = nn.BatchNorm2d(out_channels)
        if activation:
            layers["activate"] = nn.ReLU(inplace=True)
        super().__init__(layers)


class Mlp(nn.Module):
    """FFN блока MSCA: 1x1 -> depthwise 3x3 -> GELU -> 1x1."""

    def __init__(self, channels: int, hidden_channels: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(channels, hidden_channels, 1)
        self.dwconv = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, groups=hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, channels, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.dwconv(self.fc1(x))))
        return self.drop(self.fc2(x))


class MSCAAttention(nn.Module):
    """Многомасштабное свёрточное внимание.

    Ключевая идея статьи: большое рецептивное поле нужно, но свёртка 21x21
    неподъёмна, поэтому каждая «крупная» ветка раскладывается на пару
    полосовых depthwise-свёрток (1xK и Kx1). Три ветки с K = 7, 11, 21 плюс
    базовая 5x5 дают карту внимания, на которую поэлементно умножается вход.
    """

    KERNELS = (7, 11, 21)

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        for index, kernel in enumerate(self.KERNELS):
            padding = kernel // 2
            self.add_module(
                f"conv{index}_1",
                nn.Conv2d(channels, channels, (1, kernel), padding=(0, padding), groups=channels),
            )
            self.add_module(
                f"conv{index}_2",
                nn.Conv2d(channels, channels, (kernel, 1), padding=(padding, 0), groups=channels),
            )
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv0(x)
        branches = attention
        for index in range(len(self.KERNELS)):
            branch = getattr(self, f"conv{index}_1")(attention)
            branches = branches + getattr(self, f"conv{index}_2")(branch)

        return self.conv3(branches) * x


class MSCASpatialAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj_1 = nn.Conv2d(channels, channels, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = MSCAAttention(channels)
        self.proj_2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.proj_2(self.spatial_gating_unit(self.activation(self.proj_1(x))))
        return attended + x


class DropPath(nn.Module):
    """Stochastic depth по примерам батча (Huang et al., arXiv:1603.09382)."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability <= 0.0 or not self.training:
            return x

        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class MSCABlock(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.attn = MSCASpatialAttention(channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(channels)
        self.mlp = Mlp(channels, int(channels * mlp_ratio), drop=drop)
        # layer scale: обучаемый множитель ветки, инициализированный малым
        # числом — без него глубокие стадии MSCAN расходятся на старте.
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones(channels))
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale_1[:, None, None] * self.attn(self.norm1(x)))
        return x + self.drop_path(self.layer_scale_2[:, None, None] * self.mlp(self.norm2(x)))


class StemConv(nn.Module):
    """Вход стадии 1: две свёртки 3x3 со страйдом 2 (суммарно /4)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class OverlapPatchEmbed(nn.Module):
    """Вход стадий 2..4: свёртка 3x3 со страйдом 2 с перекрытием."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class MSCAN(nn.Module):
    """Иерархический свёрточный энкодер SegNeXt: 4 стадии, страйды 4/8/16/32.

    Внутри mmsegmentation блоки работают с последовательностью [B, N, C] и
    LayerNorm в конце стадии; здесь тензор всё время остаётся [B, C, H, W],
    а нормировка стадии применяется к каналам — это та же операция, только
    без пары permute на каждый блок. Веса от этого не меняются: у LayerNorm
    по каналам ровно те же weight и bias.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: Sequence[int] = (64, 128, 320, 512),
        depths: Sequence[int] = (3, 3, 12, 3),
        mlp_ratios: Sequence[float] = (8, 8, 4, 4),
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.depths = list(depths)
        self.embed_dims = list(embed_dims)

        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        start = 0
        for stage, depth in enumerate(self.depths):
            if stage == 0:
                patch_embed = StemConv(in_channels, embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(embed_dims[stage - 1], embed_dims[stage])

            blocks = nn.ModuleList(
                MSCABlock(
                    embed_dims[stage],
                    mlp_ratio=mlp_ratios[stage],
                    drop=drop_rate,
                    drop_path=rates[start + index],
                )
                for index in range(depth)
            )
            start += depth

            setattr(self, f"patch_embed{stage + 1}", patch_embed)
            setattr(self, f"block{stage + 1}", blocks)
            setattr(self, f"norm{stage + 1}", nn.LayerNorm(embed_dims[stage]))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Инициализация из статьи: свёртки — He по fan_out с учётом групп."""
        if isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for stage in range(len(self.depths)):
            x = getattr(self, f"patch_embed{stage + 1}")(x)
            for block in getattr(self, f"block{stage + 1}"):
                x = block(x)
            # LayerNorm по каналам: [B, C, H, W] -> [B, H, W, C] -> обратно.
            x = getattr(self, f"norm{stage + 1}")(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = x.contiguous()
            features.append(x)

        return features


class NMF2D(nn.Module):
    """Неотрицательное матричное разложение как «внимание» (arXiv:2109.04553).

    Карта признаков [C, HW] раскладывается на R базисов и их коэффициенты
    мультипликативными обновлениями. Смысл: глобальный контекст описывается
    небольшим числом повторяющихся паттернов, и их поиск разложением дешевле
    и устойчивее, чем self-attention. Параметров у модуля нет — только
    итерации, поэтому и в state_dict он ничего не добавляет.

    ВНИМАНИЕ: базисы инициализируются случайно на каждом forward'е (rand_init
    в оригинале), поэтому SegNeXt не детерминирован даже в eval — два прогона
    по одному кадру дают чуть разные логиты. Итерации сходятся, так что
    разброс мал, но полностью воспроизводимого предсказания от этой модели
    ждать не стоит.
    """

    def __init__(
        self,
        bases: int = 16,
        train_steps: int = 6,
        eval_steps: int = 7,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.bases = int(bases)
        self.train_steps = int(train_steps)
        self.eval_steps = int(eval_steps)
        self.eps = float(eps)

    def _step(
        self, x: torch.Tensor, bases: torch.Tensor, coefficients: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = coefficients * torch.bmm(x.transpose(1, 2), bases) / (
            coefficients.bmm(bases.transpose(1, 2).bmm(bases)) + self.eps
        )
        bases = bases * torch.bmm(x, coefficients) / (
            bases.bmm(coefficients.transpose(1, 2).bmm(coefficients)) + self.eps
        )
        return bases, coefficients

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape

        # enabled=False обязателен: под autocast каждый bmm сам по себе ушёл бы
        # в fp16, а мультипликативные обновления делят одно почти нулевое
        # число на другое и в половинной точности расходятся в NaN.
        with torch.autocast(x.device.type, enabled=False):
            # Разложение требует неотрицательности — перед ним стоит ReLU.
            flat = x.float().reshape(batch, channels, height * width)

            bases = F.normalize(torch.rand(batch, channels, self.bases, device=x.device), dim=1)
            coefficients = F.softmax(torch.bmm(flat.transpose(1, 2), bases), dim=-1)

            for _ in range(self.train_steps if self.training else self.eval_steps):
                bases, coefficients = self._step(flat, bases, coefficients)

            # Ещё одно обновление коэффициентов при фиксированных базисах —
            # как в оригинале: после последнего шага базисы уже другие.
            coefficients = coefficients * torch.bmm(flat.transpose(1, 2), bases) / (
                coefficients.bmm(bases.transpose(1, 2).bmm(bases)) + self.eps
            )

            reconstructed = torch.bmm(bases, coefficients.transpose(1, 2))

        return reconstructed.reshape(batch, channels, height, width).to(x.dtype)


class Hamburger(nn.Module):
    """«Гамбургер»: две линейные проекции («хлеб») вокруг разложения («котлета»)."""

    def __init__(self, channels: int, bases: int = 16, num_groups: int = 32) -> None:
        super().__init__()
        self.ham_in = ConvNorm(channels, channels, norm=None, activation=False)
        self.ham = NMF2D(bases=bases)
        self.ham_out = ConvNorm(channels, channels, num_groups=num_groups, activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enjoy = F.relu(self.ham_in(x))
        enjoy = self.ham_out(self.ham(enjoy))
        return F.relu(x + enjoy)


class LightHamHead(nn.Module):
    """Декодер SegNeXt: склейка стадий 2..4, гамбургер, классификатор.

    Стадия 1 (страйд 4) в декодер не входит — так в статье: мелкие детали
    восстанавливать дешевле апсемплом, а от первой стадии на полном
    разрешении декодер только тяжелел бы.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        channels: int,
        num_classes: int,
        bases: int = 16,
        num_groups: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.squeeze = ConvNorm(sum(in_channels), channels, num_groups=num_groups)
        self.hamburger = Hamburger(channels, bases=bases, num_groups=num_groups)
        self.align = ConvNorm(channels, channels, num_groups=num_groups)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        size = features[0].shape[2:]
        resized = [
            feature
            if feature.shape[2:] == size
            else F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
            for feature in features
        ]

        x = self.squeeze(torch.cat(resized, dim=1))
        x = self.align(self.hamburger(x))
        return self.conv_seg(self.dropout(x))


class SegNeXt(nn.Module):
    """SegNeXt-T/S/B/L с логитами в разрешении входа.

    Контракт тот же, что у SegFormer и TimmUNet: forward(images) -> [B, C, H, W]
    и канонические тапы taps.stage1..stage4, поэтому модель одинаково годится
    и учеником, и учителем в любом лоссе проекта.
    """

    def __init__(
        self,
        variant: str = "t",
        num_classes: int = 19,
        in_channels: int = 3,
        drop_path_rate: float | None = None,
        dropout: float = 0.1,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if variant not in SEGNEXT_VARIANTS:
            raise ValueError(
                f"Неизвестный вариант SegNeXt: {variant!r}. Доступны: {sorted(SEGNEXT_VARIANTS)}"
            )

        spec = SEGNEXT_VARIANTS[variant]
        self.variant = variant
        self.align_corners = align_corners
        self.embed_dims = list(spec["embed_dims"])

        self.encoder = MSCAN(
            in_channels=in_channels,
            embed_dims=spec["embed_dims"],
            depths=spec["depths"],
            drop_path_rate=spec["drop_path_rate"] if drop_path_rate is None else drop_path_rate,
        )
        self.decoder = LightHamHead(
            in_channels=spec["embed_dims"][1:],
            channels=spec["decoder_channels"],
            num_classes=num_classes,
            dropout=dropout,
        )

        self.tap_channels = dict(zip(STAGE_TAPS, self.embed_dims))
        self.taps = FeatureTaps(STAGE_TAPS)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        for name, feature in zip(STAGE_TAPS, features):
            self.taps.tap(name, feature)

        logits = self.decoder(features[1:])
        return F.interpolate(
            logits,
            size=images.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )


def convert_mmseg_state_dict(state_dict: dict) -> dict:
    """Веса mmsegmentation/SegNeXt -> имена этой модели.

    Отличий всего три, и все они механические:
    - префиксы backbone. / decode_head. -> encoder. / decoder.;
    - в оригинальном репозитории SegNeXt (в отличие от mmsegmentation)
      depthwise-свёртка внутри MLP завёрнута в отдельный модуль, поэтому
      ключ содержит dwconv.dwconv;
    - веса ImageNet-энкодеров лежат без префикса вовсе.

    Всё, что не относится ни к энкодеру, ни к декодеру (например,
    auxiliary_head), отбрасывается.
    """
    converted: dict = {}

    for key, value in state_dict.items():
        name = key
        if name.startswith("backbone."):
            name = "encoder." + name.removeprefix("backbone.")
        elif name.startswith("decode_head."):
            name = "decoder." + name.removeprefix("decode_head.")
        elif name.startswith(("patch_embed", "block", "norm")):
            name = "encoder." + name
        else:
            continue

        name = name.replace("mlp.dwconv.dwconv.", "mlp.dwconv.")
        converted[name] = value

    return converted
