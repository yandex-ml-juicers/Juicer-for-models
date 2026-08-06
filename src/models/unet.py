"""U-Net (Ronneberger et al., MICCAI 2015, arXiv:1505.04597) — ученик сегментации.

Классический симметричный энкодер-декодер со skip-connections. Здесь он живёт
рядом с segformer.py по той же причине: файл описывает nn.Module, а фабричная
функция для _target_ в конфигах лежит в factory.py.

Отличие от статьи: свёртки с padding=1 сохраняют размер, поэтому выход совпадает
по разрешению со входом и логиты [B, num_classes, H, W] идут в cross_entropy
напрямую, без интерполяции и без обрезки меток.
"""

import torch
from torch import nn

from src.models.feature_taps import STAGE_TAP_STRIDES, FeatureTaps


class UNetDoubleConv(nn.Sequential):
    """Conv-BN-ReLU x2 — базовый блок U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Классический U-Net: симметричный энкодер-декодер со skip-connections.

    Требование к входу: H и W кратны 2**depth (512x1024 и 1024x2048 при
    depth=4 подходят).

    Тапы. Энкодер выдаёт карты на страйдах 1, 2, ..., 2**(depth-1), боттлнек —
    на 2**depth; канонические имена вешаются на те из них, чей страйд попадает
    в STAGE_TAP_STRIDES. При depth=4 это stage1/stage2/stage3 (страйды 4/8/16),
    stage4 (страйд 32) физически отсутствует и НЕ объявляется — пусть FitNets
    падает на сборке конфига, а не молча сравнивает не то с тем. depth=5 даёт
    все четыре, но требует входа, кратного 32.

    Ширины стадий выведены в tap_channels: их надо прописать в student_channels
    конфига FitNets, и брать их лучше отсюда, чем считать в уме.
    """

    def __init__(
        self,
        num_classes: int = 19,
        in_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth должен быть >= 2, получено {depth}")

        self.depth = int(depth)
        self.base_channels = int(base_channels)

        channels = [base_channels * 2 ** level for level in range(depth + 1)]

        self.encoders = nn.ModuleList()
        previous_channels = in_channels
        for level_channels in channels[:-1]:
            self.encoders.append(UNetDoubleConv(previous_channels, level_channels))
            previous_channels = level_channels

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = UNetDoubleConv(channels[-2], channels[-1])

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level_channels in reversed(channels[:-1]):
            self.upsamples.append(
                nn.ConvTranspose2d(level_channels * 2, level_channels, kernel_size=2, stride=2)
            )
            # На вход декодера идёт конкатенация апсемпла и skip-связи.
            self.decoders.append(UNetDoubleConv(level_channels * 2, level_channels))

        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        # Страйд -> имя тапа, только для реально существующих стадий.
        # На страйде 2**level энкодер отдаёт base_channels * 2**level каналов,
        # то есть ширина стадии равна base_channels * страйд — и для выходов
        # энкодера, и для боттлнека.
        self._tap_by_stride = {
            stride: name
            for name, stride in STAGE_TAP_STRIDES.items()
            if stride <= 2 ** self.depth
        }
        self.tap_channels: dict[str, int] = {
            name: self.base_channels * stride for stride, name in self._tap_by_stride.items()
        }
        self.taps = FeatureTaps(self.tap_channels.keys())

    def _tap(self, stride: int, tensor: torch.Tensor) -> None:
        """Прогоняет карту через тап, если для этого страйда он объявлен.

        Возврат не нужен: Identity отдаёт тот же самый тензор, а тензор и так
        остаётся в графе (это skip-связь или вход декодера). Нужен только
        побочный эффект — срабатывание forward-хука FeatureExtractor.
        """
        name = self._tap_by_stride.get(stride)
        if name is not None:
            self.taps.tap(name, tensor)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        skips = []

        features = images
        for level, encoder in enumerate(self.encoders):
            features = encoder(features)
            skips.append(features)
            self._tap(2 ** level, features)
            features = self.pool(features)

        features = self.bottleneck(features)
        self._tap(2 ** self.depth, features)

        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            features = upsample(features)
            features = torch.cat([skip, features], dim=1)
            features = decoder(features)

        return self.head(features)


# Именованные размеры U-Net. Число параметров считается как sum(p.numel())
# для num_classes=19 и проверяется тестом — если правите base_channels,
# правьте и комментарий.
#   tiny  ~1.9M   small ~4.4M   base ~7.8M   large ~17.4M   full ~31.0M
UNET_VARIANTS: dict[str, dict] = {
    "tiny": {"base_channels": 16, "depth": 4},
    "small": {"base_channels": 24, "depth": 4},
    "base": {"base_channels": 32, "depth": 4},
    "large": {"base_channels": 48, "depth": 4},
    "full": {"base_channels": 64, "depth": 4},
}
