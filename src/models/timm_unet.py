"""U-Net с ПРЕДОБУЧЕННЫМ энкодером из timm.

Зачем отдельная модель, а не флаг в src/models/unet.py. В классический U-Net
чужие веса загрузить нельзя, и это не вопрос аккуратного переименования ключей:

- страйды стадий. У U-Net (depth=4) энкодер работает на 1/2/4/8 плюс боттлнек
  на 16, у ResNet и вообще любого ImageNet-бэкбона — 2/4/8/16/32. Первая стадия
  U-Net живёт в ПОЛНОМ разрешении, у бэкбонов такой стадии нет вовсе;
- residual. UNetDoubleConv — это два conv-BN-ReLU подряд, а BasicBlock ResNet
  прибавляет к результату вход. Веса residual-блока обучены выдавать ПОПРАВКУ
  к своему входу; в блоке без сложения они означают не то же самое, даже когда
  формы совпадают;
- глубина стадии. У U-Net ровно 2 свёртки на уровень, у resnet18 layer1 — это
  два BasicBlock, то есть 4 свёртки;
- понижение разрешения. U-Net понижает MaxPool'ом между блоками, ResNet —
  свёрткой со страйдом внутри блока.

Поэтому здесь заимствуется не файл весов, а сам энкодер: timm отдаёт готовый
предобученный бэкбон в режиме features_only, а декодер остаётся юнетовским —
апсемпл, конкатенация со skip-связью, UNetDoubleConv. Классический U-Net при
этом не тронут: он остаётся бейзлайном "с нуля", и его чекпоинты живы.

Побочный выигрыш: у ImageNet-бэкбонов есть стадия на страйде 32, поэтому
здесь доступны ВСЕ четыре канонических тапа, включая stage4 — в отличие от
U-Net с depth=4.

Нормализация входа обязана быть ImageNet-овской (mean=[0.485,0.456,0.406],
std=[0.229,0.224,0.225]) — она уже такая в configs/data/dataset/cityscapes_seg.yaml.
"""

import timm
import torch
import torch.nn.functional as F
from torch import nn

from src.models.feature_taps import STAGE_TAP_STRIDES, FeatureTaps
from src.models.unet import UNetDoubleConv


class TimmUNet(nn.Module):
    """Энкодер из timm + юнетовский декодер, логиты в разрешении входа.

    Args:
        encoder_name: имя модели timm. Тег весов пишется через точку —
            "convnext_nano.in12k" возьмёт ImageNet-12k вместо дефолтного
            ImageNet-1k. Энкодер обязан поддерживать features_only=True.
        pretrained: грузить веса энкодера. False — та же архитектура со
            случайной инициализацией; это честный контроль для абляции
            "что дало именно предобучение".
        dropout: Dropout2d на боттлнеке, как в U-Net.

    Апсемпл в декодере — билинейный до размера skip-связи, а не
    ConvTranspose2d: параметров меньше, шахматных артефактов нет, и он
    переживает вход, не кратный 32.
    """

    def __init__(
        self,
        encoder_name: str = "mobilenetv2_100",
        num_classes: int = 19,
        in_channels: int = 3,
        pretrained: bool = True,
        dropout: float = 0.0,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout должен быть в [0, 1), получено {dropout}")

        self.encoder_name = encoder_name
        self.align_corners = align_corners

        try:
            self.encoder = timm.create_model(
                encoder_name,
                pretrained=pretrained,
                features_only=True,
                in_chans=in_channels,
            )
        except RuntimeError as error:
            raise ValueError(
                f"Энкодер {encoder_name!r} не собирается в режиме features_only "
                f"(нужен иерархический бэкбон с картами признаков). Исходная ошибка: {error}"
            ) from error

        channels = list(self.encoder.feature_info.channels())
        self.reductions = list(self.encoder.feature_info.reduction())
        if len(channels) < 2:
            raise ValueError(
                f"Энкодер {encoder_name!r} отдаёт {len(channels)} карт признаков, "
                f"декодеру нужно минимум две"
            )

        # Энкодер обязан быть ИЕРАРХИЧЕСКИМ. Плоские бэкбоны (ViT и подобные)
        # тоже собираются с features_only, но отдают все карты на одном
        # страйде — например vit_tiny даёт reductions=[16, 16, 16]. Декодер
        # на таком собрался бы и даже посчитался бы (интерполяция до размера
        # skip-связи стала бы no-op), но U-Net из этого не выйдет: нет ни
        # апсемпл-пути, ни разных масштабов у skip-связей. Лучше упасть здесь.
        if any(a >= b for a, b in zip(self.reductions, self.reductions[1:])):
            raise ValueError(
                f"Энкодер {encoder_name!r} не иерархический: страйды стадий "
                f"{self.reductions} не возрастают строго. U-Net-декодеру нужен "
                f"бэкбон, понижающий разрешение от стадии к стадии (ResNet, "
                f"ConvNeXt, EfficientNet, MobileNet, Swin)."
            )

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        # Декодер идёт от самой глубокой карты к самой мелкой. Ширина уровня =
        # ширина skip-связи, с которой он склеивается: так размер декодера
        # масштабируется вместе с энкодером и не начинает доминировать
        # на лёгких бэкбонах.
        self.decoders = nn.ModuleList()
        decoder_channels = channels[-1]
        for skip_channels in reversed(channels[:-1]):
            self.decoders.append(UNetDoubleConv(decoder_channels + skip_channels, skip_channels))
            decoder_channels = skip_channels

        self.head = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

        # Тапы — по страйду, как у U-Net. У ImageNet-бэкбонов есть все четыре.
        self._tap_by_stride = {
            stride: name
            for name, stride in STAGE_TAP_STRIDES.items()
            if stride in self.reductions
        }
        self.tap_channels: dict[str, int] = {
            name: channels[self.reductions.index(stride)]
            for stride, name in self._tap_by_stride.items()
        }
        self.taps = FeatureTaps(self.tap_channels.keys())

    def _tap(self, stride: int, tensor: torch.Tensor) -> None:
        name = self._tap_by_stride.get(stride)
        if name is not None:
            self.taps.tap(name, tensor)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        for feature, reduction in zip(features, self.reductions):
            self._tap(reduction, feature)

        x = self.dropout(features[-1])
        for decoder, skip in zip(self.decoders, reversed(features[:-1])):
            # Интерполяция ДО размера skip-связи, а не scale_factor=2: страйды
            # соседних стадий не обязаны отличаться ровно вдвое, а нечётные
            # размеры кропа иначе разъезжаются на пиксель.
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=self.align_corners
            )
            x = decoder(torch.cat([skip, x], dim=1))

        # Самая мелкая карта энкодера лежит на страйде 2 (ResNet, MobileNet)
        # или 4 (ConvNeXt), поэтому финальный апсемпл до входа делаем здесь —
        # как и SegFormer, чей выход тоже приходит в 1/4.
        return F.interpolate(
            self.head(x),
            size=images.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )


# Именованные размеры. Подобраны так, чтобы перекрыть диапазон UNET_VARIANTS
# (1.9M..31M) и дать прямые пары для сравнения "с нуля против предобучения".
# Числа замерены и проверяются тестом test_variant_sizes_match_the_table.
#
#   вариант                     всего   энкодер  ~парный U-Net
#   mobilenetv3_small             1.25M    0.93M   наносайз (0.5-2M), см. ниже
#   edgenext_xxs                  1.53M    1.16M   наносайз (0.5-2M), см. ниже
#   mobilenetv2                  2.33M    1.81M   tiny  (1.94M)
#   efficientnet_b0              4.24M    3.60M   small (4.37M)
#   convnext_femto               6.58M    4.83M
#   efficientnetv2_b2            9.04M    8.39M   base  (7.76M)
#   mobilenetv4_medium_in12k     9.39M    7.20M   base  (7.76M)
#   convnext_pico               11.63M    8.53M
#   resnet18                    14.39M   11.18M
#   convnext_nano_in12k         19.79M   14.95M   large (17.46M)
#   resnet34                    24.50M   21.28M
#
# mobilenetv3_small/edgenext_xxs — под задачу "модели 0.5-2M параметров":
# оба энкодера иерархические, с весами ImageNet-1k в timm
# (mobilenetv3_small_100.lamb_in1k, edgenext_xx_small.in1k) и вместе с
# декодером укладываются в 0.5-2M — специально подбирались по этому бюджету
# (сравнивались реальные params-counts нескольких кандидатов: ghostnet_100,
# mobilevit_xxs, lcnet_*, repghostnet_050 — у части из них декодер на широких
# skip-каналах сам по себе не вписывается в бюджет, эти два — вписываются
# с запасом и оба обучены на ImageNet).
#
# Про ImageNet-21k. Среди чистых CNN такого размера весов на полном 21k
# практически нет: публично доступны либо in1k, либо ImageNet-12k (подмножество
# 21k на 11821 класс) — и то лишь у двух моделей из списка. Тег весов timm
# читает прямо из имени, после точки.
TIMM_UNET_VARIANTS: dict[str, dict] = {
    # ImageNet-1k
    "mobilenetv3_small": {"encoder_name": "mobilenetv3_small_100"},
    "edgenext_xxs": {"encoder_name": "edgenext_xx_small"},
    "mobilenetv2": {"encoder_name": "mobilenetv2_100"},
    "efficientnet_b0": {"encoder_name": "efficientnet_b0"},
    "convnext_femto": {"encoder_name": "convnext_femto"},
    "efficientnetv2_b2": {"encoder_name": "tf_efficientnetv2_b2"},
    "convnext_pico": {"encoder_name": "convnext_pico"},
    "resnet18": {"encoder_name": "resnet18"},
    "resnet34": {"encoder_name": "resnet34"},
    # ImageNet-12k (подмножество ImageNet-21k)
    "convnext_nano_in12k": {"encoder_name": "convnext_nano.in12k"},
    "mobilenetv4_medium_in12k": {"encoder_name": "mobilenetv4_conv_medium.e250_r384_in12k"},
}
