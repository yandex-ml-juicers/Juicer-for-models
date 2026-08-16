"""ESPNetv2 (Mehta et al., CVPR 2019, arXiv:1811.11431): энкодер EESP + наш
UNet-декодер.

Зачем он в проекте. Модели вроде timm_unet/mobilenetv3_small берут
ImageNet-классификатор и приделывают к нему универсальный U-Net-декодер —
сам энкодер при этом ни разу не видел сегментацию как задачу. EESP
(Extremely Efficient Spatial Pyramid) — блок REDUCE->SPLIT->TRANSFORM->MERGE
с иерархическими dilated depthwise-свёртками разного radius — спроектирован
авторами именно под плотное предсказание (сегментация/детекция), а не
классификацию: большой рецептивный слой набирается почти бесплатно по
параметрам за счёт group/depthwise свёрток. У энкодера ЕСТЬ отдельные
ImageNet-веса (не только сборка целиком на Cityscapes, как у большинства
lightweight-сеток его класса) — то, что и нужно для честного сравнения
"pretrained-энкодер + наш KD-рецепт", без короткого пути через чужой
Cityscapes-чекпоинт.

Почему архитектура написана здесь, а не взята из библиотеки. Оригинал живёт
в отдельном репозитории (sacmehta/ESPNetv2) без пакетной установки, и его
ImageNet-классификатор (EESPNet) не выдаёт карты признаков — только logits
после global pooling. Здесь воспроизведены ровно классификационные слои
(level1..level5), без classifier и без двух последних слоёв level5,
расширяющих ширину под classifier (1x1/3x3 groupwise conv) — сегментационному
декодеру они не нужны, а без них веса заведомо не подходят под чекпоинт.

ИМЕНА ПОДМОДУЛЕЙ намеренно повторяют оригинал (level1, level2_0, level3_0,
level3, level4_0, level4, level5_0, level5, а внутри — proj_1x1, spp_dw,
conv_1x1_exp, br_after_cat, module_act, eesp, avg, inp_reinf): это и есть
способ загружать официальные ImageNet-веса — см. convert_espnetv2_state_dict.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.models.feature_taps import STAGE_TAPS, FeatureTaps
from src.models.unet import UNetDoubleConv

# ============================================================================
# Примитивы и EESP/DownSampler/EESPNet — порт sacmehta/ESPNetv2 (MIT license).
# Логика и имена полей не изменены (иначе официальный чекпоинт не загрузится
# без ручного переименования); удалены только classifier и global classes=.
# ============================================================================


class CBR(nn.Module):
    """conv(без bias) -> BatchNorm -> PReLU."""

    def __init__(self, n_in: int, n_out: int, k_size: int, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        padding = (k_size - 1) // 2
        self.conv = nn.Conv2d(n_in, n_out, k_size, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(n_out)
        self.act = nn.PReLU(n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class BR(nn.Module):
    """BatchNorm -> PReLU (без свёртки)."""

    def __init__(self, n_out: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(n_out)
        self.act = nn.PReLU(n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(x))


class CB(nn.Module):
    """conv(без bias) -> BatchNorm, без активации."""

    def __init__(self, n_in: int, n_out: int, k_size: int, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        padding = (k_size - 1) // 2
        self.conv = nn.Conv2d(n_in, n_out, k_size, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class CDilated(nn.Module):
    """Dilated-свёртка без bias и без нормализации (нормируется снаружи)."""

    def __init__(
        self, n_in: int, n_out: int, k_size: int, stride: int = 1, d: int = 1, groups: int = 1
    ) -> None:
        super().__init__()
        padding = ((k_size - 1) // 2) * d
        self.conv = nn.Conv2d(
            n_in, n_out, k_size, stride=stride, padding=padding, bias=False, dilation=d, groups=groups
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class C(nn.Module):
    """Голая свёртка: без bias, без нормализации, без активации."""

    def __init__(self, n_in: int, n_out: int, k_size: int, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        padding = (k_size - 1) // 2
        self.conv = nn.Conv2d(n_in, n_out, k_size, stride=stride, padding=padding, bias=False, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EESP(nn.Module):
    """REDUCE (group 1x1) -> SPLIT в k параллельных dilated depthwise-веток
    разного radius -> иерархический fusion соседних веток (устраняет
    гридинг-артефакт dilated-свёрток) -> MERGE (group 1x1) -> residual."""

    def __init__(self, n_in: int, n_out: int, stride: int = 1, k: int = 4, r_lim: int = 7, down_method: str = "esp") -> None:
        super().__init__()
        self.stride = stride
        n = n_out // k
        n1 = n_out - (k - 1) * n
        if down_method not in ("avg", "esp"):
            raise ValueError(f"down_method должен быть 'avg' или 'esp', получено {down_method!r}")
        if n != n1:
            raise ValueError(f"n_out={n_out} должен делиться на k={k} без остатка (Depth-wise conv)")

        self.proj_1x1 = CBR(n_in, n, 1, stride=1, groups=k)

        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        k_sizes = []
        for i in range(k):
            ksize = 3 + 2 * i
            ksize = ksize if ksize <= r_lim else 3
            k_sizes.append(ksize)
        k_sizes.sort()

        self.spp_dw = nn.ModuleList()
        for ksize in k_sizes:
            d_rate = map_receptive_ksize[ksize]
            self.spp_dw.append(CDilated(n, n, k_size=3, stride=stride, groups=n, d=d_rate))

        self.conv_1x1_exp = CB(n_out, n_out, 1, 1, groups=k)
        self.br_after_cat = BR(n_out)
        self.module_act = nn.PReLU(n_out)
        self.downAvg = down_method == "avg"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            out_k = out_k + output[k - 1]  # Hierarchical Feature Fusion
            output.append(out_k)

        expanded = self.conv_1x1_exp(self.br_after_cat(torch.cat(output, 1)))

        if self.stride == 2 and self.downAvg:
            return expanded
        if expanded.shape == x.shape:
            expanded = expanded + x
        return self.module_act(expanded)


class DownSampler(nn.Module):
    """avg-pool || EESP(stride=2) -> concat -> (+ пере-впрыск исходного
    изображения, average-pooled до текущего разрешения) -> PReLU."""

    def __init__(
        self, n_in: int, n_out: int, k: int = 4, r_lim: int = 9, reinf: bool = True, reinf_channels: int = 3
    ) -> None:
        super().__init__()
        n_out_new = n_out - n_in
        self.eesp = EESP(n_in, n_out_new, stride=2, k=k, r_lim=r_lim, down_method="avg")
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        if reinf:
            self.inp_reinf = nn.Sequential(
                CBR(reinf_channels, reinf_channels, 3, 1),
                CB(reinf_channels, n_out, 1, 1),
            )
        self.act = nn.PReLU(n_out)

    def forward(self, x: torch.Tensor, raw_input: torch.Tensor | None = None) -> torch.Tensor:
        avg_out = self.avg(x)
        eesp_out = self.eesp(x)
        output = torch.cat([avg_out, eesp_out], 1)

        if raw_input is not None:
            target_size = avg_out.shape[2]
            while raw_input.shape[2] != target_size:
                raw_input = F.avg_pool2d(raw_input, kernel_size=3, padding=1, stride=2)
            output = output + self.inp_reinf(raw_input)

        return self.act(output)


class EESPNet(nn.Module):
    """Энкодер ESPNetv2 (уровни 1-5) без ImageNet-классификатора.

    forward возвращает 4 карты признаков на стандартных стайдах 4/8/16/32
    (level1, страйд 2, — это стем, канонического тапа под него нет нигде
    в проекте, см. src/models/feature_taps.py). self.channels[1:] — ширины
    этих четырёх карт, ИМЕННО в этом порядке.
    """

    def __init__(self, s: float = 1.0, in_channels: int = 3) -> None:
        super().__init__()
        reps = [0, 3, 7, 3]  # повторов EESP-блока на уровнях 2..5 (уровень 2 — только DownSampler)
        r_lim = [13, 11, 9, 7, 5]
        k = [4] * len(r_lim)

        base = 32
        base_s = math.ceil(int(base * s) / k[0]) * k[0]
        config = [base if base_s > base else base_s]
        for i in range(1, len(r_lim)):
            config.append(base_s * (2**i))
        self.channels = config  # [stage_stem, stage1, stage2, stage3, stage4]

        self.level1 = CBR(in_channels, config[0], 3, 2)

        self.level2_0 = DownSampler(config[0], config[1], k=k[0], r_lim=r_lim[0], reinf_channels=in_channels)

        self.level3_0 = DownSampler(config[1], config[2], k=k[1], r_lim=r_lim[1], reinf_channels=in_channels)
        self.level3 = nn.ModuleList(EESP(config[2], config[2], k=k[2], r_lim=r_lim[2]) for _ in range(reps[1]))

        self.level4_0 = DownSampler(config[2], config[3], k=k[2], r_lim=r_lim[2], reinf_channels=in_channels)
        self.level4 = nn.ModuleList(EESP(config[3], config[3], k=k[3], r_lim=r_lim[3]) for _ in range(reps[2]))

        # reinf=True здесь и в оригинале — level5_0.inp_reinf СУЩЕСТВУЕТ (и
        # входит в официальный чекпоинт), но в forward ниже не вызывается
        # (raw_input не передаётся): так авторы никогда не увеличивают глубину
        # входного пути дальше уровня 4. Убирать эти веса нельзя — тогда
        # официальный чекпоинт перестанет грузиться (missing keys).
        self.level5_0 = DownSampler(config[3], config[4], k=k[3], r_lim=r_lim[3], reinf_channels=in_channels)
        self.level5 = nn.ModuleList(EESP(config[4], config[4], k=k[4], r_lim=r_lim[4]) for _ in range(reps[3]))

    def forward_until_level4(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Уровни 1-4 (страйды 2/4/8/16), без level5 — общая часть для обоих
        декодеров проекта (ESPNetV2 ниже досчитывает level5 сам, у
        ESPNetV2Native он не нужен вовсе, ровно как в оригинале, см.
        EESPNet.forward(..., seg=True) в sacmehta/ESPNetv2)."""
        out_l1 = self.level1(images)  # 1/2, стем

        out_l2 = self.level2_0(out_l1, images)  # 1/4 -> stage1

        out_l3 = self.level3_0(out_l2, images)  # 1/8
        for layer in self.level3:
            out_l3 = layer(out_l3)  # -> stage2

        out_l4 = self.level4_0(out_l3, images)  # 1/16
        for layer in self.level4:
            out_l4 = layer(out_l4)  # -> stage3

        return out_l1, out_l2, out_l3, out_l4

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        _, out_l2, out_l3, out_l4 = self.forward_until_level4(images)

        out_l5 = self.level5_0(out_l4)  # 1/32, без пере-впрыска входа (как в оригинале)
        for layer in self.level5:
            out_l5 = layer(out_l5)  # -> stage4

        return [out_l2, out_l3, out_l4, out_l5]


def convert_espnetv2_state_dict(state_dict: dict) -> dict:
    """Официальный ImageNet-чекпоинт (sacmehta/ESPNetv2) -> имена этой модели.

    Префикс module. (DataParallel) уже снят до вызова этой функции, см.
    load_converted_checkpoint. Здесь остаётся:
    - добавить префикс encoder. (веса лежат без него — это классификатор
      целиком, а не под-объект нашей модели);
    - выбросить classifier.* (в этой модели головы классификации нет) и
      хвостовые level5.3/level5.4 (два groupwise/depthwise conv, которыми
      оригинал расширяет ширину под classifier — тоже не воспроизведены).
    """
    converted = {}
    for key, value in state_dict.items():
        if key.startswith("classifier."):
            continue
        if key.startswith(("level5.3.", "level5.4.")):
            continue
        converted[f"encoder.{key}"] = value
    return converted


# Официальные ImageNet-веса — по одному файлу на масштаб s (Table в README
# sacmehta/ESPNetv2). Имя варианта = "s" * 100, без точки (footgun: голое
# "1.0" в YAML читалось бы OmegaConf как float, а не строка-ключ словаря).
ESPNETV2_VARIANTS: dict[str, dict] = {
    "s050": {"s": 0.5},
    "s100": {"s": 1.0},
    "s125": {"s": 1.25},
    "s150": {"s": 1.5},
    "s200": {"s": 2.0},
}

IMAGENET_WEIGHTS: dict[str, str] = {
    "s050": "https://raw.githubusercontent.com/sacmehta/ESPNetv2/master/imagenet/pretrained_weights/espnetv2_s_0.5.pth",
    "s100": "https://raw.githubusercontent.com/sacmehta/ESPNetv2/master/imagenet/pretrained_weights/espnetv2_s_1.0.pth",
    "s125": "https://raw.githubusercontent.com/sacmehta/ESPNetv2/master/imagenet/pretrained_weights/espnetv2_s_1.25.pth",
    "s150": "https://raw.githubusercontent.com/sacmehta/ESPNetv2/master/imagenet/pretrained_weights/espnetv2_s_1.5.pth",
    "s200": "https://raw.githubusercontent.com/sacmehta/ESPNetv2/master/imagenet/pretrained_weights/espnetv2_s_2.0.pth",
}


class ESPNetV2(nn.Module):
    """EESPNet-энкодер + наш UNet-декодер (UNetDoubleConv), логиты в
    разрешении входа. Контракт тот же, что у TimmUNet/SegNeXt: forward ->
    [B, C, H, W], тапы taps.stage1..stage4.

    Декодер здесь НЕ из оригинальной статьи — тот же UNetDoubleConv, что и у
    остальных энкодеров TimmUNet: декодер держится постоянным, чтобы вклад
    именно энкодера был сравним между архитектурами (тот же принцип, что и
    в timm_unet.py). У этого декодера плотные 3x3-свёртки на широких
    skip-каналах, поэтому он ДОМИНИРУЕТ бюджет параметров (для s050 —
    0.78M из 0.92M, энкодер — только 0.14M) и заведомо тяжелее декодера из
    статьи. Если важна прежде всего сама архитектура ESPNetv2 (включая её
    декодер PSP+EESP) — см. ESPNetV2Native ниже: там итог меньше и точнее
    соответствует авторскому рецепту, ценой того, что декодер держится
    архитектурно разным между студентами (сравнение перестаёт быть чистой
    абляцией "тот же декодер, другой энкодер").
    """

    def __init__(
        self,
        variant: str = "s050",
        num_classes: int = 19,
        in_channels: int = 3,
        dropout: float = 0.1,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if variant not in ESPNETV2_VARIANTS:
            raise ValueError(f"Неизвестный вариант ESPNetv2: {variant!r}. Доступны: {sorted(ESPNETV2_VARIANTS)}")

        self.variant = variant
        self.align_corners = align_corners
        self.encoder = EESPNet(s=ESPNETV2_VARIANTS[variant]["s"], in_channels=in_channels)
        channels = self.encoder.channels[1:]  # 4 стадии, без стема (level1)

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.decoders = nn.ModuleList()
        decoder_channels = channels[-1]
        for skip_channels in reversed(channels[:-1]):
            self.decoders.append(UNetDoubleConv(decoder_channels + skip_channels, skip_channels))
            decoder_channels = skip_channels

        self.head = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

        self.tap_channels: dict[str, int] = dict(zip(STAGE_TAPS, channels))
        self.taps = FeatureTaps(STAGE_TAPS)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        for name, feature in zip(STAGE_TAPS, features):
            self.taps.tap(name, feature)

        x = self.dropout(features[-1])
        for decoder, skip in zip(self.decoders, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=self.align_corners)
            x = decoder(torch.cat([skip, x], dim=1))

        return F.interpolate(
            self.head(x), size=images.shape[2:], mode="bilinear", align_corners=self.align_corners
        )


class PSPModule(nn.Module):
    """Пирамидальный пулинг из декодера оригинала (EESPNet_Seg): вместо
    параллельных пулов фиксированного размера (классический PSPNet) — 4
    ПОСЛЕДОВАТЕЛЬНЫХ average-pool(stride=2), каждый следующий огрубляет
    предыдущий, и на каждом уровне — ДЕПТВАЙЗ 3x3 (groups=channels, почти
    бесплатно по параметрам). Все четыре карты возвращаются к исходному
    размеру и конкатенируются с непулленным входом, затем 1x1 проекция.
    """

    def __init__(self, channels: int, out_channels: int, num_stages: int = 4) -> None:
        super().__init__()
        self.stages = nn.ModuleList(C(channels, channels, 3, 1, groups=channels) for _ in range(num_stages))
        self.project = CBR(channels * (num_stages + 1), out_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        pooled = x
        outputs = [x]
        for stage in self.stages:
            pooled = F.avg_pool2d(pooled, kernel_size=3, stride=2, padding=1)
            outputs.append(F.interpolate(stage(pooled), size=size, mode="bilinear", align_corners=True))
        return self.project(torch.cat(outputs, dim=1))


class ESPNetV2Native(nn.Module):
    """EESPNet-энкодер + декодер ИЗ ОРИГИНАЛЬНОЙ статьи (EESPNet_Seg,
    sacmehta/ESPNetv2/segmentation/cnn/SegmentationModel.py), а не наш
    UNetDoubleConv (см. ESPNetV2 выше). Разница не только в размере:

    - level5 энкодеру не нужен вовсе (сегментации хватает страйда 16) —
      экономит и параметры, и сам forward;
    - декодер сразу проецирует в num_classes-мерное пространство и дальше
      работает в НЁМ (project_l3 -> act_l3 -> project_l2 -> project_l1), а
      не в широких каналах энкодера, как UNetDoubleConv — отсюда и разница
      в размере на порядок (см. таблицу вариантов в
      configs/model/student/espnetv2_native.yaml);
    - между stage4(l4, страйд 16) и stage3(l3, страйд 8) стоит не голый
      1x1, а PSPModule (пирамидальный контекст) поверх EESP-блока —
      попытка тем же бюджетом собрать контекст пошире, раз stride-32 карты
      больше нет.

    Тапы — ТОЛЬКО stage1..stage3 (страйды 4/8/16): stage4 (страйд 32) у
    этой модели физически не существует, как и у классического UNet(depth=4)
    (см. src/models/feature_taps.py) — FitNets/HeteroAKD на stage4 с этой
    моделью несовместимы, на stage1-3 работают как обычно.

    Оригинал во время обучения возвращает ещё и вспомогательный выход на
    страйде 8 (deep supervision) — здесь опущен: ни один лосс/тренер проекта
    не ждёт от student.forward ничего, кроме одного тензора логитов в
    разрешении входа (тот же контракт, что у SegFormer/SegNeXt/TimmUNet).
    """

    def __init__(
        self,
        variant: str = "s050",
        num_classes: int = 19,
        in_channels: int = 3,
        dropout: float = 0.2,
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        if variant not in ESPNETV2_VARIANTS:
            raise ValueError(f"Неизвестный вариант ESPNetv2: {variant!r}. Доступны: {sorted(ESPNETV2_VARIANTS)}")

        self.variant = variant
        self.align_corners = align_corners
        self.encoder = EESPNet(s=ESPNETV2_VARIANTS[variant]["s"], in_channels=in_channels)
        # level5/level5_0 энкодеру для сегментации не нужны (forward_until_level4
        # их не считает) — убираем совсем, как в оригинале (EESPNet_Seg.__init__
        # делает `del self.net.level5`), а не просто игнорируем: иначе эти веса
        # тихо остались бы в .parameters()/state_dict неиспользуемым мёртвым грузом.
        del self.encoder.level5
        del self.encoder.level5_0

        c1, c2, c3, c4 = self.encoder.channels[:4]  # стем, stage1, stage2, stage3(=l4 энкодера)

        self.proj_l4_to_l3 = CBR(c4, c3, 1, 1)
        psp_channels = 2 * c3
        self.psp = nn.Sequential(
            EESP(psp_channels, psp_channels // 2, stride=1, k=4, r_lim=7),
            PSPModule(psp_channels // 2, psp_channels // 2),
        )
        self.project_l3 = nn.Sequential(nn.Dropout2d(p=dropout), C(psp_channels // 2, num_classes, 1, 1))
        self.act_l3 = BR(num_classes)
        self.project_l2 = CBR(c2 + num_classes, num_classes, 1, 1)
        self.project_l1 = nn.Sequential(nn.Dropout2d(p=dropout), C(c1 + num_classes, num_classes, 1, 1))

        # taps.stage4 умышленно нет — см. докстринг класса.
        self.tap_channels: dict[str, int] = dict(zip(STAGE_TAPS[:3], (c2, c3, c4)))
        self.taps = FeatureTaps(STAGE_TAPS[:3])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out_l1, out_l2, out_l3, out_l4 = self.encoder.forward_until_level4(images)
        for name, feature in zip(STAGE_TAPS[:3], (out_l2, out_l3, out_l4)):
            self.taps.tap(name, feature)

        up_l4 = F.interpolate(
            self.proj_l4_to_l3(out_l4), scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        merged_l3 = self.psp(torch.cat([out_l3, up_l4], dim=1))
        proj_l3 = self.act_l3(self.project_l3(merged_l3))

        up_l3 = F.interpolate(proj_l3, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        merged_l2 = self.project_l2(torch.cat([out_l2, up_l3], dim=1))

        up_l2 = F.interpolate(merged_l2, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        merged_l1 = self.project_l1(torch.cat([out_l1, up_l2], dim=1))

        # merged_l1 — на страйде 2 (уровень l1 = стем); последний апсемпл x2
        # приводит к разрешению входа, как и в оригинале.
        return F.interpolate(merged_l1, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
