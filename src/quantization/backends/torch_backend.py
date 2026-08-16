"""PyTorch-раннер"""

import logging

import torch
from torch import Tensor, nn

from src.quantization.backends.base import Runner

log = logging.getLogger(__name__)

DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


class TorchRunner(Runner):
    def __init__(
        self,
        model: nn.Module,
        *,
        precision: str = "fp32",
        device: torch.device | str = "cuda",
        channels_last: bool = False,
    ) -> None:
        """channels_last — раскладка NHWC вместо NCHW.

        Тензорные ядра читают данные в NHWC, и в NCHW cuDNN транспонирует
        тензоры сам, на каждом слое. Для свёрточных сетей в fp16 это часто и
        есть разница между «ускорения почти нет» и «ускорение вдвое».

        Раскладку обязательно включать ОБОИМ раннерам сразу: иначе эффект
        точности смешается с эффектом раскладки, и сравнение fp32 с fp16
        перестанет что-либо означать.
        """
        if precision not in DTYPES:
            raise ValueError(f"precision={precision!r}; доступны {tuple(DTYPES)}.")

        self.device = torch.device(device)
        self.dtype = DTYPES[precision]
        self.channels_last = bool(channels_last)
        self.name = f"torch_{precision}"

        if self.dtype is torch.float16 and self.device.type == "cpu":
            raise ValueError(
                "fp16 на CPU в eager-режиме считается программно и меряет что угодно, "
                "кроме производительности. Для fp16 нужен device=cuda."
            )

        self.model = model.eval().to(device=self.device, dtype=self.dtype)
        if self.channels_last:
            # type: ignore — в стабах torch у Module.to нет перегрузки с
            # memory_format, хотя в рантайме аргумент поддержан и документирован.
            self.model = self.model.to(memory_format=torch.channels_last)  # type: ignore
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def infer(self, batch: Tensor) -> Tensor:
        images = batch.to(device=self.device, dtype=self.dtype, non_blocking=True)
        if self.channels_last:
            # Вход тоже обязан быть в NHWC: модель в channels_last при
            # NCHW-входе просто вернёт всё к NCHW, и раскладка не даст ничего.
            images = images.contiguous(memory_format=torch.channels_last)
        return self.model(images).float()
