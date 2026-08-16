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
    ) -> None:
        if precision not in DTYPES:
            raise ValueError(f"precision={precision!r}; доступны {tuple(DTYPES)}.")

        self.device = torch.device(device)
        self.dtype = DTYPES[precision]
        self.name = f"torch_{precision}"

        if self.dtype is torch.float16 and self.device.type == "cpu":
            raise ValueError(
                "fp16 на CPU в eager-режиме считается программно и меряет что угодно, "
                "кроме производительности. Для fp16 нужен device=cuda."
            )

        self.model = model.eval().to(device=self.device, dtype=self.dtype)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def infer(self, batch: Tensor) -> Tensor:
        output = self.model(batch.to(device=self.device, dtype=self.dtype, non_blocking=True))
        return output.float()
