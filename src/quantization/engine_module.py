import logging

from torch import Tensor, nn

from src.quantization.backends.base import Runner

log = logging.getLogger(__name__)


class RunnerModule(nn.Module):
    """nn.Module - обёртка над раннером. Обучению не подлежит: параметров нет."""

    def __init__(self, runner: Runner) -> None:
        super().__init__()
        self.runner = runner

    def forward(self, batch: Tensor) -> Tensor:
        output = self.runner.infer(batch)
        return output.to(batch.device) if output.device != batch.device else output

    def train(self, mode: bool = True) -> "RunnerModule":
        if mode:
            log.debug("RunnerModule.train() проигнорирован: движок неизменяем.")
        return super().train(False)

    def extra_repr(self) -> str:
        return f"runner={self.runner.name}"
