"""onnxruntime-раннер"""

import logging

import numpy as np
import torch
from torch import Tensor

from src.quantization.backends.base import Runner, resolve_artifact

log = logging.getLogger(__name__)


class OnnxRuntimeRunner(Runner):
    def __init__(
        self,
        onnx_path: str,
        *,
        providers: list[str] | None = None,
        device: torch.device | str = "cpu",
        name: str | None = None,
    ) -> None:
        import onnxruntime as ort

        self.path = resolve_artifact(onnx_path)
        self.device = torch.device(device)
        providers = providers or ["CPUExecutionProvider"]

        available = ort.get_available_providers()
        missing = [provider for provider in providers if provider not in available]
        if missing:
            raise RuntimeError(
                f"Провайдеры {missing} недоступны в этой сборке onnxruntime "
                f"(есть: {available}). Молчаливый откат на CPU дал бы замер, "
                f"который выглядит как GPU, но им не является."
            )

        self._closed = False
        self.session = ort.InferenceSession(str(self.path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.name = name or f"ort_{providers[0].removesuffix('ExecutionProvider').lower()}"

    def infer(self, batch: Tensor) -> Tensor:
        array = batch.detach().cpu().numpy().astype(np.float32, copy=False)
        (output,) = self.session.run([self.output_name], {self.input_name: array})
        return torch.from_numpy(output).float().to(self.device)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        del self.session
