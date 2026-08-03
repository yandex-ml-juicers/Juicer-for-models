"""Выбор вычислительного устройства."""

import logging

import torch

log = logging.getLogger(__name__)

def resolve_device(device: str = "auto") -> torch.device:
    """Разворачивает строку из конфига в torch.device.
    Args:
        device: "auto" (cuda если доступна, иначе cpu), "cuda", "cuda:N" или "cpu".
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("В конфиге запрошена CUDA, но torch.cuda.is_available() == False")
    if resolved.type == "cuda":
        #torch.cuda.set_device(resolved)
        log.info("Устройство: %s (%s)", resolved, torch.cuda.get_device_name(resolved))
    else:
        log.info("Устройство: %s", resolved)
    return resolved
