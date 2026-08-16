"""Выбор вычислительного устройства"""

import logging

import torch

log = logging.getLogger(__name__)

def resolve_device(device: str = "auto", local_rank: int = 0) -> torch.device:
    """Разворачивает строку из конфига в torch.device. 
        torch.device - пара <ТИП, LOCAL_RANK>.
        На данном этапе это просто структура-описание, по которой потом вызовут
        torch.cuda.set_device(device) - syscall, чтобы все вычисления шли на этом устройстве
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("В конфиге запрошена CUDA, но torch.cuda.is_available() == False")
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", local_rank)
    if resolved.type == "cuda":
        #torch.cuda.set_device(resolved)
        log.info("Устройство: %s (%s)", resolved, torch.cuda.get_device_name(resolved))
    else:
        log.info("Устройство: %s", resolved)
    return resolved
