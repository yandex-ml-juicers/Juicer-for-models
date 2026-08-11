"""Работа с весами: каталог, скачивание, разбор чекпоинтов.

Ничего специфичного для задачи здесь нет — функции одинаково нужны
классификации, детекции и сегментации. Раньше та же логика жила инлайн
в нескольких фабриках моделей сразу; вынесена, чтобы форматы чекпоинтов
разбирались в одном месте.
"""

import shutil
import urllib.request
from pathlib import Path

import torch
from hydra.utils import to_absolute_path
from torch import nn

# Куда складываются все скачанные веса. Путь относительный: to_absolute_path
# разворачивает его от корня репозитория, а не от рабочего каталога Hydra
# (Hydra делает chdir в outputs/<name>/<дата>).
DEFAULT_WEIGHTS_DIR = "data/weights"

# Ключи, под которыми разные инструменты прячут state_dict внутри чекпоинта.
# Порядок = приоритет: наши тренеры пишут student_state, torchvision-подобные
# скрипты — model_state_dict/model, остальные — state_dict.
STATE_DICT_KEYS: tuple[str, ...] = ("student_state", "model_state_dict", "state_dict", "model")


def resolve_weights_dir(weights_dir: str | None = None) -> Path:
    """Каталог для скачиваемых весов; создаётся при первом обращении."""
    path = Path(to_absolute_path(weights_dir or DEFAULT_WEIGHTS_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, destination: Path) -> Path:
    """Скачивает url в destination, если файла ещё нет.

    Пишет во временный .part и переименовывает только после проверки
    размера. Без этого прерванная загрузка оставляет усечённый файл,
    который выглядит скачанным, а падает уже при torch.load — и понять,
    что дело в закачке, по ошибке "failed finding central directory"
    практически невозможно.
    """
    if destination.is_file():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_suffix(destination.suffix + ".part")

    print(f"Скачивание {url} -> {destination}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            expected_size = response.headers.get("Content-Length")
            with open(part_path, "wb") as part_file:
                shutil.copyfileobj(response, part_file, 1024 * 1024)

        if expected_size is not None and part_path.stat().st_size != int(expected_size):
            raise IOError(
                f"Файл скачан не полностью: {part_path.stat().st_size} из {expected_size} байт"
            )
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise

    part_path.replace(destination)
    return destination


def extract_state_dict(checkpoint: dict | object) -> dict:
    """Достаёт веса из чекпоинта любого из встречающихся форматов.

    Самый простой случай — сам state_dict без обёртки; остальные варианты
    перечислены в STATE_DICT_KEYS.

    isinstance(value, dict) обязателен, а не просто `key in checkpoint`:
    иначе чекпоинт, где под "model" лежит не state_dict (число, объект,
    строка с именем архитектуры), был бы взят как веса и упал бы позже
    и в другом месте.
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Ожидался dict-чекпоинт, получено {type(checkpoint)}")

    matched = [key for key in STATE_DICT_KEYS if isinstance(checkpoint.get(key), dict)]
    if not matched:
        return checkpoint

    if len(matched) > 1:
        # Такого среди наших чекпоинтов не бывает (тренеры пишут только
        # student_state), но у чужого чекпоинта выбор мог бы оказаться
        # неожиданным — пусть он будет виден в логе, а не угадывался.
        print(f"Чекпоинт содержит несколько наборов весов {matched}, взят {matched[0]!r}")

    return checkpoint[matched[0]]


def load_checkpoint_into(
    model: nn.Module,
    checkpoint_path: str,
    model_name: str,
    strict: bool = True,
) -> nn.Module:
    """Грузит в model веса из чекпоинта нашего тренера.

    Задача-агностична: так же работает и для классификатора, и для
    сегментатора. Типичный сценарий — модель, обученная на первом этапе,
    подставляется замороженным учителем на этапе дистилляции.

    Префикс "module." снимается: под DDP/DataParallel state_dict сохраняется
    вместе с обёрткой, а грузим мы всегда в развёрнутую модель.
    """
    weights_path = Path(to_absolute_path(checkpoint_path))
    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint file was not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = extract_state_dict(checkpoint)
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=strict)
    print(f"Weights for {model_name} has been loaded: {weights_path}")
    return model
