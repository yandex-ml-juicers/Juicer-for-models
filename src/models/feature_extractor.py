"""Съём промежуточных карт признаков через forward-хуки."""

from torch import nn
from torch.nn.parallel import DataParallel, DistributedDataParallel


def unwrap_model(model: nn.Module) -> nn.Module:
    """Разворачивает модель из-под обёрток DDP / DataParallel / torch.compile.

    Обёртка добавляет свой уровень в named_modules(): тап, который в конфиге
    лосса записан как "taps.stage3", под DDP называется "module.taps.stage3",
    а под torch.compile — "_orig_mod.taps.stage3". Имена в конфиге про обёртки
    ничего не знают и знать не должны, поэтому хуки вешаются на развёрнутую
    модель.

    На срабатывание хуков это не влияет: DDP.forward вызывает self.module(...),
    то есть forward внутренних подмодулей идёт как обычно. Каждый ранг при этом
    видит карты СВОЕЙ реплики — ровно то, что нужно, так как лосс тоже
    считается по локальному под-батчу.

    Цикл, а не одна проверка: обёртки комбинируются (DDP поверх compiled-модели
    и наоборот).
    """
    while True:
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model = model.module
            continue

        # torch.compile возвращает OptimizedModule с оригиналом в _orig_mod.
        # Проверяем по атрибуту, а не по типу: импорт torch._dynamo ради
        # isinstance тянет компилятор и ломается между версиями torch.
        inner = getattr(model, "_orig_mod", None)
        if isinstance(inner, nn.Module):
            model = inner
            continue

        return model


class FeatureExtractor:
    """Регистрирует forward-хуки на именованных подмодулях и складывает
    их выходы в self.features после каждого forward'а модели.

    Модель разворачивается из-под DDP/DataParallel/torch.compile (см.
    unwrap_model), поэтому имена слоёв в конфигах лоссов одинаковы для
    одиночного и распределённого запуска.

    Контракт использования (им управляет Trainer):
      1) clear() перед forward'ом — иначе можно прочитать карты прошлого батча;
      2) remove() по окончании работы — хуки держат ссылки на модули.
    """

    def __init__(self, model: nn.Module, layer_names: list[str]) -> None:
        self.features: dict = {}
        self._handles = []

        modules = dict(unwrap_model(model).named_modules())
        for layer_name in layer_names:
            if layer_name not in modules:
                available = ", ".join(name for name in modules if name.count(".") == 0 and name)
                raise ValueError(
                    f"Слой {layer_name!r} не найден в модели. Слои верхнего уровня: {available}"
                )
            handle = modules[layer_name].register_forward_hook(self._make_hook(layer_name))
            self._handles.append(handle)

    def _make_hook(self, layer_name: str):
        def hook(module, inputs, output):
            self.features[layer_name] = output

        return hook

    def clear(self) -> None:
        self.features.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.features.clear()
