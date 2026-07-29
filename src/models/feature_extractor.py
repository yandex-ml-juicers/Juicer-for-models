"""Съём промежуточных карт признаков через forward-хуки."""

from torch import nn


class FeatureExtractor:
    """Регистрирует forward-хуки на именованных подмодулях и складывает
    их выходы в self.features после каждого forward'а модели.

    Контракт использования (им управляет Trainer):
      1) clear() перед forward'ом — иначе можно прочитать карты прошлого батча;
      2) remove() по окончании работы — хуки держат ссылки на модули.
    """

    def __init__(self, model: nn.Module, layer_names: list[str]) -> None:
        self.features: dict = {}
        self._handles = []

        modules = dict(model.named_modules())
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
