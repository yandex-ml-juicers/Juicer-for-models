"""Экспорт обученной модели в ONNX"""

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
import torch
from torch import Tensor, nn

log = logging.getLogger(__name__)

MIN_DYNAMO_OPSET = 18
EXTERNAL_DATA_THRESHOLD_BYTES = int(1.8 * 1024**3)

DynamicAxisSpec = Mapping[int, tuple[str, int, int]]


@dataclass(frozen=True)
class ExportResult:
    path: Path
    meta: dict


def _param_bytes(model: nn.Module) -> int:
    """Сколько весов и буферов у модели в байтах"""
    parameters = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return parameters + buffers


def sha256_file(path: Path) -> str:
    """Хеш артефакта"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _modules_with_hooks(model: nn.Module) -> list[str]:
    """Ищем модули с forward-hooks"""
    return [
        name or "<root>"
        for name, module in model.named_modules()
        if module._forward_hooks or module._forward_pre_hooks
    ]


def _validate_dynamic_axes(spec: DynamicAxisSpec, sample: Tensor) -> None:
    for axis, (name, low, high) in spec.items():
        if not 0 <= axis < sample.ndim:
            raise ValueError(
                f"Динамическая ось {axis} ({name}) вне ранга образца: "
                f"у входа {sample.ndim} осей, форма {tuple(sample.shape)}."
            )
        if not name.isidentifier():
            raise ValueError(f"Имя оси {name!r} должно быть валидным идентификатором.")
        if low < 1 or high < low:
            raise ValueError(f"Ось {name}: ожидалось 1 <= min <= max, получено ({low}, {high}).")
        if low == high:
            raise ValueError(
                f"Ось {name}: min == max == {low}. Такая ось не динамическая - убери её "
                f"из спецификации, иначе экспортер и TensorRT будут описывать её по-разному."
            )
        if not low <= sample.shape[axis] <= high:
            raise ValueError(
                f"Ось {name}: образец имеет размер {sample.shape[axis]}, что вне заявленного диапазона [{low}, {high}]."
            )


def _dynamo_dynamic_shapes(spec: DynamicAxisSpec) -> tuple[dict, ...]:
    from torch.export import Dim

    return ({axis: Dim(name, min=low, max=high) for axis, (name, low, high) in spec.items()},)


def _legacy_dynamic_axes(
    spec: DynamicAxisSpec, input_name: str, output_name: str
) -> dict[str, dict[int, str]]:
    axes: dict[str, dict[int, str]] = {
        input_name: {axis: name for axis, (name, _, _) in spec.items()}
    }
    if 0 in spec:
        axes[output_name] = {0: spec[0][0]}
    return axes


def _verification_inputs(sample: Tensor, spec: DynamicAxisSpec) -> list[Tensor]:
    inputs = [torch.randn_like(sample)]

    if 0 in spec:
        _, low, high = spec[0]
        current = sample.shape[0]
        other = low if current != low else min(high, current + 1)
        if other != current:
            shape = (other, *sample.shape[1:])
            inputs.append(torch.randn(shape, dtype=sample.dtype, device=sample.device))

    return inputs


def check_onnx_parity(
    onnx_path: str | Path,
    model: nn.Module,
    inputs: list[Tensor],
    *,
    input_name: str = "input",
    atol: float = 1e-4,
) -> dict:
    """Уровень L0: сходятся ли выходы torch и ONNX на одном и том же входе"""
    import onnxruntime as ort

    from src.quantization.numerics import tensor_diff

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output_name = session.get_outputs()[0].name

    was_training = model.training
    model.eval()

    checks = []
    try:
        for tensor in inputs:
            with torch.no_grad():
                reference = model(tensor)
            (candidate,) = session.run(
                [output_name], {input_name: tensor.detach().cpu().numpy()}
            )
            metrics : dict[str, Any] = tensor_diff(reference, torch.from_numpy(candidate))
            metrics["shape"] = list(tensor.shape)
            checks.append(metrics)
    finally:
        if was_training:
            model.train()

    worst = max(check["max_abs"] for check in checks)
    return {"atol": atol, "passed": worst <= atol, "max_abs": worst, "checks": checks}


def export_onnx(
    model: nn.Module,
    sample: Tensor,
    output_path: str | Path,
    *,
    opset: int = MIN_DYNAMO_OPSET,
    dynamo: bool = True,
    dynamic_axes: DynamicAxisSpec | None = None,
    external_data: bool | None = None, # как сохранять .onnx в один файл или .onnx + .onnx.data (для больших моделей)
    optimize: bool = True,
    input_name: str = "input",
    output_name: str = "output",
    verify: bool = True, # пронать проверку
    parity_atol: float = 1e-4,
) -> ExportResult:
    """Экспортирует модель в ONNX и возвращает путь + метаданные запуска.
       Модель переводится в eval и возвращается в исходный режим на выходе
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec: DynamicAxisSpec = dict(dynamic_axes or {})
    if spec:
        _validate_dynamic_axes(spec, sample)

    if dynamo and opset < MIN_DYNAMO_OPSET:
        raise ValueError(
            f"opset={opset} несовместим с dynamo-экспортером: его реализации операторов "
            f"начинаются с {MIN_DYNAMO_OPSET}. "
            f"Ставь opset>={MIN_DYNAMO_OPSET} либо dynamo=false."
        )
    if not dynamo and spec and set(spec) - {0}:
        raise ValueError(
            f"Динамические оси {sorted(set(spec) - {0})} требуют dynamo=true: легаси-экспортер "
            f"не выводит форму выхода из графа и объявить их у выхода нечем."
        )

    param_bytes = _param_bytes(model)
    if external_data is None:
        external_data = param_bytes > EXTERNAL_DATA_THRESHOLD_BYTES

    hooked = _modules_with_hooks(model)
    if hooked:
        log.warning(
            "На модели висят forward-хуки (%s%s). Для экспорта они не нужны и могут "
            "сломать трассировку: сними FeatureExtractor перед вызовом.",
            ", ".join(hooked[:3]),
            " и др." if len(hooked) > 3 else "",
        )
        raise ValueError("Убери forward-hooks с модели!")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            reference = model(sample)
        if not isinstance(reference, Tensor):
            raise TypeError(
                f"forward вернул {type(reference).__name__}, а ONNX-экспорт ждёт Tensor. "
                f"Оберни модель так, чтобы наружу шли логиты"
            )

        log.info(
            "ONNX-экспорт: вход %s -> выход %s | opset=%d | %s | %s",
            tuple(sample.shape),
            tuple(reference.shape),
            opset,
            "dynamo" if dynamo else "torchscript (legacy)",
            f"динамические оси: { {axis: name for axis, (name, *_) in spec.items()} }"
            if spec
            else "статический вход",
        )

        if dynamo:
            torch.onnx.export(
                model,
                (sample,),
                str(output_path),
                dynamo=dynamo,
                external_data=external_data,
                optimize=optimize,
                opset_version=opset,
                input_names=[input_name],
                output_names=[output_name],
                dynamic_shapes=_dynamo_dynamic_shapes(spec) if spec else None,
            )
        else:
            torch.onnx.export(
                model,
                (sample,),
                str(output_path),
                dynamo=False,
                opset_version=opset,
                export_params=True,
                do_constant_folding=True,
                input_names=[input_name],
                output_names=[output_name],
                dynamic_axes=_legacy_dynamic_axes(spec, input_name, output_name) if spec else None,
            )

        onnx.checker.check_model(str(output_path), full_check=True)
        graph = onnx.load(str(output_path), load_external_data=False)

        sidecars = sorted(
            p.name for p in output_path.parent.glob(f"{output_path.name}*") if p != output_path
        )
        if sidecars:
            log.warning(
                "Веса вынесены во внешние файлы (%s): .onnx без них нерабочий, "
                "копировать и класть в data/deploy надо всё вместе.",
                ", ".join(sidecars),
            )

        meta  = {
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "external_data": external_data,
            "external_data_files": sidecars,
            "exporter": "dynamo" if dynamo else "torchscript",
            "opset": max(imp.version for imp in graph.opset_import if imp.domain in ("", "ai.onnx")),
            "opset_requested": opset,
            "optimize": optimize,
            "ir_version": graph.ir_version,
            "input_name": input_name,
            "output_name": output_name,
            "input_shape": list(sample.shape),
            "output_shape": list(reference.shape),
            "dynamic_axes": {str(axis): list(value) for axis, value in spec.items()},
            "param_count": sum(p.numel() for p in model.parameters()),
            "param_bytes": param_bytes,
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "parity": None,
        }

        if verify:
            try:
                meta["parity"] = check_onnx_parity(
                    output_path,
                    model,
                    _verification_inputs(sample, spec),
                    input_name=input_name,
                    atol=parity_atol,
                )
            except ModuleNotFoundError:
                log.warning(
                    "onnxruntime не установлен. L0-проверка экспорта пропущена, "
                    "meta['parity'] останется null. На сервере проверка обязана проходить"
                )
    finally:
        if was_training:
            model.train()

    parity = meta["parity"]
    if parity is not None:
        log.info(
            "L0 (torch vs onnx): max_abs=%.3g при atol=%.3g -> %s",
            parity["max_abs"],
            parity["atol"],
            "OK" if parity["passed"] else "РАСХОЖДЕНИЕ",
        )
        if not parity["passed"]:
            raise RuntimeError(
                f"Экспорт изменил численность модели: max_abs={parity['max_abs']:.3g} "
                f"> atol={parity['atol']:.3g}. Дальше по пайплайну идти нельзя - расхождение "
                f"fp16-движка будет списано на fp16, хотя сломан экспорт. "
                f"Проверки: {parity['checks']}"
            )

    log.info("ONNX сохранён: %s (%.1f МиБ)", output_path, meta["size_bytes"] / 1024**2)
    return ExportResult(path=output_path, meta=meta)
