"""PTQ-пайплайн: композиция конфигов, экспорт, профиль движка, адаптер раннера.

Всё считается на CPU и на игрушечной модели — тесты обязаны проходить на
машине без CUDA и без TensorRT, иначе их не будет запускать никто, кроме
сервера.
"""

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.quantization.backends.base import make_runner
from src.quantization.build_engine import onnx_inputs, profile_shapes
from src.quantization.engine_module import RunnerModule
from src.quantization.export import export_onnx
from src.quantization.numerics import compare_runners, evaluate_runner, tensor_diff
from src.quantization.report import check_acceptance

RECIPES = ["base", "trt_fp16", "trt_fp32", "torch_fp16", "ort_cpu", "ort_cuda"]


class TinyNet(nn.Module):
    """Три слоя: экспортируется за доли секунды, но содержит и conv, и BN."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, num_classes),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.body(batch)


@pytest.fixture
def model() -> nn.Module:
    torch.manual_seed(0)
    return TinyNet().eval()


@pytest.fixture
def loader() -> DataLoader:
    torch.manual_seed(0)
    images = torch.randn(16, 3, 8, 8)
    labels = torch.randint(0, 4, (16,))
    return DataLoader(TensorDataset(images, labels), batch_size=4)


@pytest.mark.parametrize("recipe", RECIPES)
def test_quantize_recipes_compose(recipe):
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="config", overrides=[f"quantize={recipe}"])

    assert cfg.quantize.precision in ("fp32", "fp16")
    assert cfg.quantize.backend in ("torch", "onnxruntime", "tensorrt")
    assert set(cfg.quantize.stages) <= {"export", "calibrate", "build", "validate", "benchmark"}
    # opset ниже 18 dynamo-экспортер не отдаёт — молча получили бы не тот граф.
    assert not cfg.quantize.export.dynamo or cfg.quantize.export.opset >= 18


def test_training_config_has_no_quantize_by_default():
    """Ось добавлена в корень, но обучение её не видит.

    При `- quantize: null` Hydra не кладёт в конфиг None, а не создаёт ключ
    вовсе, поэтому обращаться к нему можно только через .get() — иначе
    ConfigAttributeError. Тест фиксирует именно это поведение: на нём
    построена проверка в scripts/quantize.py.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="config")
    assert cfg.get("quantize") is None
    assert "quantize" not in cfg


def test_export_produces_single_file_and_passes_parity(model, tmp_path):
    result = export_onnx(
        model,
        torch.randn(2, 3, 8, 8),
        tmp_path / "model.onnx",
        dynamic_axes={0: ("batch", 1, 8)},
    )

    assert result.path.is_file()
    # Внешних данных быть не должно: .onnx без спутника нерабочий, а по
    # расширению это не видно.
    assert result.meta["external_data_files"] == []
    assert result.meta["opset"] == 18
    assert result.meta["parity"]["passed"]
    # Проверка идёт и на другом размере батча — иначе динамическая ось могла бы
    # оказаться константой, запёкшейся при трассировке.
    assert {check["shape"][0] for check in result.meta["parity"]["checks"]} == {2, 1}


def test_export_rejects_low_opset_on_dynamo(model, tmp_path):
    with pytest.raises(ValueError, match="opset"):
        export_onnx(model, torch.randn(1, 3, 8, 8), tmp_path / "m.onnx", opset=13, verify=False)


def test_export_legacy_path_keeps_requested_opset(model, tmp_path):
    result = export_onnx(
        model, torch.randn(1, 3, 8, 8), tmp_path / "legacy.onnx", dynamo=False, opset=17
    )
    assert result.meta["opset"] == 17
    assert result.meta["parity"]["passed"]


def test_onnx_inputs_reports_dynamic_axis(model, tmp_path):
    path = tmp_path / "dyn.onnx"
    export_onnx(model, torch.randn(2, 3, 8, 8), path, dynamic_axes={0: ("batch", 1, 8)},
                verify=False)

    (name, dims), = onnx_inputs(path)
    assert name == "input"
    assert dims == [None, 3, 8, 8]


def test_profile_shapes_defaults_opt_to_max():
    assert profile_shapes([None, 3, 8, 8], {0: ("batch", 1, 16)}) == (
        (1, 3, 8, 8), (16, 3, 8, 8), (16, 3, 8, 8)
    )


def test_profile_shapes_rejects_config_out_of_sync_with_graph():
    """Граф статический, а конфиг обещает динамику — движок описал бы не то."""
    with pytest.raises(ValueError, match="статические"):
        profile_shapes([4, 3, 8, 8], {0: ("batch", 1, 16)})


def test_profile_shapes_requires_range_for_dynamic_axis():
    with pytest.raises(ValueError, match="динамические оси"):
        profile_shapes([None, 3, 8, 8], None)


def test_runner_module_works_with_trainer_evaluate(model, loader):
    """Ключевая интеграция: метрика раннера считается штатным evaluate()."""
    from src.training import evaluate

    runner = make_runner("torch", model=model, precision="fp32", device="cpu")
    direct_loss, direct_acc = evaluate(model, loader, torch.device("cpu"))
    wrapped_loss, wrapped_acc = evaluate(RunnerModule(runner), loader, torch.device("cpu"))

    assert wrapped_acc == pytest.approx(direct_acc)
    assert wrapped_loss == pytest.approx(direct_loss, rel=1e-6)


def test_compare_runners_on_identical_models_is_exact(model, loader):
    reference = make_runner("torch", model=model, precision="fp32", device="cpu")
    candidate = make_runner("torch", model=model, precision="fp32", device="cpu")

    result = compare_runners(reference, candidate, loader, device=torch.device("cpu"))
    assert result["max_abs"] == 0.0
    assert result["argmax_agreement"] == 1.0
    assert result["samples"] == 16


def test_evaluate_runner_rejects_detection(model, loader):
    runner = make_runner("torch", model=model, precision="fp32", device="cpu")
    with pytest.raises(ValueError, match="task_type"):
        evaluate_runner(runner, loader, torch.device("cpu"), task_type="detection")


def test_tensor_diff_sees_shifted_logits():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    metrics = tensor_diff(reference, reference + 0.5)

    assert metrics["max_abs"] == pytest.approx(0.5)
    # Сдвиг одинаков по классам, argmax не меняется — и это ровно тот случай,
    # ради которого max_abs смотрят отдельно от совпадения предсказаний.
    assert metrics["argmax_agreement"] == 1.0


def test_acceptance_catches_silent_prediction_drift():
    """Метрика та же, а предсказания поменялись — приёмка обязана падать."""
    verdict = check_acceptance(
        {"accuracy": 0.78}, {"accuracy": 0.78}, {"argmax_agreement": 0.90},
        max_metric_drop=0.005, min_agreement=0.99,
    )
    assert not verdict["passed"]
    assert "совпадение" in verdict["violations"][0]


def test_acceptance_passes_on_small_drop():
    verdict = check_acceptance(
        {"accuracy": 0.7878}, {"accuracy": 0.7872}, {"argmax_agreement": 0.997},
        max_metric_drop=0.005, min_agreement=0.99,
    )
    assert verdict["passed"]


class _FakeTensorRT:
    """Минимальный двойник модуля tensorrt: только то, от чего зависит выбор пути."""

    def __init__(self, *, has_fp16_flag: bool) -> None:
        self.__version__ = "10.7.0" if has_fp16_flag else "11.2.1.2"
        self.BuilderFlag = type("BuilderFlag", (), {"FP16": 4} if has_fp16_flag else {})
        self.NetworkDefinitionCreationFlag = type(
            "NetworkFlag", (), {"STRONGLY_TYPED": 1} | ({"EXPLICIT_BATCH": 0} if has_fp16_flag else {})
        )


class _FakeNetwork:
    def __init__(self, dtype_name: str) -> None:
        output = type("Tensor", (), {"dtype": type("DT", (), {"name": dtype_name})})()
        layer = type("Layer", (), {"num_outputs": 1, "get_output": lambda self, i: output})()
        self.num_layers = 1
        self.get_layer = lambda index: layer


def test_precision_route_uses_builder_flag_on_old_tensorrt():
    from src.quantization.build_engine import _apply_precision

    trt = _FakeTensorRT(has_fp16_flag=True)
    flags = []
    config = type("Config", (), {"set_flag": lambda self, flag: flags.append(flag)})()

    route = _apply_precision(trt, object(), config, _FakeNetwork("FLOAT"), "fp16")
    assert route == "BuilderFlag.FP16"
    assert flags == [4]


def test_strongly_typed_tensorrt_rejects_fp32_graph():
    """TRT 11: точность берётся из графа, и fp32-граф молча дал бы fp32-движок."""
    from src.quantization.build_engine import _apply_precision

    trt = _FakeTensorRT(has_fp16_flag=False)
    with pytest.raises(RuntimeError, match="стадии экспорта"):
        _apply_precision(trt, object(), object(), _FakeNetwork("FLOAT"), "fp16")


def test_strongly_typed_tensorrt_accepts_fp16_graph():
    from src.quantization.build_engine import _apply_precision, _network_flags

    trt = _FakeTensorRT(has_fp16_flag=False)
    assert _apply_precision(trt, object(), object(), _FakeNetwork("HALF"), "fp16")
    # STRONGLY_TYPED ставится только там, где флага точности не осталось.
    assert _network_flags(trt, "fp16") == 1 << 1
    assert _network_flags(_FakeTensorRT(has_fp16_flag=True), "fp16") == 1 << 0


def test_layer_precision_reads_weight_type():
    """Схема EngineInspector в TRT 10: точность видна по типу весов слоя."""
    from src.quantization.build_engine import layer_precision

    conv = {
        "Name": "node_Conv_292 + node_relu",
        "LayerType": "CaskConvolution",
        "Weights": {"Type": "Half", "Count": 9408},
        "Outputs": [{"Format/Datatype": "N/A due to dynamic shapes"}],
    }
    assert layer_precision(conv) == "Half"

    # У Reformat весов нет — считать его вместе со свёртками нельзя.
    reformat = {"LayerType": "Reformat", "Outputs": [{"Format/Datatype": "N/A due to dynamic shapes"}]}
    assert layer_precision(reformat) == "без весов (Reformat)"

    # Статические формы: формат тензора известен и годится как запасной источник.
    pooling = {"LayerType": "Pooling", "Outputs": [{"Format/Datatype": "Half(64,1:8,...)"}]}
    assert layer_precision(pooling).startswith("Half")


def test_batch_outside_engine_profile_is_caught_before_the_dataset(tmp_path):
    """Батч лоадера приходит из конфига обучения и легко не влезает в профиль."""
    from scripts.quantize import check_batch_fits_profile

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=["quantize=trt_fp16", "data/dataset=fake_cifar10"],
        )

    cfg.data.loader.eval_batch_size = 256          # профиль по умолчанию 1..64
    with pytest.raises(ValueError, match="вне профиля движка"):
        check_batch_fits_profile(cfg)

    cfg.data.loader.eval_batch_size = 64
    assert check_batch_fits_profile(cfg) is None


def test_report_survives_non_ascii_payload(tmp_path):
    """Отчёт содержит русский текст (причины пропуска стадий) — он обязан писаться."""
    from src.quantization.report import QuantizationReport

    report = QuantizationReport(tmp_path)
    report.stage("calibrate", {"reason": "ни масштабов, ни zero-point здесь нет"})

    written = json.loads(report.path.read_text(encoding="utf-8"))
    assert written["stages"]["calibrate"]["reason"].startswith("ни масштабов")


def test_text_io_always_declares_encoding():
    """На сервере локаль ASCII, и write_text без encoding роняет запись отчёта.

    Проверяем не поведение, а исходники: воспроизвести чужую локаль внутри
    процесса нельзя — Python читает её на уровне C при открытии файла.
    """
    import ast

    root = Path(__file__).resolve().parent.parent
    files = [*(root / "src" / "quantization").rglob("*.py"), root / "scripts" / "quantize.py"]

    offenders = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in ("write_text", "read_text"):
                continue
            if not any(keyword.arg == "encoding" for keyword in node.keywords):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")

    assert not offenders, f"файловые операции без encoding: {offenders}"
    """Проверенные на V100 границы: 10.16 и 11.2 уже без Volta, 10.3 не проверяли."""
    from src.quantization.backends.tensorrt_backend import unsupported_gpu

    assert unsupported_gpu((10, 16), (7, 5)) is None   # Turing проходит
    assert unsupported_gpu((11, 2), (8, 0)) is None    # Ampere проходит
    # Ниже 10.16 не зарекаемся: пусть отвечает сам билдер, ошибка у него внятная.
    assert unsupported_gpu((10, 3), (7, 0)) is None

    for version in ((10, 16), (11, 2)):
        message = unsupported_gpu(version, (7, 0))
        assert message is not None
        assert "sm_70" in message
        # Сообщение обязано давать рабочий путь, а не только диагноз.
        assert "torch_fp16" in message


def test_cuda_variant_mismatch_is_caught_before_tensorrt_touches_cuda():
    """`pip install tensorrt` тянет свежайшую сборку — она бывает под чужую CUDA.

    Внутри TensorRT это выглядит как cudaError 35 и pybind-nullptr, поэтому
    несовпадение обязано ловиться до первого обращения к CUDA.
    """
    from src.quantization.backends.tensorrt_backend import cuda_variant_mismatch

    assert cuda_variant_mismatch("cu12", "12.6") is None
    assert cuda_variant_mismatch(None, "12.6") is None

    message = cuda_variant_mismatch("cu13", "12.6")
    assert message is not None
    # В сообщении должна быть готовая команда починки, а не только диагноз.
    assert "pip install 'tensorrt-cu12'" in message
    assert "cudaError 35" in message


def test_require_tensorrt_is_the_single_import_point():
    """И раннер, и сборка движка ходят за TRT в одно место.

    Без tensorrt функция обязана объяснить, что делать, а не свалиться голым
    ImportError; с ним — вернуть модуль и проверить версию.
    """
    from src.quantization.backends.tensorrt_backend import MIN_TRT_VERSION, require_tensorrt
    from src.quantization.build_engine import require_tensorrt as same_function

    assert same_function is require_tensorrt

    try:
        import tensorrt  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(ModuleNotFoundError, match="целевой машине"):
            require_tensorrt()
    else:
        module = require_tensorrt()
        assert tuple(int(part) for part in module.__version__.split(".")[:2]) >= MIN_TRT_VERSION
