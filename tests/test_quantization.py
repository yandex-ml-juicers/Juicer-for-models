"""PTQ-пайплайн: композиция конфигов, экспорт, профиль движка, адаптер раннера.

Всё считается на CPU и на игрушечной модели — тесты обязаны проходить на
машине без CUDA и без TensorRT, иначе их не будет запускать никто, кроме
сервера.
"""

import json
import math
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

RECIPES = ["base", "trt_fp16", "trt_fp32", "trt_fp16_seg", "trt_int8", "torch_fp16", "ort_cpu", "ort_cuda"]


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

    assert cfg.quantize.precision in ("fp32", "fp16", "int8")
    assert cfg.quantize.backend in ("torch", "onnxruntime", "tensorrt")
    assert set(cfg.quantize.stages) <= {"export", "calibrate", "build", "validate", "benchmark"}
    # int8 без калибровки — движок с масштабами «из воздуха»: стадия обязана быть.
    assert cfg.quantize.precision != "int8" or "calibrate" in cfg.quantize.stages
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


class _NoisyNet(TinyNet):
    """Модель со случайностью внутри forward — как NMF-разложение в SegNeXt."""

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return super().forward(batch) + torch.rand(batch.shape[0], 4) * 0.5


def test_channels_last_keeps_the_same_answers(model, loader):
    """NHWC — про скорость, не про математику: ответы обязаны совпасть."""
    from src.quantization.backends.torch_backend import TorchRunner

    nchw = TorchRunner(model, precision="fp32", device="cpu")
    nhwc = TorchRunner(model, precision="fp32", device="cpu", channels_last=True)

    assert nhwc.channels_last and not nchw.channels_last
    result = compare_runners(nchw, nhwc, loader, device=torch.device("cpu"))
    assert result["max_abs"] == pytest.approx(0.0, abs=1e-5)


def test_speedup_summary_reports_the_ratio(caplog):
    """В логе должно быть «во сколько раз», а не только миллисекунды."""
    import logging

    from src.quantization.benchmark import log_speedup_summary

    rows = [
        {"runner": "torch_fp32", "mode": "compute", "batch_size": 1, "p50_ms": 36.5},
        {"runner": "torch_fp16", "mode": "compute", "batch_size": 1, "p50_ms": 30.9},
    ]
    with caplog.at_level(logging.INFO):
        log_speedup_summary(rows, baseline="torch_fp32")

    assert "1.18x" in caplog.text


def test_noise_floor_separates_model_randomness_from_precision(loader):
    """Раннер, сравнённый сам с собой, и есть собственный шум модели."""
    torch.manual_seed(0)
    runner = make_runner("torch", model=_NoisyNet().eval(), precision="fp32", device="cpu")

    noise = compare_runners(runner, runner, loader, device=torch.device("cpu"))
    assert noise["max_abs"] > 0.1, "случайная модель обязана расходиться сама с собой"

    deterministic = make_runner("torch", model=TinyNet().eval(), precision="fp32", device="cpu")
    assert compare_runners(deterministic, deterministic, loader,
                           device=torch.device("cpu"))["max_abs"] == 0.0


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


def test_kl_is_per_position_and_comparable_across_tasks():
    """У сегментации на кадр миллионы пикселей — KL обязан быть на позицию.

    Иначе метрика превращается в сумму по пикселям (в реальном прогоне выходило
    2.5e3 вместо долей нáта) и не сравнима ни с классификацией, ни между
    разрешениями.
    """
    from src.quantization.numerics import kl_divergence

    torch.manual_seed(0)
    logits = torch.randn(2, 19)
    shifted = logits + torch.randn(2, 19) * 0.01

    flat = kl_divergence(logits, shifted)
    # Тот же тензор, разложенный по 64x64 пикселям: KL на позицию не меняется.
    spatial = kl_divergence(
        logits[:, :, None, None].expand(2, 19, 64, 64).contiguous(),
        shifted[:, :, None, None].expand(2, 19, 64, 64).contiguous(),
    )
    # Допуск на порядок суммирования: 64*64 одинаковых слагаемых во float32
    # дают доли процента расхождения. Без нормировки разница была бы в 4096 раз.
    assert spatial == pytest.approx(flat, rel=1e-2)
    assert 0 < flat < 1


def test_tensor_diff_sees_shifted_logits():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    metrics = tensor_diff(reference, reference + 0.5)

    assert metrics["max_abs"] == pytest.approx(0.5)
    # Сдвиг одинаков по классам, argmax не меняется — и это ровно тот случай,
    # ради которого max_abs смотрят отдельно от совпадения предсказаний.
    assert metrics["argmax_agreement"] == 1.0


def test_nan_does_not_hide_in_the_worst_case_metric(loader):
    """NaN проигрывает сравнение в max(), и катастрофа исчезает из max_abs.

    Реальный случай: TensorRT-движок SegNeXt выдавал NaN, а max_abs показывал
    безобидные 2.69 — потому что первый батч был конечным и выиграл сравнение.
    """
    from src.quantization.backends.torch_backend import TorchRunner

    class _Diverging(TinyNet):
        """Ломается не сразу: первый батч считает честно, дальше выдаёт NaN."""

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            output = super().forward(batch)
            return output if self.calls <= 1 else output * float("nan")

    reference = TorchRunner(TinyNet().eval(), precision="fp32", device="cpu")
    broken = TorchRunner(_Diverging().eval(), precision="fp32", device="cpu")

    result = compare_runners(reference, broken, loader, device=torch.device("cpu"))
    assert math.isnan(result["max_abs"]), "NaN обязан доехать до худшего случая"
    assert result["nonfinite"] > 0.5


def test_acceptance_names_nan_a_breakage_not_a_metric_drop():
    verdict = check_acceptance(
        {"miou": 0.808}, {"miou": 0.255},
        {"argmax_agreement": 0.733, "nonfinite": 0.21},
        task_type="segmentation", max_metric_drop=0.005, min_agreement=0.999,
    )
    assert not verdict["passed"]
    # Именно первым: иначе пойдут искать деградацию точности вместо переполнения.
    assert "NaN/Inf" in verdict["violations"][0]


class _ConstraintLayer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.num_outputs = 1
        self.precision = None
        self.output_types: list = []

    def set_output_type(self, index: int, dtype) -> None:
        self.output_types.append((index, dtype))


class _ConstraintNetwork:
    def __init__(self, names: list[str]) -> None:
        self.layers = [_ConstraintLayer(name) for name in names]
        self.num_layers = len(names)

    def get_layer(self, index: int) -> _ConstraintLayer:
        return self.layers[index]


def test_precision_constraints_cover_the_whole_chain():
    """Опасна цепочка, а не отдельная операция: между якорями всё в fp32.

    Реальный случай: `bmm` в fp32, а следующее за ним деление в fp16 — и
    результат всё равно NaN, потому что знаменатель там почти ноль.
    """
    from src.quantization.build_engine import constrain_layer_precision

    trt = type("trt", (), {"float32": "FP32"})
    network = _ConstraintNetwork([
        "node_conv2d_1", "node_relu_2",           # энкодер, остаётся в fp16
        "node_bmm_1", "node_div_5", "node_bmm_2",  # NMF: якоря и то, что между
        "node_conv2d_9",                           # снова fp16
    ])

    summary = constrain_layer_precision(trt, network, ["bmm"])

    assert summary["index_range"] == [2, 4]
    assert summary["layers_constrained"] == 3, "деление между bmm обязано попасть в fp32"
    assert network.get_layer(3).precision == "FP32"
    assert network.get_layer(0).precision is None and network.get_layer(5).precision is None


def test_precision_constraints_fail_loudly_on_stale_pattern():
    """Не нашедший ничего шаблон = молча отключённая защита. Это хуже, чем её нет."""
    from src.quantization.build_engine import constrain_layer_precision

    trt = type("trt", (), {"float32": "FP32"})
    with pytest.raises(ValueError, match="не нашли ни одного слоя"):
        constrain_layer_precision(trt, _ConstraintNetwork(["node_conv2d_1"]), ["bmm"])


def test_acceptance_uses_the_noise_floor_when_the_model_is_stochastic():
    """Порог 0.999 недостижим, если модель даёт 0.9978 сама с собой.

    На этом реально забраковался исправный fp32-движок: 0.9979 против порога
    0.9990 при шуме 0.9978.
    """
    metrics = ({"miou": 0.8079}, {"miou": 0.8076})
    comparison = {"argmax_agreement": 0.99788, "nonfinite": 0.0}

    strict = check_acceptance(*metrics, comparison, task_type="segmentation",
                              max_metric_drop=0.005, min_agreement=0.999)
    assert not strict["passed"], "без замера шума порог остаётся абсолютным"

    aware = check_acceptance(*metrics, comparison, task_type="segmentation",
                             max_metric_drop=0.005, min_agreement=0.999,
                             noise_floor={"argmax_agreement": 0.99783})
    assert aware["passed"], "кандидат не хуже, чем модель сама себе"
    assert aware["noise_agreement"] == 0.99783


def test_acceptance_catches_nan_metrics_that_l1_missed():
    """Редкий NaN не виден на десяти кадрах L1, но виден на всей выборке L2.

    Реальный случай: `nonfinite=0` на сравнении, `loss=NaN` на валидации.
    mIoU при этом почти не просела — argmax от NaN молча возвращает класс.
    """
    verdict = check_acceptance(
        {"miou": 0.8079, "loss": 0.134},
        {"miou": 0.8052, "loss": float("nan")},
        {"argmax_agreement": 0.9979, "nonfinite": 0.0},
        task_type="segmentation", max_metric_drop=0.005, min_agreement=0.999,
        noise_floor={"argmax_agreement": 0.9978},
    )
    assert not verdict["passed"]
    assert "NaN/Inf" in verdict["violations"][0] and "loss" in verdict["violations"][0]


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


def test_normalize_drift_is_reported(caplog):
    """Обучали с одной нормировкой, валидируем с другой — метрика молча просядет."""
    import logging

    from omegaconf import OmegaConf

    from scripts.quantize import warn_on_normalize_drift

    trained = OmegaConf.create(
        {"data": {"normalize": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.247, 0.2435, 0.2616]}}}
    )
    current = OmegaConf.create(
        {"data": {"dataset": {"normalize": {"mean": [0.485, 0.456, 0.406],
                                            "std": [0.229, 0.224, 0.225]}}}}
    )

    with caplog.at_level(logging.WARNING):
        warn_on_normalize_drift(current, trained)
    assert "Нормировка входа разошлась" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        warn_on_normalize_drift(trained, trained)
    assert caplog.text == ""


def test_layer_precision_reads_weight_type():
    """Схема EngineInspector в TRT 10: точность видна по типу весов слоя."""
    from src.quantization.build_engine import layer_precision

    conv = {
        "Name": "node_Conv_292 + node_relu",
        "LayerType": "CaskConvolution",
        "Weights": {"Type": "Half", "Count": 9408},
        "Outputs": [{"Format/Datatype": "N/A due to dynamic shapes"}],
    }
    assert layer_precision(conv) == "FP16"

    # У Reformat весов нет, а формат при динамическом входе неизвестен. Это
    # «определить не удалось», а не «точность отсутствует»: путать нельзя ровно
    # там, где разбор и нужен — в вопросе «применился ли int8 вообще».
    reformat = {"LayerType": "Reformat", "Outputs": [{"Format/Datatype": "N/A due to dynamic shapes"}]}
    assert layer_precision(reformat) == "не определена (Reformat)"

    # У слоя без весов тип берётся из формата выходного тензора. Строка формата
    # несёт и раскладку — в сводку должен попасть только тип, иначе гистограмма
    # разъезжается на десяток категорий вида "Channel major FP16 format ...".
    pooling = {
        "LayerType": "Pooling",
        "Outputs": [{"Format/Datatype": "Channel major FP16 format where channel % 8 == 0"}],
    }
    assert layer_precision(pooling) == "FP16"
    assert layer_precision({"Weights": {"Type": "Float"}}) == "FP32"
    assert layer_precision({"Outputs": [{"Format/Datatype": "Row major linear FP32"}]}) == "FP32"


def test_batch_outside_engine_profile_is_caught_before_the_dataset(tmp_path):
    """Батч лоадера приходит из конфига обучения и легко не влезает в профиль."""
    from scripts.quantize import check_batch_fits_profile

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=["quantize=trt_fp16", "data/dataset=fake_cifar10"],
        )

    cfg.data.loader.eval_batch_size = 256          # профиль по умолчанию 1..64
    with pytest.raises(ValueError, match=r"принимает \[1, 64\]"):
        check_batch_fits_profile(cfg)

    cfg.data.loader.eval_batch_size = 64
    assert check_batch_fits_profile(cfg) is None


def test_sample_shape_comes_from_onnx_not_from_the_dataset(model, tmp_path):
    """Замер считает на случайных тензорах — датасет ему не нужен вовсе."""
    from scripts.quantize import resolve_sample_shape

    onnx_path = tmp_path / "model.onnx"
    export_onnx(model, torch.randn(2, 3, 8, 8), onnx_path,
                dynamic_axes={0: ("batch", 1, 8)}, verify=False)

    def explode():
        raise AssertionError("лоадер не должен строиться, если форма известна из .onnx")

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="config", overrides=["quantize=trt_fp16"])

    assert resolve_sample_shape(cfg, onnx_path, explode) == [3, 8, 8]
    # Без .onnx форма берётся из конфига датасета — тоже без обращения к данным.
    cfg.data.dataset.image_size = 224
    assert resolve_sample_shape(cfg, tmp_path / "нет.onnx", explode) == [3, 224, 224]


def test_benchmark_batches_are_checked_against_the_profile():
    """Замер без датасета не должен требовать согласования батча валидации."""
    from scripts.quantize import check_batch_fits_profile

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="config", overrides=["quantize=trt_fp16"])

    cfg.quantize.stages = ["benchmark"]
    cfg.data.loader.eval_batch_size = 256          # валидации нет — не важно
    assert check_batch_fits_profile(cfg) is None

    cfg.quantize.benchmark.batch_sizes = [1, 128]  # а вот это вне профиля 1..64
    with pytest.raises(ValueError, match="батчей замера"):
        check_batch_fits_profile(cfg)


def test_static_input_requires_exact_batch():
    """У сегментации в полном разрешении вход статический — батч обязан совпасть."""
    from scripts.quantize import check_batch_fits_profile

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=["quantize=trt_fp16_seg", "data/dataset=fake_cifar10"],
        )

    assert not cfg.quantize.export.dynamic_axes, "у сегментационного рецепта вход статический"

    cfg.data.loader.eval_batch_size = 4
    with pytest.raises(ValueError, match="ровно 1"):
        check_batch_fits_profile(cfg)

    cfg.data.loader.eval_batch_size = 1
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


# ---------------------------------------------------------------------------
# Калибровка под int8
# ---------------------------------------------------------------------------


class _FakeInt8Base:
    """Двойник базового класса калибратора TensorRT: нужен только конструктор."""


class _FakeCalibratorTRT:
    """Модуль tensorrt в объёме, достаточном для сборки калибратора."""

    __version__ = "10.3.0"
    IInt8EntropyCalibrator2 = _FakeInt8Base
    IInt8MinMaxCalibrator = _FakeInt8Base


def _calibration_data(tmp_path, *, num_samples=8, batch_size=4, source=None):
    from src.quantization.calibration import calibration_dir, collect_calibration_samples

    torch.manual_seed(0)
    images = torch.randn(max(num_samples, 8), 3, 8, 8)
    labels = torch.zeros(images.shape[0], dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    return collect_calibration_samples(
        loader,
        output_dir=calibration_dir(tmp_path),
        num_samples=num_samples,
        batch_size=batch_size,
        source=source if source is not None else {"split": "train"},
    )


def test_calibration_sample_count_is_rounded_down_to_whole_batches(tmp_path):
    """Неполный батч TensorRT отдать нельзя — выборка обязана делиться нацело."""
    data = _calibration_data(tmp_path, num_samples=10, batch_size=4)

    assert data.num_samples == 8
    assert data.num_batches == 2
    assert data.batch_shape == (4, 3, 8, 8)


def test_calibration_manifest_survives_a_round_trip(tmp_path):
    from src.quantization.calibration import load_calibration_data

    data = _calibration_data(tmp_path)
    reloaded = load_calibration_data(tmp_path)

    assert reloaded.meta == data.meta
    assert reloaded.batch_shape == data.batch_shape
    # Статистики выборки — страховка от калибровки с чужой нормировкой.
    assert len(reloaded.meta["stats"]["channel_mean"]) == 3


def test_calibration_reuse_notices_a_different_split(tmp_path):
    """Выборка с другого сплита неотличима на вид, а масштабы даёт другие."""
    from src.quantization.calibration import load_calibration_data, matches_request

    _calibration_data(tmp_path, num_samples=8, batch_size=4, source={"split": "train"})
    data = load_calibration_data(tmp_path)

    assert matches_request(data, num_samples=8, batch_size=4, source={"split": "train"})
    assert not matches_request(data, num_samples=8, batch_size=4, source={"split": "eval"})
    assert not matches_request(data, num_samples=16, batch_size=4, source={"split": "train"})


def test_calibrator_serves_every_batch_once_and_then_stops(tmp_path):
    """None из get_batch — штатный конец выборки, по нему билдер строит гистограммы."""
    from src.quantization.calibration import make_calibrator

    data = _calibration_data(tmp_path, num_samples=8, batch_size=4)
    handle = make_calibrator(_FakeCalibratorTRT(), data, cache_path=None, device="cpu")

    assert handle.calibrator.get_batch_size() == 4
    assert handle.shape == (4, 3, 8, 8)

    served = [handle.calibrator.get_batch(["input"]) for _ in range(3)]
    assert [isinstance(item, list) for item in served] == [True, True, False]
    assert served[-1] is None


def test_calibration_cache_is_not_reused_for_another_model(tmp_path):
    """Кэш масштабов не помнит, из какой сети он получен, — помнит паспорт рядом."""
    from src.quantization.calibration import make_calibrator

    data = _calibration_data(tmp_path)
    cache = tmp_path / "scales.cache"
    common = {"cache_path": cache, "device": "cpu"}

    first = make_calibrator(_FakeCalibratorTRT(), data, onnx_sha256="aaa", **common)
    assert first.meta["cache_reused"] is False
    first.calibrator.write_calibration_cache(b"scales")

    same = make_calibrator(_FakeCalibratorTRT(), data, onnx_sha256="aaa", **common)
    assert same.meta["cache_reused"] is True
    assert same.calibrator.read_calibration_cache() == b"scales"

    other = make_calibrator(_FakeCalibratorTRT(), data, onnx_sha256="bbb", **common)
    assert other.meta["cache_reused"] is False
    assert other.calibrator.read_calibration_cache() is None


def test_calibration_cache_without_a_passport_is_ignored(tmp_path):
    from src.quantization.calibration import make_calibrator

    data = _calibration_data(tmp_path)
    cache = tmp_path / "scales.cache"
    cache.write_bytes(b"scales from nowhere")

    handle = make_calibrator(
        _FakeCalibratorTRT(), data, cache_path=cache, onnx_sha256="aaa", device="cpu"
    )
    assert handle.meta["cache_reused"] is False


def test_int8_needs_a_calibration_set(tmp_path):
    """Диапазоны активаций взять неоткуда, кроме как прогнав модель на данных."""
    from src.quantization.build_engine import _attach_calibrator

    with pytest.raises(ValueError, match="calibrate"):
        _attach_calibrator(
            _FakeCalibratorTRT(), None, None,
            calibration=None, cache_path=None, algorithm="entropy2",
            onnx_sha256="x", input_name="input", shape_min=(1, 3, 8, 8), shape_max=(8, 3, 8, 8),
        )


def test_calibration_batch_outside_the_profile_is_caught(tmp_path):
    """Калибровка идёт через тот же вход, что и инференс: форма обязана быть допустимой."""
    from types import SimpleNamespace

    from src.quantization.build_engine import _attach_calibrator

    data = _calibration_data(tmp_path, num_samples=8, batch_size=4)

    with pytest.raises(ValueError, match="профиль движка"):
        _attach_calibrator(
            _FakeCalibratorTRT(), SimpleNamespace(), SimpleNamespace(),
            calibration=data, cache_path=None, algorithm="entropy2",
            onnx_sha256="x", input_name="input",
            shape_min=(1, 3, 8, 8), shape_max=(2, 3, 8, 8),
            device="cpu",
        )


def test_int8_route_lets_the_builder_fall_back_to_fp16():
    """Фолбэк — правильный деплойный режим и плохой инструмент измерения."""
    from src.quantization.build_engine import _apply_int8

    trt = _FakeTensorRT(has_fp16_flag=True)
    trt.BuilderFlag = type("BuilderFlag", (), {"FP16": 4, "INT8": 3})
    builder = type("Builder", (), {"platform_has_fast_int8": True})()

    flags: list[int] = []
    config = type("Config", (), {"set_flag": lambda self, flag: flags.append(flag)})()

    assert _apply_int8(trt, builder, config, fp16_fallback=True) == "BuilderFlag.INT8 + FP16"
    assert flags == [3, 4]

    flags.clear()
    assert _apply_int8(trt, builder, config, fp16_fallback=False) == "BuilderFlag.INT8"
    assert flags == [3]


def test_int8_on_tensorrt_without_calibrator_api_says_what_to_do():
    """В ветке без неявной квантизации int8 задаётся узлами QDQ, а не флагом."""
    from src.quantization.build_engine import _apply_int8

    trt = _FakeTensorRT(has_fp16_flag=False)
    with pytest.raises(RuntimeError, match="QuantizeLinear"):
        _apply_int8(trt, object(), object())


def test_unknown_calibration_algorithm_lists_the_available_ones(tmp_path):
    from src.quantization.calibration import make_calibrator

    data = _calibration_data(tmp_path)
    with pytest.raises(ValueError, match="entropy2"):
        make_calibrator(_FakeCalibratorTRT(), data, algorithm="percentile", device="cpu")


def test_build_options_separate_engines_that_look_identical():
    """Движок с fp32_layers и без — один путь, один размер, разное поведение."""
    from src.quantization.build_engine import build_options

    plain = build_options(precision="fp16")
    guarded = build_options(precision="fp16", fp32_layers=["bmm"])
    assert plain != guarded

    # Порядок шаблонов сборку не меняет — пересобирать из-за него незачем.
    assert build_options(precision="fp16", fp32_layers=["bmm", "clamp"]) == build_options(
        precision="fp16", fp32_layers=["clamp", "bmm"]
    )

    # int8-ключи есть только у int8: иначе fp16-движки пересобирались бы от
    # смены настройки, которая к ним не относится.
    assert "int8_fp16_fallback" not in plain
    assert build_options(precision="int8", int8_fp16_fallback=False)["int8_fp16_fallback"] is False


# ---------------------------------------------------------------------------
# Достоверность замера
# ---------------------------------------------------------------------------


def test_impossible_measurements_are_flagged():
    """Числа из прогона int8 от 2026-08-18: оба нарушения выглядели обычно."""
    from src.quantization.benchmark import check_measurement_sanity

    measured = {
        ("torch_fp32", "compute"): {1: 6.41, 8: 6.72, 32: 19.28},
        ("torch_fp32", "e2e"): {1: 4.15, 8: 7.46, 32: 13.24},
        ("trt_int8", "compute"): {1: 1.23, 8: 3.93, 32: 3.71},
        ("trt_int8", "e2e"): {1: 3.79, 8: 2.04, 32: 8.21},
    }
    # p99 задан ровным, чтобы проверка хвоста не примешивалась: тест про
    # структурные инварианты, а не про разлёт.
    rows = [
        {"runner": runner, "mode": mode, "batch_size": batch,
         "p50_ms": p50, "p99_ms": p50 * 1.05}
        for (runner, mode), timings in measured.items()
        for batch, p50 in timings.items()
    ]
    warnings = check_measurement_sanity(rows)

    # e2e — это compute плюс копия с хоста, меньше он быть не может.
    assert any("torch_fp32 batch=1" in text for text in warnings)
    # Батч 8 не может считаться быстрее батча 1 — работы больше.
    assert any("batch=8" in text and "batch=1" in text for text in warnings)

    # А вот compute batch=32 (3.71) против batch=8 (3.93) — разница 5.6%,
    # это ничья в пределах допуска, и жаловаться на неё не на что.
    assert not any("batch=32" in text and "batch=8" in text for text in warnings)


def test_a_clean_measurement_raises_no_flags():
    """Числа прогона №1 из журнала, включая законный разлёт хвоста у eager."""
    from src.quantization.benchmark import check_measurement_sanity

    rows = [
        # p99/p50 = 1.73: eager шумит и на свободной карте, это не аномалия.
        {"runner": "torch_fp32", "mode": "compute", "batch_size": 1, "p50_ms": 4.07, "p99_ms": 7.02},
        {"runner": "torch_fp32", "mode": "compute", "batch_size": 8, "p50_ms": 5.87, "p99_ms": 6.10},
        {"runner": "torch_fp32", "mode": "compute", "batch_size": 32, "p50_ms": 8.68, "p99_ms": 8.90},
        {"runner": "trt_fp16", "mode": "compute", "batch_size": 1, "p50_ms": 1.29, "p99_ms": 1.30},
        {"runner": "trt_fp16", "mode": "compute", "batch_size": 8, "p50_ms": 1.39, "p99_ms": 1.41},
        {"runner": "trt_fp16", "mode": "compute", "batch_size": 32, "p50_ms": 2.63, "p99_ms": 2.66},
        {"runner": "trt_fp16", "mode": "e2e", "batch_size": 1, "p50_ms": 1.39, "p99_ms": 1.42},
        {"runner": "trt_fp16", "mode": "e2e", "batch_size": 8, "p50_ms": 1.82, "p99_ms": 1.85},
        {"runner": "trt_fp16", "mode": "e2e", "batch_size": 32, "p50_ms": 4.33, "p99_ms": 4.40},
    ]
    assert check_measurement_sanity(rows) == []


def test_layer_precision_falls_back_to_the_tactic_name():
    """Движок с динамическим входом пишет в Format/Datatype «N/A»."""
    from src.quantization.build_engine import layer_precision

    dynamic_output = {"Outputs": [{"Format/Datatype": "N/A"}], "LayerType": "CaskConvolution"}

    # Без тактики честно признаёмся, что не знаем, вместо «без весов».
    assert layer_precision(dynamic_output).startswith("не определена")

    # Имя ядра NVIDIA точность содержит: i8i8 — целочисленное, f16f16 — половинное.
    assert layer_precision({
        **dynamic_output,
        "TacticName": "sm80_xmma_fprop_implicit_gemm_i8i8_i8i32_f32_nhwc",
    }) == "INT8"
    assert layer_precision({
        **dynamic_output,
        "TacticName": "sm70_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc",
    }) == "FP16"


def test_layer_precision_reads_the_input_when_the_output_is_unknown():
    """У Reformat выход бывает N/A, а вход — нет."""
    from src.quantization.build_engine import layer_precision

    assert layer_precision({
        "LayerType": "Reformat",
        "Outputs": [{"Format/Datatype": "N/A"}],
        "Inputs": [{"Format/Datatype": "Channel major FP16 format where channel % 8 == 0"}],
    }) == "FP16"
