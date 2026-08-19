"""Численная валидация: какая потеря качества"""

import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from src.quantization.backends.base import Runner
from src.quantization.engine_module import RunnerModule

log = logging.getLogger(__name__)


def tensor_diff(reference: Tensor, candidate: Tensor) -> dict[str, float]:
    """Насколько разошлись два выхода одной и той же модели"""
    reference = reference.detach().float()
    candidate = candidate.detach().float().to(reference.device)

    diff = (reference - candidate).abs()
    scale = reference.abs().max().clamp_min(torch.finfo(torch.float32).eps)

    metrics = {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rel_to_max": (diff.max() / scale).item(),
        # Доля NaN/Inf в выходе кандидата. Это не «потеря точности», а поломка:
        # fp16 переполняется или делит почти ноль на почти ноль. Меряем явно,
        # потому что в остальных метриках NaN только портит числа, не называя
        # причину.
        "nonfinite": (~torch.isfinite(candidate)).float().mean().item(),
    }

    if reference.ndim >= 2:
        flat_reference = reference.reshape(reference.shape[0], -1)
        flat_candidate = candidate.reshape(candidate.shape[0], -1)
        # Суммы копятся в float64. У сегментации в полном разрешении на строку
        # приходится 40 миллионов чисел, и накопленная в float32 погрешность
        # даёт косинус БОЛЬШЕ единицы (наблюдали 1.0011) — величину, которой не
        # существует, и по ней потом судят об исправности экспорта.
        dot = (flat_reference * flat_candidate).sum(dim=1, dtype=torch.float64)
        norms = (
            flat_reference.pow(2).sum(dim=1, dtype=torch.float64).sqrt()
            * flat_candidate.pow(2).sum(dim=1, dtype=torch.float64).sqrt()
        )
        metrics["cosine"] = (dot / norms.clamp_min(1e-30)).mean().item()
        agreement = (reference.argmax(dim=1) == candidate.argmax(dim=1)).float().mean()
        metrics["argmax_agreement"] = agreement.item()

    return metrics


def kl_divergence(reference: Tensor, candidate: Tensor) -> float:
    """KL(fp32 || fp16) в натах НА ОДНУ ПОЗИЦИЮ.

    Нормировка на число позиций, а не на батч: у сегментации на кадр
    приходится два миллиона пикселей, и `reduction="batchmean"` превращал бы
    метрику в сумму по ним — числа порядка тысяч, ничего не значащие и
    несравнимые с классификацией. Для классификации (тензор [B, C]) позиция
    одна на объект, поэтому там значение не меняется.
    """
    reference = reference.detach().float()
    candidate = candidate.detach().float().to(reference.device)

    log_reference = F.log_softmax(reference, dim=1)
    log_candidate = F.log_softmax(candidate, dim=1)
    per_position = (log_reference.exp() * (log_reference - log_candidate)).sum(dim=1)
    return per_position.mean().item()


@torch.no_grad()
def noise_separates(comparison: dict, noise_floor: dict) -> list[str]:
    """Сигналы, по которым кандидат ОТЛИЧИМ от собственного шума модели.

    Пустой список означает, что расхождение целиком тонет в шуме: вердикт
    приёмки тогда относится к сумме двух эффектов и об эффекте точности сам по
    себе не говорит ничего.

    Почему трёх сигналов мало одного `max_abs`. Максимум берётся по миллиарду
    чисел и у недетерминированной модели определяется её собственной
    случайностью — он оказывается «внутри шума» даже тогда, когда кандидат
    заметно хуже. Проверено на int8-движке SegNeXt: max_abs 5.80 против шума
    5.91 (внутри), при совпадении предсказаний 0.9960 против 0.9978 и KL
    0.00149 против 0.000621 — то есть втрое выше шума. По одному максимуму
    пайплайн объявил бы разницу неизмеримой и увёл бы от верного вывода.
    """
    separating = []
    if comparison["max_abs"] > noise_floor["max_abs"]:
        separating.append("max_abs")
    if comparison.get("argmax_agreement", 1.0) < noise_floor.get("argmax_agreement", 0.0):
        separating.append("совпадение предсказаний")
    if comparison.get("kl", 0.0) > noise_floor.get("kl", 0.0):
        separating.append("KL")
    return separating


def compare_runners(
    reference: Runner,
    candidate: Runner,
    loader: DataLoader,
    *,
    device: torch.device,
    limit_batches: int | None = None,
) -> dict:
    """L1: прогоняет оба раннера по одним и тем же батчам и сводит расхождение"""
    # Максимумы копим списком, а не через max() на лету: NaN проигрывает любое
    # сравнение, поэтому max(2.69, nan) == 2.69 — и катастрофа исчезает из
    # метрики, которая ровно для катастроф и заведена. Реальный случай:
    # TensorRT-движок SegNeXt выдавал NaN, а max_abs показывал безобидные 2.69.
    peaks: list[float] = []
    totals = {"mean_abs": 0.0, "rel_to_max": 0.0, "cosine": 0.0, "argmax_agreement": 0.0,
              "kl": 0.0, "nonfinite": 0.0}
    samples = 0
    batches = 0

    for step, batch in enumerate(loader):
        if limit_batches is not None and step >= limit_batches:
            break

        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device, non_blocking=True)

        reference_output = reference.infer(images)
        candidate_output = candidate.infer(images)

        metrics = tensor_diff(reference_output, candidate_output)
        count = images.shape[0]

        peaks.append(metrics["max_abs"])
        for key in ("mean_abs", "rel_to_max", "cosine", "argmax_agreement", "nonfinite"):
            totals[key] += metrics.get(key, 0.0) * count
        totals["kl"] += kl_divergence(reference_output, candidate_output) * count

        samples += count
        batches += 1

    if samples == 0:
        raise ValueError("Сравнение не получило ни одного батча — пустой лоадер?")

    result = {
        "reference": reference.name,
        "candidate": candidate.name,
        "batches": batches,
        "samples": samples,
        # Худший случай, а не средний: одно переполнение fp16 на одном батче —
        # это уже поломка, и усреднение по выборке её замажет. NaN не теряется:
        # если он был хоть в одном батче, он и окажется в максимуме.
        "max_abs": float("nan") if any(math.isnan(peak) for peak in peaks) else max(peaks),
        **{key: value / samples for key, value in totals.items()},
    }

    if result["nonfinite"] > 0:
        log.error(
            "%s выдаёт NaN/Inf на %.2f%% выходов. Это не потеря точности, а поломка: "
            "fp16 переполнился или разделил почти ноль на почти ноль. Метрики ниже "
            "считать нельзя, чинить надо точность отдельных слоёв.",
            candidate.name, result["nonfinite"] * 100,
        )

    log.info(
        "L1 (%s vs %s) на %d батчах: max_abs=%.3g | cos=%.6f | argmax=%.4f | KL=%.3g",
        reference.name,
        candidate.name,
        batches,
        result["max_abs"],
        result["cosine"],
        result["argmax_agreement"],
        result["kl"],
    )
    return result


def evaluate_runner(
    runner: Runner,
    loader: DataLoader,
    device: torch.device,
    *,
    task_type: str = "classification",
    num_classes: int | None = None,
    ignore_index: int = 255,
    limit_batches: int | None = None,
) -> dict:
    """L2: метрика качества штатным кодом тренера, но на раннере"""
    module = RunnerModule(runner)

    if task_type == "classification":
        from src.training import evaluate

        loss, accuracy = evaluate(module, loader, device, limit_batches=limit_batches)
        metrics = {"loss": loss, "accuracy": accuracy}

    elif task_type == "segmentation":
        from src.training.trainer import segmentation_evaluate

        if num_classes is None:
            raise ValueError("Для сегментации нужен num_classes — из cfg.data.dataset.")
        loss, iou = segmentation_evaluate(
            module,
            loader,
            device,
            num_classes=num_classes,
            ignore_index=ignore_index,
            limit_batches=limit_batches,
        )
        results = iou.compute()
        metrics = {"loss": loss, "miou": results["miou"], "pixel_acc": results["pixel_acc"]}

    else:
        raise ValueError(
            f"task_type={task_type!r} не поддержан валидацией квантизации. "
            f"Доступны: classification, segmentation. Для детекции нужен свой "
            f"постпроцессинг предсказаний, он к движку не привязан."
        )

    log.info(
        "L2 (%s): %s",
        runner.name,
        " | ".join(f"{key}={value:.4f}" for key, value in metrics.items()),
    )
    return {"runner": runner.name, **metrics}
