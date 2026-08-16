"""Оценка модели на валидации — обученного чекпоинта или готовых весов.

Тем же конфигом, что и обучение: датасет, трансформы и метрика берутся из
эксперимента, поэтому число сравнимо с тем, что писал тренер.

Примеры:
    # ученик из чекпоинта
    python scripts/eval.py experiment=segmentation/CWD/cityscapes_CWD_segformer_b5_to_unet_small \\
        ckpt_path=outputs/<name>/<run>/best.pt

    # учитель как он есть (веса с Hugging Face, чекпоинт не нужен)
    python scripts/eval.py experiment=segmentation/CWD/cityscapes_CWD_segformer_b5_to_unet_small \\
        eval_slot=teacher

eval_slot выбирает, какую модель конфига оценивать: student (по умолчанию)
или teacher. Для teacher ckpt_path не обязателен — если он не задан, берутся
веса, с которыми модель собралась (pretrained: cityscapes и т.п.).
"""

import logging
import math

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.data import base_loader
from src.models import MultiScaleInference
from src.training import evaluate
from src.training.trainer import segmentation_evaluate
from src.utils import resolve_device, seed_everything

log = logging.getLogger(__name__)


def _load_weights(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    """Веса из чекпоинта нашего тренера в уже собранную модель."""
    checkpoint = torch.load(to_absolute_path(ckpt_path), map_location=device, weights_only=True)

    state_dict = checkpoint.get("student_state", checkpoint)
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    # Тренеры кладут разные ключи качества: best_acc / best_miou / best_map.
    achieved = {
        name: checkpoint[name]
        for name in ("best_acc", "best_miou", "best_map")
        if name in checkpoint
    }
    log.info("Загружен чекпоинт эпохи %s %s", checkpoint.get("epoch", "?"), achieved or "")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    slot = cfg.get("eval_slot", "student")
    if slot not in {"student", "teacher"}:
        raise ValueError(f"eval_slot должен быть 'student' или 'teacher', получено {slot!r}")
    if cfg.model.get(slot) is None:
        raise ValueError(f"В конфиге нет model/{slot} — нечего оценивать")

    # Ученик без чекпоинта — это случайная инициализация, и почти всегда
    # это забытый аргумент, а не намерение. Учителю чекпоинт не нужен:
    # он собирается уже с весами.
    if slot == "student" and cfg.ckpt_path is None:
        raise ValueError("Укажи чекпоинт: python scripts/eval.py ckpt_path=outputs/…/best.pt")

    seed_everything(cfg.seed, deterministic=cfg.deterministic, warn_only=cfg.deterministic_warn_only)
    device = resolve_device(cfg.device)

    _, eval_loader = base_loader(cfg.data, cfg.task_type, seed=cfg.seed)

    model = instantiate(cfg.model[slot]).to(device)
    if cfg.ckpt_path is not None:
        _load_weights(model, cfg.ckpt_path, device)

    # Мультимасштабный прогон — та же обёртка, что тренер надевает на учителя,
    # поэтому число здесь совпадает с качеством таргетов в дистилляции.
    if slot == "teacher":
        teacher_inference = cfg.model.get("teacher_inference")
        if teacher_inference:
            teacher_inference = OmegaConf.to_container(teacher_inference, resolve=True)
            model = MultiScaleInference(model, **teacher_inference)
            log.info("Мультимасштабный прогон: %s", teacher_inference)

    amp = bool(cfg.trainer.get("amp", False)) and device.type == "cuda"
    log.info("Оцениваю model/%s на %s, amp=%s", slot, device, amp)

    if cfg.task_type == "segmentation":
        eval_loss, iou = segmentation_evaluate(
            model=model,
            loader=eval_loader,
            device=device,
            num_classes=cfg.data.dataset.num_classes,
            ignore_index=cfg.data.dataset.ignore_index,
            limit_batches=cfg.trainer.get("limit_eval_batches"),
            amp=amp,
        )
        metrics = iou.compute()
        log.info(
            "eval loss=%.4f | mIoU=%.4f | pixel acc=%.4f",
            eval_loss, metrics["miou"], metrics["pixel_acc"],
        )

        class_names = getattr(eval_loader.dataset, "classes", None)
        per_class = iou.per_class_iou().tolist()
        if class_names is not None and len(class_names) == len(per_class):
            log.info("IoU по классам (худшие сверху):")
            # NaN — класс не встретился в выборке; такие строки уводим вниз,
            # иначе они перемешиваются со значимыми при сортировке.
            for name, value in sorted(
                zip(class_names, per_class),
                key=lambda pair: float("inf") if math.isnan(pair[1]) else pair[1],
            ):
                log.info("  %-16s %s", name, "—" if math.isnan(value) else f"{value:.4f}")

        return metrics["miou"]

    if cfg.task_type == "classification":
        eval_loss, eval_acc = evaluate(model, eval_loader, device)
        log.info("eval loss=%.4f | eval acc=%.2f%%", eval_loss, eval_acc * 100)
        return eval_acc

    raise ValueError(
        f"eval.py пока не поддерживает task_type={cfg.task_type!r} "
        f"(детекция считается через DetectionTrainer)"
    )


if __name__ == "__main__":
    main()
