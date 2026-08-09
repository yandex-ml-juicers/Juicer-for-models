"""Распределённый запуск: шардирование, деление батча, коллективные операции.

Тесты с маркером `distributed` поднимают несколько процессов через mp.spawn.
Бэкенд выбирается сам: на машине с картами — NCCL и тензоры на GPU, без карт
— gloo на CPU. Поэтому один и тот же набор проверок работает и локально без
железа, и на сервере по настоящему пути.

    pytest tests/ -m "not distributed"    # быстрый прогон, без многопроцессных
    pytest tests/                         # всё
    pytest tests/ -m distributed -v       # только распределённые, с именами
"""

import contextlib
import os
import tempfile

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from hydra import compose, initialize
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from src.data import base_loader
from src.data.samplers import ShardSampler
from src.losses.base import DistillationLoss
from src.losses.cross_entropy import CrossEntropy
from src.training import Trainer, evaluate
from src.utils import distributed as D
from src.utils.metrics import AverageMeter, ConfusionMatrixAccumulator, sync_meters
from src.utils.distributed import DistInfo


# --------------------------------------------------------------------------
# ShardSampler
# --------------------------------------------------------------------------

@pytest.mark.parametrize("total", [0, 1, 3, 97, 100])
@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_shards_cover_dataset_exactly_once(total, world_size):
    """Объединение шардов = вся выборка, без пропусков и без дубликатов.

    Это главное свойство, ради которого ShardSampler написан вместо
    DistributedSampler: тот ради равной длины шардов дополняет выборку
    повторами, и метрика на eval оказывается смещённой.
    """
    dataset = range(total)
    shards = [list(ShardSampler(dataset, world_size, rank)) for rank in range(world_size)]

    merged = sorted(index for shard in shards for index in shard)
    assert merged == list(range(total))


@pytest.mark.parametrize("total", [0, 1, 3, 97, 100])
@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_shard_len_matches_iteration(total, world_size):
    """__len__ считается формулой — она обязана совпасть с реальной длиной.

    DataLoader берёт число батчей именно из len(sampler), не перебирая его.
    Разойдись формула с итерацией — часть данных потерялась бы молча.
    """
    for rank in range(world_size):
        sampler = ShardSampler(range(total), world_size, rank)
        assert len(sampler) == len(list(sampler))


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_shard_sizes_differ_by_at_most_one(world_size):
    sizes = [len(ShardSampler(range(97), world_size, rank)) for rank in range(world_size)]
    assert max(sizes) - min(sizes) <= 1


# --------------------------------------------------------------------------
# base_loader: глобальный батч и шардирование
# --------------------------------------------------------------------------

def make_data_cfg(**overrides):
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=["data/dataset=fake_cifar10", *[f"{k}={v}" for k, v in overrides.items()]],
        )
    return cfg.data


def fake_dist(rank: int, world_size: int) -> DistInfo:
    return DistInfo(rank=rank, local_rank=rank, world_size=world_size, device=torch.device("cpu"))


def test_batch_size_is_global():
    """batch_size из конфига делится между процессами, а не умножается на них.

    Следствие, ради которого это сделано: число шагов оптимизации за эпоху
    не зависит от числа карт, то есть одна строка конфига описывает один и
    тот же эксперимент на любом количестве GPU.
    """
    cfg = make_data_cfg()
    single, _ = base_loader(cfg, seed=42, dist=fake_dist(0, 1))
    sharded, _ = base_loader(cfg, seed=42, dist=fake_dist(0, 4))

    assert single.batch_size == cfg.loader.batch_size
    assert sharded.batch_size == cfg.loader.batch_size // 4
    assert len(single) == len(sharded)


def test_batch_size_not_divisible_raises():
    cfg = make_data_cfg()
    cfg.loader.batch_size = 10
    with pytest.raises(ValueError, match="не делится"):
        base_loader(cfg, seed=42, dist=fake_dist(0, 4))


def test_train_shards_are_disjoint():
    cfg = make_data_cfg()
    per_rank = []
    for rank in range(2):
        loader, _ = base_loader(cfg, seed=42, dist=fake_dist(rank, 2))
        loader.sampler.set_epoch(0)
        per_rank.append(set(loader.sampler))

    assert per_rank[0] & per_rank[1] == set()


def test_set_epoch_changes_order():
    """Без set_epoch перестановка одинакова во всех эпохах — и это молчаливая
    потеря перемешивания, а не ошибка."""
    cfg = make_data_cfg()
    loader, _ = base_loader(cfg, seed=42, dist=fake_dist(0, 2))

    loader.sampler.set_epoch(0)
    first = list(loader.sampler)
    loader.sampler.set_epoch(1)
    second = list(loader.sampler)

    assert first != second


def test_single_process_path_unchanged():
    """При world_size=1 сэмплеры не подставляются: однопроцессные запуски
    обязаны воспроизводить ранее полученные результаты бит-в-бит."""
    cfg = make_data_cfg()
    train_loader, eval_loader = base_loader(cfg, seed=42, dist=None)

    assert not isinstance(train_loader.sampler, ShardSampler)
    assert not isinstance(eval_loader.sampler, ShardSampler)
    assert train_loader.batch_size == cfg.loader.batch_size


def test_eval_loader_has_no_padding():
    """Сумма длин eval-шардов равна размеру выборки, без дополнения.

    world_size=4, а не 3: batch_size из конфига должен делиться нацело.
    """
    cfg = make_data_cfg()
    total = 0
    for rank in range(4):
        _, eval_loader = base_loader(cfg, seed=42, dist=fake_dist(rank, 4))
        total += len(eval_loader.sampler)

    _, reference = base_loader(cfg, seed=42, dist=None)
    assert total == len(reference.dataset)


# --------------------------------------------------------------------------
# distributed.py вне группы
# --------------------------------------------------------------------------

def test_dist_info_flags():
    assert fake_dist(0, 1).is_main and not fake_dist(0, 1).is_distributed
    assert fake_dist(0, 4).is_main and fake_dist(0, 4).is_distributed
    assert not fake_dist(3, 4).is_main


def test_collectives_are_noop_without_group():
    """Без поднятой группы коллективы обязаны молча возвращать своё значение:
    на этом держится работоспособность eval.py и однопроцессных запусков."""
    tensor = torch.tensor([2.0])
    assert D.all_reduce_sum_(tensor).item() == 2.0
    assert D.all_reduce_max_(tensor).item() == 2.0
    assert D.broadcast_object("value", device=torch.device("cpu")) == "value"
    D.barrier()


def test_unwrap():
    module = nn.Linear(2, 2)
    assert D.unwrap(module) is module
    assert D.unwrap(None) is None


# --------------------------------------------------------------------------
# Многопроцессные тесты (gloo, CPU)
# --------------------------------------------------------------------------

def dist_setup_for_test(world_size: int):
    """Поднимает группу: на машине с картами — через NCCL, иначе gloo на CPU.

    Автоопределение намеренное. Локально и в CI тесты идут на CPU и не требуют
    железа, а на сервере ровно те же проверки прогоняются по НАСТОЯЩЕМУ пути:
    NCCL, обмен через PCIe, тензоры на карте. Без этого весь набор
    подтверждал бы только логику поверх gloo.
    """
    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
        return D.setup(device_cfg="auto", backend="auto")
    return D.setup(device_cfg="cpu", backend="gloo")


def _wrap(module: nn.Module, info: DistInfo) -> DistributedDataParallel:
    """DDP-обёртка так же, как её строит Trainer: с device_ids на CUDA."""
    return DistributedDataParallel(
        module,
        device_ids=[info.local_rank] if info.device.type == "cuda" else None,
    )


def _worker(rank: int, world_size: int, port: int, result_queue) -> None:
    """Тело процесса: поднимает группу теми же средствами, что и train.py."""
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
    )
    info = dist_setup_for_test(world_size)
    try:
        # сумма рангов+1 = N(N+1)/2, максимум = N-1
        summed = D.all_reduce_sum_(torch.tensor([float(rank + 1)], device=info.device))
        maximum = D.all_reduce_max_(torch.tensor([float(rank)], device=info.device))

        # DDP: веса расходятся до обёртки, но должны совпасть после неё
        torch.manual_seed(rank)  # намеренно РАЗНЫЕ веса на ранках
        model = nn.Linear(4, 2).to(info.device)
        wrapped = _wrap(model, info)
        weight_after_wrap = wrapped.module.weight.detach().clone()

        # градиенты усредняются: у каждого ранка свой вход
        wrapped(torch.full((2, 4), float(rank + 1), device=info.device)).sum().backward()

        result_queue.put({
            "rank": info.rank,
            "world_size": info.world_size,
            "is_main": info.is_main,
            "sum": summed.item(),
            "max": maximum.item(),
            # Именно списками, а не тензорами: torch.multiprocessing передаёт
            # тензоры через разделяемую память, и к моменту чтения в
            # родительском процессе процесс-отправитель уже завершён.
            "weight": weight_after_wrap.tolist(),
            "grad": wrapped.module.weight.grad.detach().tolist(),
            "unwrap_ok": D.unwrap(wrapped) is model,
            "run_dir": D.broadcast_object(f"dir-from-rank-{rank}", device=info.device),
        })
    finally:
        D.cleanup()


def _run_workers(world_size: int) -> list[dict]:
    from src.utils.distributed import find_free_port

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(_worker, args=(world_size, find_free_port(), queue), nprocs=world_size, join=True)
    return sorted([queue.get() for _ in range(world_size)], key=lambda row: row["rank"])


@pytest.mark.distributed
def test_group_of_two_ranks():
    results = _run_workers(2)

    assert [row["rank"] for row in results] == [0, 1]
    assert [row["is_main"] for row in results] == [True, False]
    # all_reduce: 1 + 2 = 3, max(0, 1) = 1 — одинаково видны обоим
    assert all(row["sum"] == 3.0 for row in results)
    assert all(row["max"] == 1.0 for row in results)
    assert all(row["unwrap_ok"] for row in results)


@pytest.mark.distributed
def test_ddp_broadcasts_weights_and_averages_gradients():
    """Два инварианта DDP, на которых держится всё обучение."""
    results = _run_workers(2)

    # (1) при создании обёртки веса rank 0 разошлись всем — несмотря на то,
    # что инициализировали модели разными сидами
    assert results[0]["weight"] == results[1]["weight"]

    # (2) после backward градиенты одинаковы на всех ранках, хотя входы разные
    assert torch.allclose(
        torch.tensor(results[0]["grad"]), torch.tensor(results[1]["grad"])
    )


@pytest.mark.distributed
def test_broadcast_object_delivers_value_from_rank_zero():
    """На этом построена сверка директории запуска в train.py."""
    results = _run_workers(2)
    assert all(row["run_dir"] == "dir-from-rank-0" for row in results)


# --------------------------------------------------------------------------
# Распределённый evaluate
# --------------------------------------------------------------------------

def _fixed_eval_dataset(size: int = 17):
    """17 объектов: на 2 ранка делится как 9 + 8, на 4 — как 5+4+4+4.

    Неделимость намеренная: именно на ней ломаются реализации, которые
    усредняют средние вместо того, чтобы делить сумму на общее число объектов.
    """
    generator = torch.Generator().manual_seed(7)
    return TensorDataset(
        torch.randn(size, 4, generator=generator),
        torch.randint(0, 3, (size,), generator=generator),
    )


def _fixed_model():
    torch.manual_seed(0)
    return _ToyStudent()


def test_evaluate_is_independent_of_batch_size():
    """Разбиение на батчи не должно влиять на результат.

    Свойство ловит классическую ошибку — усреднение по батчам вместо
    взвешивания по числу объектов: тогда неполный последний батч получает
    тот же вес, что и полный.
    """
    dataset = _fixed_eval_dataset()
    model = _fixed_model()
    device = torch.device("cpu")

    whole = evaluate(model, DataLoader(dataset, batch_size=len(dataset)), device)
    split = evaluate(model, DataLoader(dataset, batch_size=5), device)

    assert split[0] == pytest.approx(whole[0], abs=1e-5)
    assert split[1] == pytest.approx(whole[1], abs=1e-9)


def _evaluate_worker(rank: int, world_size: int, port: int, result_queue) -> None:
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
    )
    info = dist_setup_for_test(world_size)
    try:
        dataset = _fixed_eval_dataset()
        loader = DataLoader(
            dataset, batch_size=4, sampler=ShardSampler(dataset, world_size, rank)
        )
        loss, acc = evaluate(_fixed_model().to(info.device), loader, info.device)
        result_queue.put({"rank": rank, "loss": loss, "acc": acc, "shard": len(loader.sampler)})
    finally:
        D.cleanup()


@pytest.mark.distributed
def test_distributed_evaluate_matches_single_process():
    """КОНТРОЛЬНАЯ ТОЧКА внедрения DDP.

    Одна и та же модель на одной выборке обязана дать одно и то же число
    независимо от числа процессов. Это единственная метрика, которая может
    совпасть точно: она не зависит ни от порядка батчей, ни от истории
    обучения. Разойдись здесь — значит либо шардирование теряет/дублирует
    объекты, либо суммы сводятся неверно.
    """
    from src.utils.distributed import find_free_port

    dataset = _fixed_eval_dataset()
    expected = evaluate(_fixed_model(), DataLoader(dataset, batch_size=4), torch.device("cpu"))

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(_evaluate_worker, args=(2, find_free_port(), queue), nprocs=2, join=True)
    results = sorted([queue.get() for _ in range(2)], key=lambda row: row["rank"])

    # шарды действительно разной длины — путь без padding задействован
    assert sorted(row["shard"] for row in results) == [8, 9]

    for row in results:
        assert row["loss"] == pytest.approx(expected[0], abs=1e-5)
        assert row["acc"] == pytest.approx(expected[1], abs=1e-9)


# --------------------------------------------------------------------------
# Главное математическое утверждение DDP
# --------------------------------------------------------------------------

GRAD_BATCH = 12  # делится на 1, 2, 3 и 4


def _fixed_batch():
    generator = torch.Generator().manual_seed(11)
    return (
        torch.randn(GRAD_BATCH, 4, generator=generator),
        torch.randint(0, 3, (GRAD_BATCH,), generator=generator),
    )


def _grad_worker(rank: int, world_size: int, port: int, result_queue) -> None:
    """Считает градиент по СВОЕМУ шарду и отдаёт то, что осталось после DDP."""
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
    )
    info = dist_setup_for_test(world_size)
    try:
        model = _fixed_model().to(info.device)
        wrapped = _wrap(model, info)

        # Данные переносим руками. DDP сам двигает на устройство только входы
        # forward'а, а метки идут мимо него — прямо в лосс.
        features, labels = _fixed_batch()
        features = features.to(info.device)
        labels = labels.to(info.device)
        shard = slice(rank * (GRAD_BATCH // world_size), (rank + 1) * (GRAD_BATCH // world_size))

        # reduction="mean" по своему шарду; DDP усредняет эти средние по рангам.
        # При равных шардах среднее средних = среднее по всему батчу — это и
        # есть утверждение, которое проверяется.
        F.cross_entropy(wrapped(features[shard]), labels[shard]).backward()

        result_queue.put({
            "rank": rank,
            "grad": model.head.weight.grad.detach().tolist(),
        })
    finally:
        D.cleanup()


@pytest.mark.distributed
@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_ddp_gradient_equals_single_process_full_batch(world_size):
    """Градиент под DDP равен градиенту однопроцессного прогона по всему батчу.

    Это и есть математическое обещание DDP: обучение на N картах с глобальным
    батчом B эквивалентно обучению на одной карте с тем же батчом. Всё
    остальное — скорость.

    Проверка точная, в отличие от сравнения кривых обучения: там расхождение
    неизбежно из-за BatchNorm по локальному куску и другого порядка данных, и
    отличить настоящую поломку от нормального шума невозможно. Здесь же любое
    расхождение — ошибка.

    Модель намеренно без BatchNorm: его статистики считаются по локальному
    батчу и равенство сломали бы законно, а не по нашей вине.
    """
    from src.utils.distributed import find_free_port

    reference = _fixed_model()
    features, labels = _fixed_batch()
    F.cross_entropy(reference(features), labels).backward()
    expected = reference.head.weight.grad

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(_grad_worker, args=(world_size, find_free_port(), queue),
             nprocs=world_size, join=True)
    results = [queue.get() for _ in range(world_size)]

    for row in results:
        # Допуск не нулевой: эталон считается на CPU, а ранки могут считать на
        # GPU другими ядрами и с другим порядком сложения. Настоящая поломка
        # синхронизации даёт расхождение на порядки больше.
        assert torch.allclose(torch.tensor(row["grad"]), expected, atol=1e-5, rtol=1e-4), (
            f"градиент на ранке {row['rank']} разошёлся с однопроцессным"
        )


# --------------------------------------------------------------------------
# Синхронизация метрик
# --------------------------------------------------------------------------

def test_sync_meters_is_noop_without_group():
    meters = {"acc": AverageMeter()}
    meters["acc"].update(0.5, n=4)
    sync_meters(meters, torch.device("cpu"))

    assert meters["acc"].sum == 2.0
    assert meters["acc"].count == 4


def test_sync_meters_ignores_empty_mapping():
    sync_meters({}, torch.device("cpu"))


def _meter_worker(rank: int, world_size: int, port: int, result_queue) -> None:
    """Каждый ранк кормит метры своей половиной значений."""
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
    )
    info = dist_setup_for_test(world_size)
    try:
        meters = {"acc": AverageMeter(), "loss": AverageMeter()}
        for value, weight in _METER_INPUTS[rank]:
            meters["acc"].update(value, n=weight)
            meters["loss"].update(value * 2, n=weight)

        confmat = ConfusionMatrixAccumulator(3, info.device)
        preds, labels = _CONFMAT_INPUTS[rank]
        confmat.update(torch.tensor(preds), torch.tensor(labels))

        sync_meters(meters, info.device)
        confmat.synchronize()

        result_queue.put({
            "rank": rank,
            "acc": meters["acc"].avg,
            "loss": meters["loss"].avg,
            "confmat": confmat.compute(),
        })
    finally:
        D.cleanup()


# Веса намеренно разные: если синхронизация усредняет средние вместо
# складывания сумм и счётчиков, результат разойдётся с эталоном.
_METER_INPUTS = {
    0: [(0.80, 128), (0.90, 128)],
    1: [(0.40, 128), (0.20, 16)],
}
_CONFMAT_INPUTS = {
    0: ([0, 1, 2, 1], [0, 1, 2, 1]),   # всё угадано
    1: ([2, 2, 0, 0], [1, 2, 0, 1]),   # часть спутана
}


@pytest.mark.distributed
def test_sync_meters_matches_single_process():
    """Метры после синхронизации = метры, накормленные объединением значений."""
    from src.utils.distributed import find_free_port

    reference = {"acc": AverageMeter(), "loss": AverageMeter()}
    for rank_inputs in _METER_INPUTS.values():
        for value, weight in rank_inputs:
            reference["acc"].update(value, n=weight)
            reference["loss"].update(value * 2, n=weight)

    reference_confmat = ConfusionMatrixAccumulator(3, torch.device("cpu"))
    for preds, labels in _CONFMAT_INPUTS.values():
        reference_confmat.update(torch.tensor(preds), torch.tensor(labels))
    expected_confmat = reference_confmat.compute()

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(_meter_worker, args=(2, find_free_port(), queue), nprocs=2, join=True)
    results = [queue.get() for _ in range(2)]

    for row in results:
        assert row["acc"] == pytest.approx(reference["acc"].avg)
        assert row["loss"] == pytest.approx(reference["loss"].avg)
        for key, value in expected_confmat.items():
            assert row["confmat"][key] == pytest.approx(value)


# --------------------------------------------------------------------------
# Trainer под DDP
# --------------------------------------------------------------------------

class _ToyStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 3)

    def forward(self, x):
        return self.head(torch.relu(self.backbone(x)))


class _LossWithAdapter(DistillationLoss):
    """Лосс с обучаемым параметром — модель адаптеров feature-KD и MGD.

    Именно он проверяет вторую DDP-обёртку: без неё этот параметр
    рассинхронизируется по ранкам молча, без единой ошибки.
    """

    requires_teacher = False

    def __init__(self) -> None:
        super().__init__()
        self.adapter = nn.Linear(3, 3, bias=False)

    def forward(self, student_logits, teacher_logits, labels, **kwargs):
        adapted = self.adapter(student_logits)
        return {"total": F.cross_entropy(adapted, labels), "ce": F.cross_entropy(adapted, labels)}


class _FeatureLossWithAdapter(DistillationLoss):
    """Лосс, воспроизводящий устройство FeatureKD и MGD: карты признаков с
    промежуточного слоя ПЛЮС обучаемые параметры.

    Именно эта комбинация и есть риск: хуки регистрируются на подмодулях до
    оборачивания в DDP, а forward потом идёт через обёртку. Если хуки
    перестанут срабатывать, карты признаков не приедут — и это выяснится
    здесь, а не на ResNet-50.
    """

    requires_teacher = False
    required_features = ("backbone",)

    def __init__(self) -> None:
        super().__init__()
        self.adapter = nn.Linear(4, 4, bias=False)

    def forward(self, student_logits, teacher_logits, labels,
                student_features=None, teacher_features=None):
        if not student_features or "backbone" not in student_features:
            raise RuntimeError(
                "forward-хуки не сработали: карты признаков не приехали в лосс. "
                "Скорее всего FeatureExtractor построен поверх DDP-обёртки."
            )
        adapted = self.adapter(student_features["backbone"])
        ce = F.cross_entropy(student_logits, labels)
        feature = adapted.pow(2).mean()
        return {"total": ce + feature, "ce": ce, "feature": feature}


def _feature_kd_worker(rank: int, world_size: int, port: int, result_queue) -> None:
    """Полный цикл обучения с feature-based лоссом на двух рангах."""
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
    )
    info = dist_setup_for_test(world_size)
    try:
        torch.manual_seed(rank)  # разные веса до оборачивания

        dataset = TensorDataset(torch.randn(16, 4), torch.randint(0, 3, (16,)))
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=4, sampler=sampler)

        student = _ToyStudent().to(info.device)
        criterion = _FeatureLossWithAdapter().to(info.device)
        optimizer = torch.optim.SGD(
            list(student.parameters()) + list(criterion.parameters()), lr=0.1
        )

        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                student=student, teacher=None, criterion=criterion,
                optimizer=optimizer, scheduler=None,
                train_loader=loader, eval_loader=loader,
                num_classes=3, dist=info, output_dir=tmp, epochs=2,
                save_best=False, save_last=False, progress_bar=False, scalars={},
            )
            trainer.fit()

        result_queue.put({
            "rank": rank,
            "student_weight": D.unwrap(trainer.student).head.weight.detach().tolist(),
            "adapter_weight": D.unwrap(trainer.criterion).adapter.weight.detach().tolist(),
            "extractor_built": trainer.student_extractor is not None,
        })
    finally:
        D.cleanup()


@pytest.mark.distributed
def test_feature_kd_hooks_and_adapters_under_ddp():
    """Этап 4: связка хуков и адаптеров под двумя редьюсерами.

    Проверяет три вещи разом:
      1) хуки, поставленные до оборачивания, срабатывают при forward через
         DDP — иначе лосс бросит RuntimeError и процесс упадёт;
      2) обучаемые параметры лосса остаются синхронными между рангами;
      3) два редьюсера под одним backward не приводят к зависанию.
    """
    from src.utils.distributed import find_free_port

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(_feature_kd_worker, args=(2, find_free_port(), queue), nprocs=2, join=True)
    results = sorted([queue.get() for _ in range(2)], key=lambda row: row["rank"])

    assert all(row["extractor_built"] for row in results), "FeatureExtractor не построен"
    assert results[0]["student_weight"] == results[1]["student_weight"]
    assert results[0]["adapter_weight"] == results[1]["adapter_weight"]


def _trainer_worker(rank: int, world_size: int, port: int, result_queue, output_root: str | None = None) -> None:
    """Прогоняет настоящий Trainer на игрушечных данных и отдаёт веса."""
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
    )
    info = dist_setup_for_test(world_size)
    try:
        torch.manual_seed(rank)  # намеренно разные веса до оборачивания

        features = torch.randn(16, 4)
        labels = torch.randint(0, 3, (16,))
        dataset = TensorDataset(features, labels)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=4, sampler=sampler)

        student = _ToyStudent().to(info.device)
        criterion = _LossWithAdapter().to(info.device)
        optimizer = torch.optim.SGD(
            list(student.parameters()) + list(criterion.parameters()), lr=0.1
        )

        # Своя директория на ранк: если чекпоинт запишет не только rank 0,
        # это будет видно прямо по содержимому папок.
        stack = contextlib.ExitStack()
        with stack:
            if output_root is None:
                run_dir = stack.enter_context(tempfile.TemporaryDirectory())
                save = False
            else:
                run_dir = os.path.join(output_root, f"rank{rank}")
                os.makedirs(run_dir, exist_ok=True)
                save = True

            trainer = Trainer(
                student=student, teacher=None, criterion=criterion,
                optimizer=optimizer, scheduler=None,
                train_loader=loader, eval_loader=loader,
                num_classes=3, dist=info, output_dir=run_dir, epochs=2,
                save_best=False, save_last=save, progress_bar=False,
                scalars={},
            )
            metrics = trainer._train_epoch(epoch=1)
            trainer.fit()
            written = sorted(os.listdir(run_dir))

        result_queue.put({
            "rank": rank,
            "metrics": metrics,
            "written": written,
            "student_weight": D.unwrap(trainer.student).head.weight.detach().tolist(),
            "adapter_weight": D.unwrap(trainer.criterion).adapter.weight.detach().tolist(),
            "student_wrapped": isinstance(trainer.student, DistributedDataParallel),
            "criterion_wrapped": isinstance(trainer.criterion, DistributedDataParallel),
        })
    finally:
        D.cleanup()


@pytest.mark.distributed
def test_trainer_keeps_student_and_criterion_in_sync():
    """Главный инвариант этапа 3.1.

    Ученик и обучаемые параметры лосса обязаны остаться одинаковыми на всех
    ранках после нескольких шагов оптимизации. Расхождение здесь не выражается
    в ошибке — обучение продолжает идти, просто ранки начинают оптимизировать
    разные модели, а в чекпоинт попадает состояние одного из них.
    """
    from src.utils.distributed import find_free_port

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(_trainer_worker, args=(2, find_free_port(), queue), nprocs=2, join=True)
    results = sorted([queue.get() for _ in range(2)], key=lambda row: row["rank"])

    assert all(row["student_wrapped"] for row in results), "ученик не обёрнут в DDP"
    assert all(row["criterion_wrapped"] for row in results), "лосс с параметрами не обёрнут в DDP"
    assert results[0]["student_weight"] == results[1]["student_weight"]
    assert results[0]["adapter_weight"] == results[1]["adapter_weight"]


@pytest.mark.distributed
def test_trainer_reports_identical_metrics_on_all_ranks():
    """Главный инвариант этапа 3.3.

    Точного совпадения с однопроцессным прогоном здесь быть не может:
    метрики считаются по модели, которая меняется в течение эпохи, и порядок
    батчей другой. А вот равенство МЕЖДУ РАНКАМИ обязано выполняться — и его
    нарушение означает ровно одно: какой-то метр забыли синхронизировать.
    """
    from src.utils.distributed import find_free_port

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(_trainer_worker, args=(2, find_free_port(), queue), nprocs=2, join=True)
    results = sorted([queue.get() for _ in range(2)], key=lambda row: row["rank"])

    loss_components_0, other_metrics_0 = results[0]["metrics"]
    loss_components_1, other_metrics_1 = results[1]["metrics"]

    assert loss_components_0.keys() == loss_components_1.keys()
    assert other_metrics_0.keys() == other_metrics_1.keys()

    for key in loss_components_0:
        assert loss_components_0[key] == pytest.approx(loss_components_1[key]), key
    for key in other_metrics_0:
        assert other_metrics_0[key] == pytest.approx(other_metrics_1[key]), key


@pytest.mark.distributed
def test_checkpoint_written_only_by_main_rank(tmp_path):
    """Чекпоинт пишет ровно один процесс, и он пригоден к загрузке.

    Каждому ранку выдана своя директория, поэтому факт лишней записи виден
    напрямую: у не-главного ранка папка обязана остаться пустой.

    Вторая половина проверки — про `unwrap`: без него ключи state_dict
    получают префикс `module.`, и такой чекпоинт не грузится ни в eval.py,
    ни в однопроцессный запуск. Ошибка проявилась бы не при сохранении, а
    много позже, при попытке воспользоваться результатом обучения.
    """
    from src.utils.distributed import find_free_port

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(
        _trainer_worker,
        args=(2, find_free_port(), queue, str(tmp_path)),
        nprocs=2,
        join=True,
    )
    results = sorted([queue.get() for _ in range(2)], key=lambda row: row["rank"])

    # Заодно проверяется, что history.csv тоже пишет только главный ранк.
    assert results[0]["written"] == ["history.csv", "last.pt"], "главный ранк не записал артефакты"
    assert results[1]["written"] == [], "не-главный ранк тоже писал артефакты"

    checkpoint = torch.load(tmp_path / "rank0" / "last.pt", map_location="cpu", weights_only=True)
    prefixed = [key for key in checkpoint["student_state"] if key.startswith("module.")]
    assert prefixed == [], f"в чекпоинт уехала DDP-обёртка: {prefixed[:3]}"
    assert [key for key in checkpoint["criterion_state"] if key.startswith("module.")] == []
    assert checkpoint["world_size"] == 2

    # Загружается в чистую модель, собранную без всякого DDP
    _ToyStudent().load_state_dict(checkpoint["student_state"])
    _LossWithAdapter().load_state_dict(checkpoint["criterion_state"])


@pytest.mark.distributed
def test_trainer_does_not_wrap_parameterless_loss():
    """DDP на модуле без обучаемых параметров бросает исключение, поэтому
    лоссы вроде CrossEntropy и HintonKD оборачивать нельзя."""
    from src.utils.distributed import find_free_port

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(_plain_loss_worker, args=(2, find_free_port(), queue), nprocs=2, join=True)
    results = [queue.get() for _ in range(2)]

    assert all(not row["criterion_wrapped"] for row in results)


def _plain_loss_worker(rank: int, world_size: int, port: int, result_queue) -> None:
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
    )
    info = dist_setup_for_test(world_size)
    try:
        dataset = TensorDataset(torch.randn(8, 4), torch.randint(0, 3, (8,)))
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=4, sampler=sampler)
        student = _ToyStudent().to(info.device)

        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                student=student, teacher=None, criterion=CrossEntropy().to(info.device),
                optimizer=torch.optim.SGD(student.parameters(), lr=0.1), scheduler=None,
                train_loader=loader, eval_loader=loader,
                num_classes=3, dist=info, output_dir=tmp, epochs=1,
                save_best=False, save_last=False, progress_bar=False, scalars={},
            )
        result_queue.put({
            "criterion_wrapped": isinstance(trainer.criterion, DistributedDataParallel),
        })
    finally:
        D.cleanup()
