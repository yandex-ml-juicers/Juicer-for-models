from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    epochs: int,
    start_factor: float = 0.01,
    end_factor: float = 1.0,
    eta_min: float = 1e-6
) -> SequentialLR:
    """
    LinearLR scheduler for warm up ecpochs
    CosineAnnealingLR scheduler for other epochs
    """
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=start_factor,  # начинаем с 1% основного LR
        end_factor=end_factor,
        total_iters=warmup_epochs,
    )

    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=eta_min,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            warmup_scheduler,
            main_scheduler,
        ],
        milestones=[warmup_epochs],
    )

    return scheduler