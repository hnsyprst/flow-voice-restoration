import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int = 1000,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    """
    Linear warmup for `warmup_steps`, then cosine decay to `min_lr_ratio` of the peak over the
    remaining `total_steps`. Scales each param group's own base LR, so per-group LRs are preserved.

    :param optimizer: Optimizer whose learning rates to schedule.
    :type optimizer: Optimizer
    :param total_steps: Total number of optimizer steps in the run.
    :type total_steps: int
    :param warmup_steps: Number of linear-warmup steps (clamped to `[1, total_steps - 1]`).
    :type warmup_steps: int
    :param min_lr_ratio: Floor of the cosine decay, as a fraction of the peak LR.
    :type min_lr_ratio: float
    """

    warmup_steps = max(1, min(warmup_steps, total_steps - 1))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)
