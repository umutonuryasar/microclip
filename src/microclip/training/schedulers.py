"""LR schedules: linear warmup + cosine decay, or constant."""
import math

from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer, cfg: dict, total_steps: int):
    kind = cfg["train"]["schedule"]
    if kind == "constant":
        return LambdaLR(optimizer, lambda _: 1.0)
    if kind == "warmup_cosine":
        warmup = cfg["train"]["warmup_steps"]

        def fn(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            t = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))

        return LambdaLR(optimizer, fn)
    raise ValueError(f"Unknown schedule: {kind}")
