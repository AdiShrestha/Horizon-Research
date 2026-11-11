"""Training utilities shared by the scratch-trained emulators (eager PyTorch).

Conventions: inputs are the normalized state (u_t - m_u) / s_u plus the normalized parameter; the
network predicts the normalized one-step increment (u_{t+1} - u_t) / s_d with one history step.
All statistics come from the training trajectories only.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file


@dataclass(frozen=True)
class Normalization:
    state_mean: float
    state_sd: float
    increment_sd: float
    mu_mean: float
    mu_sd: float

    @classmethod
    def from_training(cls, u: np.ndarray, mu: np.ndarray) -> Normalization:
        """u: (n, T + 1, N) training trajectories; mu: (n,) conditioning parameter (e.g. log nu)."""
        u64 = np.asarray(u, dtype=np.float64)
        increments = np.diff(u64, axis=1)
        return cls(
            state_mean=float(u64.mean()),
            state_sd=float(u64.std()),
            increment_sd=float(increments.std()),
            mu_mean=float(np.mean(mu)),
            mu_sd=float(np.std(mu)),
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample ||pred - target|| / ||target|| over all non-batch axes, shape (batch,)."""
    dims = tuple(range(1, pred.dim()))
    num = torch.linalg.vector_norm(pred - target, dim=dims)
    den = torch.linalg.vector_norm(target, dim=dims)
    return num / den


def learning_rate(epoch: float, *, peak: float, floor: float, warmup: int, total: int) -> float:
    """Linear warm-up over ``warmup`` epochs, then cosine decay to ``floor`` at ``total``."""
    if epoch < warmup:
        return peak * (epoch + 1) / warmup
    progress = min(1.0, (epoch - warmup) / max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def tail_criterion(
    epochs: list[int], values: list[float], total: int, tolerance: float = 0.02
) -> dict[str, Any]:
    """Two-window tail rule: mean validation loss over the last two 10%-of-budget windows
    differs by less than ``tolerance`` (relative to the earlier window)."""
    window = total // 10
    first = [
        v for e, v in zip(epochs, values, strict=True) if total - 2 * window < e <= total - window
    ]
    second = [v for e, v in zip(epochs, values, strict=True) if total - window < e <= total]
    if not first or not second:
        return {"pass": False, "reason": "windows incomplete"}
    a, b = float(np.mean(first)), float(np.mean(second))
    change = abs(b - a) / a
    return {"window_1": a, "window_2": b, "relative_change": change, "pass": change < tolerance}


def atomic_torch_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_safetensors(module: torch.nn.Module, path: Path, metadata: dict[str, str]) -> None:
    """CPU state dict as SafeTensors, written atomically."""
    state = {k: v.detach().to("cpu").contiguous() for k, v in module.state_dict().items()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_file(state, str(tmp), metadata=metadata)
    os.replace(tmp, path)
