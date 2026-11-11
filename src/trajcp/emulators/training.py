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


ROLLOUT_MARKS = (1, 10, 25, 50, 100, 150, 200)


def rollout_error_summary(pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """Error summary of rollouts. pred, truth: (n, T, N) predicted and true states for t = 1..T.

    Per step: relative L2, sup-norm error in physical units, and sup-norm error relative to the
    sup norm of the true u_t; median and 90th percentile over trajectories at ROLLOUT_MARKS.
    Non-finite errors count as +inf (never dropped); quantiles are order statistics
    (method 'higher') so that +inf never turns into NaN."""
    pred64 = np.asarray(pred, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    diff = pred64 - truth64
    with np.errstate(invalid="ignore", over="ignore"):
        rel_l2 = np.linalg.norm(diff, axis=2) / np.linalg.norm(truth64, axis=2)
        sup = np.abs(diff).max(axis=2)
        sup_rel = sup / np.abs(truth64).max(axis=2)
    out: dict[str, Any] = {
        "trajectories": int(pred.shape[0]),
        "non_finite_trajectories": int((~np.isfinite(diff).all(axis=(1, 2))).sum()),
    }
    for name, err in (("rel_l2", rel_l2), ("sup", sup), ("sup_rel", sup_rel)):
        err = np.where(np.isfinite(err), err, np.inf)
        marks = [t for t in ROLLOUT_MARKS if t <= err.shape[1]]
        out[f"median_{name}"] = {
            str(t): float(np.quantile(err[:, t - 1], 0.5, method="higher")) for t in marks
        }
        out[f"p90_{name}"] = {
            str(t): float(np.quantile(err[:, t - 1], 0.9, method="higher")) for t in marks
        }
    return out


def step_inputs(
    u: torch.Tensor, mu: torch.Tensor, norm: Normalization
) -> tuple[torch.Tensor, torch.Tensor]:
    """u: (b, N) physical state; mu: (b,) normalized parameter -> network input and FiLM vector."""
    x = (u - norm.state_mean) / norm.state_sd
    channel = mu[:, None].expand_as(x)
    return torch.stack([x, channel], dim=1), mu[:, None]


def predict_next(
    model: torch.nn.Module, u: torch.Tensor, mu: torch.Tensor, norm: Normalization
) -> torch.Tensor:
    x, c = step_inputs(u, mu, norm)
    return u + norm.increment_sd * model(x, c)[:, 0]


@torch.no_grad()
def rollout(
    model: torch.nn.Module, u: torch.Tensor, mu: torch.Tensor, norm: Normalization
) -> np.ndarray:
    """Autoregressive rollout from u[:, 0]; returns predicted states t = 1..T, shape (n, T, N)."""
    model.eval()
    state = u[:, 0]
    states = []
    for _ in range(1, u.shape[1]):
        state = predict_next(model, state, mu, norm)
        states.append(state.cpu().numpy())
    return np.stack(states, axis=1)
