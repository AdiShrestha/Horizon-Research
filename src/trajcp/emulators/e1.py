"""Shared pieces of the teacher-forced FNO emulator (E1): model, epoch loop and validation.

Used by ``scripts/02_train_e1.py`` and the throughput measurement. The loop is the parallel
workspace's (ported from commit a1500f9, ``02_train_e1.py``) with the batch size and peak learning
rate as arguments; at batch 64 and peak 1e-3 it is numerically the same loop.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from trajcp.emulators.fno import FNO
from trajcp.emulators.training import (
    Normalization,
    learning_rate,
    relative_l2,
    step_inputs,
)

WIDTH, MODES, LAYERS = 64, 16, 4
PEAK_LR, FLOOR_LR, WARMUP, WEIGHT_DECAY, BETAS, BATCH = (
    1e-3,
    1e-6,
    5,
    1e-4,
    (0.9, 0.99),
    64,
)
LR_CAP = 3e-3


def build_model() -> FNO:
    return FNO(
        1,
        in_channels=2,
        out_channels=1,
        width=WIDTH,
        modes=MODES,
        layers=LAYERS,
        cond_dim=1,
    )


def scaled_peak_lr(
    batch: int, base: float = PEAK_LR, ref: int = BATCH, cap: float = LR_CAP
) -> float:
    """A06: peak learning rate x sqrt(B / 64), capped."""
    return min(cap, base * (batch / ref) ** 0.5)


def make_optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(), lr=PEAK_LR, betas=BETAS, weight_decay=WEIGHT_DECAY
    )


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    u_fit: torch.Tensor,
    mu_fit: torch.Tensor,
    norm: Normalization,
    *,
    epoch: int,
    total_epochs: int,
    seed: int,
    batch: int,
    peak_lr: float,
    stop: Callable[[], bool] = lambda: False,
    on_step: Callable[[int, torch.Tensor], None] | None = None,
) -> tuple[float, int]:
    """One epoch over all (trajectory, step) pairs; returns (mean loss, batches).

    ``on_step(i, loss)`` is called after every optimizer step (used by the device cross-check);
    ``stop()`` is polled after each step so a signal can interrupt the epoch."""
    device = u_fit.device
    steps = u_fit.shape[1] - 1
    pairs = u_fit.shape[0] * steps
    batches = (pairs + batch - 1) // batch
    order = torch.randperm(
        pairs, generator=torch.Generator().manual_seed(seed * 100_003 + epoch)
    )
    order = order.to(device)
    total = torch.zeros((), device=device)
    for b in range(batches):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(
                epoch + b / batches,
                peak=peak_lr,
                floor=FLOOR_LR,
                warmup=WARMUP,
                total=total_epochs,
            )
        idx = order[b * batch : (b + 1) * batch]
        traj, t = idx // steps, idx % steps
        cur, nxt = u_fit[traj, t], u_fit[traj, t + 1]
        x, c = step_inputs(cur, mu_fit[traj], norm)
        loss = relative_l2(model(x, c)[:, 0], (nxt - cur) / norm.increment_sd).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.detach()
        if on_step is not None:
            on_step(b, loss.detach())
        if stop():
            break
    return float(total) / batches, batches
