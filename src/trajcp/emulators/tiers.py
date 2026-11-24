"""Stability tiers E1, E2 and E3 (brief §8.1-8.3; IC-02 WP-D; A07, A18) for the 1D tasks.

Every tier predicts the normalized one-step increment from the normalized state plus the task's
normalized parameter vector c, given both as constant input channels and as the FiLM vector:
    u_{t+1} = u_t + increment_sd * net([(u_t - state_mean) / state_sd, c], c).

* E1 FNO-TF: one-step teacher forcing over every (trajectory, step) pair per epoch. On P2 this is
  numerically the loop of ``trajcp.emulators.e1.train_epoch`` (tested).
* E3 FNO-unroll: the E1 network; teacher forcing for the first 10% of the epochs, then unrolled
  backpropagation through m = 10 steps from N_pairs // m random starts per epoch (A07), loss = mean
  over the m steps of the relative L2 between the predicted normalized increment and the normalized
  increment that reaches the true next state from the current predicted state.
* E2 U-Net-PF: the pushforward trick with K = 3 (two no-grad model steps from the true state, then
  one step with gradient), Gaussian input noise sigma = 1e-3 in normalized units on the gradient step,
  peak learning rate 3e-4; one sample per (trajectory, start) pair with start <= T - K per epoch.
Optimizer, schedule and checkpoints are shared: AdamW (0.9, 0.99), weight decay 1e-4, linear warm-up
over 5 epochs, cosine to 1e-6 at the budget, batch 64.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch

from trajcp.emulators.fno import FNO
from trajcp.emulators.training import learning_rate, relative_l2
from trajcp.emulators.unet import UNet

TIERS = ("E1", "E2", "E3")
AUX_TIERS = ("E1D", "E1W")  # B10 E1-dropout, B15 E1-Wang label perturbation (brief §8.4)
DROPOUT, WANG_SIGMA = 0.1, 0.05
FNO_WIDTH, FNO_MODES, FNO_LAYERS = 64, 16, 4
UNET_CHANNELS, UNET_BLOCKS = (32, 64, 96), 2
PEAK_LR = {"E1": 1e-3, "E2": 3e-4, "E3": 1e-3, "E1D": 1e-3, "E1W": 1e-3}
FLOOR_LR, WARMUP, WEIGHT_DECAY, BETAS, BATCH = 1e-6, 5, 1e-4, (0.9, 0.99), 64
E3_UNROLL, E3_WARM_FRACTION = 10, 0.1
E2_PUSHFORWARD, E2_NOISE = 3, 1e-3


@dataclass(frozen=True)
class TaskSpec:
    name: str
    cond: tuple[str, ...]  # meta columns, in order; each is standardized with D_fit statistics
    steps: int


TASKS = {
    "p1": TaskSpec("p1", ("c", "log_nu", "log_delta"), 500),
    "p2": TaskSpec("p2", ("log_nu",), 200),
    "p3": TaskSpec("p3", ("nu4",), 500),
}


@dataclass(frozen=True)
class CondNormalization:
    state_mean: float
    state_sd: float
    increment_sd: float
    cond_mean: tuple[float, ...]
    cond_sd: tuple[float, ...]

    @classmethod
    def from_training(cls, u: np.ndarray, cond: np.ndarray) -> CondNormalization:
        """u: (n, T + 1, N) fit trajectories; cond: (n, k) raw parameters of the same trajectories."""
        u64 = np.asarray(u, dtype=np.float64)
        c64 = np.asarray(cond, dtype=np.float64)
        return cls(
            state_mean=float(u64.mean()),
            state_sd=float(u64.std()),
            increment_sd=float(np.diff(u64, axis=1).std()),
            cond_mean=tuple(float(v) for v in c64.mean(axis=0)),
            cond_sd=tuple(float(v) for v in c64.std(axis=0)),
        )

    def cond(self, raw: np.ndarray) -> np.ndarray:
        return (np.asarray(raw, np.float64) - np.array(self.cond_mean)) / np.array(self.cond_sd)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build(tier: str, cond_dim: int) -> torch.nn.Module:
    if tier in ("E1", "E3", "E1D", "E1W"):
        return FNO(
            1,
            in_channels=1 + cond_dim,
            out_channels=1,
            width=FNO_WIDTH,
            modes=FNO_MODES,
            layers=FNO_LAYERS,
            cond_dim=cond_dim,
            dropout=DROPOUT if tier == "E1D" else 0.0,
        )
    if tier == "E2":
        return UNet(1, 1 + cond_dim, 1, UNET_CHANNELS, UNET_BLOCKS, cond_dim)
    raise ValueError(f"unknown tier {tier!r}")


def make_optimizer(model: torch.nn.Module, peak: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=peak, betas=BETAS, weight_decay=WEIGHT_DECAY)


def net_inputs(
    u: torch.Tensor, c: torch.Tensor, norm: CondNormalization
) -> tuple[torch.Tensor, torch.Tensor]:
    """u: (b, N) physical states; c: (b, k) normalized parameters -> (b, 1 + k, N) and (b, k)."""
    x = (u - norm.state_mean) / norm.state_sd
    channels = c[:, :, None].expand(-1, -1, x.shape[-1])
    return torch.cat([x[:, None, :], channels], dim=1), c


def increment(
    model: torch.nn.Module, u: torch.Tensor, c: torch.Tensor, norm: CondNormalization
) -> torch.Tensor:
    x, f = net_inputs(u, c, norm)
    return model(x, f)[:, 0]


def step(
    model: torch.nn.Module, u: torch.Tensor, c: torch.Tensor, norm: CondNormalization
) -> torch.Tensor:
    return u + norm.increment_sd * increment(model, u, c, norm)


def _set_lr(optimizer: torch.optim.Optimizer, epoch: float, peak: float, total: int) -> None:
    lr = learning_rate(epoch, peak=peak, floor=FLOOR_LR, warmup=WARMUP, total=total)
    for group in optimizer.param_groups:
        group["lr"] = lr


def _order(count: int, seed: int, epoch: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed * 100_003 + epoch)
    return torch.randperm(count, generator=gen).to(device)


def train_epoch(
    tier: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    u_fit: torch.Tensor,
    c_fit: torch.Tensor,
    norm: CondNormalization,
    *,
    epoch: int,
    total_epochs: int,
    seed: int,
    peak_lr: float,
    batch: int = BATCH,
    stop: Callable[[], bool] = lambda: False,
    on_step: Callable[[int, torch.Tensor], None] | None = None,
    labels: torch.Tensor | None = None,
) -> tuple[float, int, str]:
    """One epoch of the tier's training rule; returns (mean loss, optimizer steps, phase)."""
    device = u_fit.device
    n_traj, steps = u_fit.shape[0], u_fit.shape[1] - 1
    sd = norm.increment_sd
    if tier in ("E1", "E1D", "E1W") or (
        tier == "E3" and epoch < round(E3_WARM_FRACTION * total_epochs)
    ):
        phase, horizon = "one-step", 1
    elif tier == "E3":
        phase, horizon = "unroll", E3_UNROLL
    elif tier == "E2":
        phase, horizon = "pushforward", E2_PUSHFORWARD
    else:
        raise ValueError(f"unknown tier {tier!r}")
    starts_per_traj = steps - horizon + 1
    order = _order(n_traj * starts_per_traj, seed, epoch, device)
    if phase == "unroll":
        order = order[: (n_traj * steps) // E3_UNROLL]
    count = order.numel()
    batches = (count + batch - 1) // batch
    noise_gen = None
    if phase == "pushforward":
        noise_gen = torch.Generator(device=device).manual_seed(seed * 100_003 + epoch + 7)
    total = torch.zeros((), device=device)
    for b in range(batches):
        _set_lr(optimizer, epoch + b / batches, peak_lr, total_epochs)
        idx = order[b * batch : (b + 1) * batch]
        traj, t = idx // starts_per_traj, idx % starts_per_traj
        c = c_fit[traj]
        if phase == "one-step":
            cur = u_fit[traj, t]
            nxt = (labels if labels is not None else u_fit)[traj, t + 1]
            loss = relative_l2(increment(model, cur, c, norm), (nxt - cur) / sd).mean()
        elif phase == "unroll":
            state = u_fit[traj, t]
            losses = []
            for j in range(E3_UNROLL):
                pred = increment(model, state, c, norm)
                target = (u_fit[traj, t + j + 1] - state) / sd
                losses.append(relative_l2(pred, target))
                state = state + sd * pred
            loss = torch.stack(losses).mean()
        else:
            state = u_fit[traj, t]
            with torch.no_grad():
                for _ in range(E2_PUSHFORWARD - 1):
                    state = step(model, state, c, norm)
            x, f = net_inputs(state, c, norm)
            noise = torch.randn(x[:, :1].shape, generator=noise_gen, device=device)
            x = torch.cat([x[:, :1] + E2_NOISE * noise, x[:, 1:]], dim=1)
            pred = model(x, f)[:, 0]
            target = (u_fit[traj, t + E2_PUSHFORWARD] - state) / sd
            loss = relative_l2(pred, target).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.detach()
        if on_step is not None:
            on_step(b, loss.detach())
        if stop():
            break
    return float(total) / batches, batches, phase


@torch.no_grad()
def validate(
    model: torch.nn.Module, u: torch.Tensor, c: torch.Tensor, norm: CondNormalization
) -> dict[str, float]:
    """Selection metric for every tier: one-step relative L2 of the normalized increment over all
    validation pairs; also the one-step relative L2 of the predicted state."""
    model.eval()
    losses, state_errors = [], []
    for t in range(u.shape[1] - 1):
        cur, nxt = u[:, t], u[:, t + 1]
        pred = increment(model, cur, c, norm)
        losses.append(relative_l2(pred, (nxt - cur) / norm.increment_sd))
        state_errors.append(relative_l2(cur + norm.increment_sd * pred, nxt))
    model.train()
    return {
        "val_loss": float(torch.cat(losses).mean()),
        "val_state_rel_l2": float(torch.cat(state_errors).mean()),
    }


@torch.no_grad()
def rollout(
    model: torch.nn.Module,
    u0: torch.Tensor,
    c: torch.Tensor,
    norm: CondNormalization,
    steps: int,
    batch: int = 256,
) -> np.ndarray:
    """Predictions for t = 1..steps from u0 (n, N), shape (n, steps, N) float32 on the CPU."""
    model.eval()
    out = np.empty((u0.shape[0], steps, u0.shape[1]), dtype=np.float32)
    for first in range(0, u0.shape[0], batch):
        state, cc = u0[first : first + batch], c[first : first + batch]
        states = []
        for _ in range(steps):
            state = step(model, state, cc, norm)
            states.append(state)
        out[first : first + batch] = torch.stack(states, dim=1).cpu().numpy()
    return out


def growth_slope(median_sup: np.ndarray, first: int = 10) -> float:
    """Least-squares slope of log median sup error against log t over t in [first, T] (WP-D)."""
    t = np.arange(1, median_sup.size + 1, dtype=np.float64)
    sel = (t >= first) & np.isfinite(median_sup) & (median_sup > 0)
    if sel.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(t[sel]), np.log(median_sup[sel]), 1)[0])


def wang_labels(u_fit: torch.Tensor, seed: int, state_sd: float) -> torch.Tensor:
    """B15 (E1-Wang): a fixed label-perturbed copy of D_fit, u + N(0, (0.05 state_sd)^2), drawn once
    from a CPU generator so it is identical on every device; inputs stay unperturbed."""
    gen = torch.Generator().manual_seed(seed * 100_003 + 15)
    noise = torch.randn(u_fit.shape, generator=gen, dtype=torch.float32)
    return u_fit + (WANG_SIGMA * state_sd) * noise.to(u_fit.device)
