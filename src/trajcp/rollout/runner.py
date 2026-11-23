"""Stage 1 rollout runner (IC-02 WP-E): load an emulator, perturb initial states, roll out in batches.

Eager PyTorch float32 on a chosen device. Perturbations are seeded from
``component_key(trajectory_key, "perturbation")`` (A02), one sub-key per perturbation index so that
the first M of a larger set equal the smaller set. Non-finite rollouts are kept, never dropped: NaN
and inf propagate to the reductions, which apply the standing rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import torch
from safetensors.torch import load_file

from trajcp.data.seeds import component_key, trajectory_key
from trajcp.emulators.e1 import build_model
from trajcp.emulators.training import Normalization, predict_next


def load_emulator(run: Path, device: str, weights: str = "best.safetensors"):
    """Model (eager float32, eval mode) and normalization of a trained E1 run directory."""
    settings = json.loads((run / "settings.json").read_text())
    norm = Normalization(**settings["normalization"])
    model = build_model()
    model.load_state_dict(load_file(str(run / weights)))
    return model.to(device).eval(), norm, settings


def perturbation_fields(
    stream: str,
    task: str,
    indices: np.ndarray,
    count: int,
    points: int,
    *,
    kind: str = "lowmode",
    cutoff: int = 8,
) -> np.ndarray:
    """Unit-RMS perturbation fields, shape (len(indices), count, points), float64.

    ``lowmode``: Gaussian coefficients on cos/sin modes 1..cutoff; ``iso``: isotropic pixel noise.
    """
    x = np.arange(points) * (2.0 * np.pi / points)
    phases = np.arange(1, cutoff + 1)[:, None] * x
    basis = np.concatenate((np.cos(phases), np.sin(phases)))
    out = np.empty((len(indices), count, points))
    width = 2 * cutoff if kind == "lowmode" else points
    for row, index in enumerate(indices):
        pkey = component_key(trajectory_key(stream, task, int(index)), "perturbation")
        draws = np.stack(
            [
                np.asarray(jax.random.normal(jax.random.fold_in(pkey, m), (width,)))
                for m in range(count)
            ]
        )
        xi = draws @ basis if kind == "lowmode" else draws
        out[row] = xi / np.sqrt(np.mean(xi**2, axis=-1, keepdims=True))
    return out


@torch.no_grad()
def rollout(
    model: torch.nn.Module,
    u0: np.ndarray,
    mu: np.ndarray,
    norm: Normalization,
    steps: int,
    device: str,
    batch: int = 256,
) -> np.ndarray:
    """Autoregressive rollout of states u0 (n, N) with parameter mu (n,): predictions for
    t = 1..steps, shape (n, steps, N) float32 on the CPU. NaN/inf are kept."""
    out = np.empty((u0.shape[0], steps, u0.shape[1]), dtype=np.float32)
    for first in range(0, u0.shape[0], batch):
        sl = slice(first, first + batch)
        state = torch.tensor(u0[sl], dtype=torch.float32, device=device)
        m = torch.tensor(mu[sl], dtype=torch.float32, device=device)
        states = []
        for _ in range(steps):
            state = predict_next(model, state, m, norm)
            states.append(state)
        out[sl] = torch.stack(states, dim=1).cpu().numpy()
    return out


def perturbed_initials(u0: np.ndarray, xi: np.ndarray, epsilon: float) -> np.ndarray:
    """u0 (n, N), xi (n, M, N) unit RMS -> (n, M, N) float32 with delta = epsilon * rms(u0) * xi."""
    rms = np.sqrt(np.mean(u0.astype(np.float64) ** 2, axis=-1))[:, None, None]
    return (u0[:, None, :].astype(np.float64) + epsilon * rms * xi).astype(np.float32)


def jvp_sensitivity(
    model: torch.nn.Module,
    u0: np.ndarray,
    mu: np.ndarray,
    xi: np.ndarray,
    norm: Normalization,
    steps: int,
    device: str,
) -> np.ndarray:
    """W3-JVP: raw sensitivity s_t(x) = sqrt(mean_m (J_t v_m)^2) with tangents v_m = rms(u0) xi_m,
    by forward-mode ``torch.func.jvp`` through the whole rollout. Shape (n, steps, N) float64."""
    rms = np.sqrt(np.mean(u0.astype(np.float64) ** 2, axis=-1))
    u = torch.tensor(u0, dtype=torch.float32, device=device)
    m = torch.tensor(mu, dtype=torch.float32, device=device)

    def roll(state: torch.Tensor) -> torch.Tensor:
        states = []
        for _ in range(steps):
            state = predict_next(model, state, m, norm)
            states.append(state)
        return torch.stack(states, dim=1)

    acc = np.zeros((u0.shape[0], steps, u0.shape[1]))
    for k in range(xi.shape[1]):
        tangent = torch.tensor(
            rms[:, None] * xi[:, k], dtype=torch.float32, device=device
        )
        _, jv = torch.func.jvp(roll, (u,), (tangent,))
        acc += jv.detach().double().cpu().numpy() ** 2
    return np.sqrt(acc / xi.shape[1])
