"""Stage 1 reductions (IC-02 WP-E, brief §12.3): sufficient statistics from error and weight fields.

All arithmetic in float64 on the CPU; outputs cast to float32 for storage. Non-finite rule: NaN
errors give +inf residuals; weights that are non-finite or not positive give +inf at that cell
(an infinite band), so a trajectory is never dropped. L2 norms are grid-mean (quadrature) norms.
"""

from __future__ import annotations

import math

import numpy as np

from trajcp.conformal.extended import normalized_residuals
from trajcp.rollout.sensitivity import intrinsic_weights

GAMMAS = (0.001, 0.005, 0.01, 0.05)


def l2(x: np.ndarray) -> np.ndarray:
    """Quadrature L2 norm over the last axis: sqrt(mean x^2)."""
    with np.errstate(invalid="ignore", over="ignore"):
        return np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2, axis=-1))


def raw_sensitivity_nonfinite(
    perturbed: np.ndarray, base: np.ndarray, epsilon: float
) -> np.ndarray:
    """s_t(x) = sqrt(mean_m (u^(m)_t - u_t)^2) / epsilon for perturbed (n, M, T, N), base (n, T, N).
    Non-finite entries stay non-finite (handled in ``w3_field``)."""
    diff = perturbed.astype(np.float64) - base.astype(np.float64)[:, None]
    with np.errstate(invalid="ignore", over="ignore"):
        return np.sqrt(np.mean(diff**2, axis=1)) / epsilon


def w3_field(
    raw: np.ndarray, smoothing: int, eta: float, absolute_floor: float
) -> np.ndarray:
    """Smooth, floor and add the absolute floor (frozen ``intrinsic_weights``); a step whose raw
    sensitivity has any non-finite value gets +inf at every cell."""
    finite = np.isfinite(raw).all(axis=-1)  # (n, T)
    safe = np.where(finite[..., None], raw, 0.0)
    w = intrinsic_weights(safe, smoothing, eta, absolute_floor)
    w[~finite] = np.inf
    return w


def reduce_weight(err: np.ndarray, w: np.ndarray | None) -> dict[str, np.ndarray]:
    """Per-step statistics (n, T) float32: m, p{gamma}, wbar, wl2 for the weight field w (None = 1)."""
    err = np.asarray(err, dtype=np.float64)
    weights = np.ones_like(err) if w is None else np.asarray(w, dtype=np.float64)
    r = normalized_residuals(err, weights)
    n_cells = r.shape[-1]
    out = {"m": r.max(axis=-1)}
    for gamma in GAMMAS:
        j = math.ceil(round((1.0 - gamma) * n_cells, 12))
        out[f"p{gamma:g}"] = np.partition(r, j - 1, axis=-1)[..., j - 1]
    finite_w = np.where(np.isfinite(weights) & (weights > 0.0), weights, np.inf)
    out["wbar"] = finite_w.mean(axis=-1)
    out["wl2"] = l2(finite_w)
    return {k: v.astype(np.float32) for k, v in out.items()}


def base_statistics(truth: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    """M_{i,t} = ||e||_inf, ||u||_L2 and ||e||_L2, shapes (n, T) float32 (NaN errors give +inf)."""
    err = pred.astype(np.float64) - truth.astype(np.float64)
    with np.errstate(invalid="ignore"):
        m = np.abs(err).max(axis=-1)
    m[np.isnan(m)] = np.inf
    e2 = l2(err)
    e2[np.isnan(e2)] = np.inf
    return {
        "M": m.astype(np.float32),
        "u_l2": l2(truth).astype(np.float32),
        "e_l2": e2.astype(np.float32),
    }
