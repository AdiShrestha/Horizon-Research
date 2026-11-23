# Ported from the parallel Horizon workspace, commit a1500f9 (conformal/core.py), as additions.
"""Additions to the frozen conformal core (IC-02 WP-F step 1, amended by the WP-A disposition).

``conformal/core.py`` is frozen (engineering evidence hashes it), so the functions of the parallel
workspace's core that the production pipeline needs live here and import the frozen ``calibrate``.
Float64 on the CPU throughout; non-finite values follow the rule: NaN scores and residuals become +inf,
non-finite or non-positive weights give +inf at that cell, nothing is dropped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import numpy.typing as npt

from .core import calibrate as _calibrate

FloatArray = npt.NDArray[np.float64]

__all__ = [
    "HorizonThresholds",
    "calibrate",
    "crc_threshold",
    "excursion_fraction",
    "horizon_meta_scores",
    "horizon_scales",
    "horizon_thresholds",
    "infinite_cell_fraction",
    "nan_to_inf",
    "normalized_residuals",
    "prefix_scores",
    "prefix_thresholds",
    "sanitize_weights",
    "step_maxima",
    "sup_score_from_steps",
    "step_quantiles",
]


def calibrate(scores: npt.ArrayLike, alpha: float) -> float:
    """Frozen split-conformal threshold applied after the NaN -> +inf rule."""
    s = nan_to_inf(scores)
    if s.ndim != 1 or s.size == 0:
        raise ValueError(
            "scores must be a non-empty vector with one score per trajectory"
        )
    return _calibrate(s, alpha)


def _f64(x: npt.ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def nan_to_inf(x: npt.ArrayLike) -> FloatArray:
    """Copy of ``x`` as float64 with NaN replaced by +inf (failed rollouts keep their place)."""
    out = np.array(x, dtype=np.float64, copy=True)
    out[np.isnan(out)] = np.inf
    return out


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")


def _exact_alpha(alpha: float) -> Fraction:
    """alpha as the exact rational of its shortest decimal representation (0.1 -> 1/10)."""
    return Fraction(repr(float(alpha)))


def _time_profile(a: npt.ArrayLike, steps: int) -> FloatArray:
    profile = _f64(a)
    if profile.shape != (steps,):
        raise ValueError(
            f"time profile must have shape ({steps},), got {profile.shape}"
        )
    if not np.all(np.isfinite(profile) & (profile > 0.0)):
        raise ValueError("time profile must be finite and positive")
    return profile


# --------------------------------------------------------------------------------------
# Split-conformal threshold
# --------------------------------------------------------------------------------------


def sup_score_from_steps(
    m: npt.ArrayLike, a: npt.ArrayLike | None = None
) -> FloatArray:
    """m: (n, T) per-step maxima of |e|/w; a: optional (T,) positive time profile (W4).

    Returns the sup score S_i = max_t m_it / a_t.
    """
    steps = nan_to_inf(m)
    if steps.ndim != 2:
        raise ValueError("per-step statistics must have shape (n, T)")
    if a is not None:
        steps = steps / _time_profile(a, steps.shape[1])[None, :]
    return np.asarray(steps.max(axis=1), dtype=np.float64)


def prefix_scores(m: npt.ArrayLike, a: npt.ArrayLike | None = None) -> FloatArray:
    """Prefix scores S^(T') = max_{t <= T'} m_it / a_t for every T' (column T'-1), shape (n, T).

    The columns are non-decreasing, so one threshold calibrated at T covers every prefix.
    """
    steps = nan_to_inf(m)
    if steps.ndim != 2:
        raise ValueError("per-step statistics must have shape (n, T)")
    if a is not None:
        steps = steps / _time_profile(a, steps.shape[1])[None, :]
    return np.asarray(np.maximum.accumulate(steps, axis=1), dtype=np.float64)


def prefix_thresholds(prefix: npt.ArrayLike, alpha: float) -> FloatArray:
    """Threshold recalibrated at every horizon: calibrate each column of the prefix scores."""
    p = nan_to_inf(prefix)
    if p.ndim != 2:
        raise ValueError("prefix scores must have shape (n, T)")
    return np.array(
        [calibrate(p[:, j], alpha) for j in range(p.shape[1])], dtype=np.float64
    )


# --------------------------------------------------------------------------------------
# Scores from full fields
# --------------------------------------------------------------------------------------


def sanitize_weights(w: npt.ArrayLike) -> FloatArray:
    """Weights that are non-finite or not positive become +inf (an infinite band there)."""
    out = np.array(w, dtype=np.float64, copy=True)
    out[~np.isfinite(out) | (out <= 0.0)] = np.inf
    return out


def infinite_cell_fraction(w: npt.ArrayLike) -> float:
    """Fraction of cells whose weight is +inf after sanitizing (reported per configuration)."""
    return float(np.mean(np.isposinf(sanitize_weights(w))))


def normalized_residuals(err: npt.ArrayLike, w: npt.ArrayLike) -> FloatArray:
    """r = |err| / w with the module's non-finite rules (NaN becomes +inf)."""
    e = np.abs(_f64(err))
    weights = sanitize_weights(w)
    if np.broadcast_shapes(e.shape, weights.shape) != e.shape:
        raise ValueError(
            "weights must broadcast to the error array without enlarging it"
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(e / weights, dtype=np.float64)
    r[np.isnan(r)] = np.inf
    return r


def _fraction_rank(gamma: float, size: int) -> int:
    """j = ceil((1 - gamma) * size): the j-th smallest value is the (1 - gamma)-quantile."""
    return math.ceil(round((1.0 - gamma) * size, 12))


def step_maxima(err: npt.ArrayLike, w: npt.ArrayLike) -> FloatArray:
    """Reference reduction m_it = max_x |e_it(x)| / w_it(x), shape (n, T)."""
    r = normalized_residuals(err, w)
    n, steps = r.shape[:2]
    return np.asarray(r.reshape(n, steps, -1).max(axis=2), dtype=np.float64)


def step_quantiles(err: npt.ArrayLike, w: npt.ArrayLike, gamma: float) -> FloatArray:
    """Reference reduction: the ceil((1 - gamma) d)-th smallest of r_it(.) at every step."""
    if not 0.0 <= gamma < 1.0:
        raise ValueError(f"gamma must lie in [0, 1), got {gamma!r}")
    r = normalized_residuals(err, w)
    n, steps = r.shape[:2]
    r = r.reshape(n, steps, -1)
    j = _fraction_rank(gamma, r.shape[2])
    return np.asarray(np.partition(r, j - 1, axis=2)[:, :, j - 1], dtype=np.float64)


# --------------------------------------------------------------------------------------
# Covariate shift
# --------------------------------------------------------------------------------------


def excursion_fraction(r: npt.ArrayLike, lam: float) -> FloatArray:
    """L_i(lam): fraction of cells of trajectory i with r > lam, shape (n,)."""
    x = nan_to_inf(r)
    flat = x.reshape(x.shape[0], -1)
    return np.asarray(np.mean(flat > lam, axis=1), dtype=np.float64)


def crc_threshold(r: npt.ArrayLike, alpha: float) -> float:
    """lambda_hat = inf{lambda >= 0 : (sum_i L_i(lambda) + 1) / (n + 1) <= alpha}.

    r: normalized residuals of the calibration trajectories, shape (n, ...). With M cells per
    trajectory the condition reads: at most B = ((n + 1) alpha - 1) M of the n M pooled values
    exceed lambda, so lambda_hat is the (floor(B) + 1)-th largest pooled value. The result is
    +inf when (n + 1) alpha < 1 (no lambda qualifies) and 0 when every lambda >= 0 qualifies.
    """
    _check_alpha(alpha)
    x = nan_to_inf(r)
    if x.ndim < 2 or x.size == 0:
        raise ValueError("residuals must have shape (n, ...) with at least one cell")
    n = x.shape[0]
    flat = x.reshape(n, -1)
    budget = (Fraction(n + 1) * _exact_alpha(alpha) - 1) * flat.shape[1]
    if budget < 0:
        return math.inf
    rank = math.floor(budget) + 1
    if rank > flat.size:
        return 0.0
    pooled = flat.ravel()
    position = pooled.size - rank
    return float(np.partition(pooled, position)[position])


# --------------------------------------------------------------------------------------
# Horizon-specific thresholds
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HorizonThresholds:
    """Thresholds q_{T'} = q_Z * gamma_{T'} on a horizon grid, simultaneously valid."""

    horizons: tuple[int, ...]
    scales: FloatArray
    meta_threshold: float
    thresholds: FloatArray


def _horizon_columns(horizons: Sequence[int], steps: int) -> list[int]:
    grid = [int(h) for h in horizons]
    if not grid or any(h < 1 or h > steps for h in grid) or sorted(set(grid)) != grid:
        raise ValueError(
            f"horizons must be increasing integers in [1, {steps}], got {grid}"
        )
    return [h - 1 for h in grid]


def horizon_scales(
    prefix_fit: npt.ArrayLike, horizons: Sequence[int], alpha: float
) -> FloatArray:
    """gamma_{T'}: the empirical (1 - alpha)-quantile of S^(T') on D_fit (prefix_fit: (n, T))."""
    _check_alpha(alpha)
    p = nan_to_inf(prefix_fit)
    if p.ndim != 2:
        raise ValueError("prefix scores must have shape (n, T)")
    cols = _horizon_columns(horizons, p.shape[1])
    scales = np.quantile(p[:, cols], 1.0 - alpha, axis=0, method="higher")
    return np.asarray(scales, dtype=np.float64)


def horizon_meta_scores(
    prefix: npt.ArrayLike, horizons: Sequence[int], scales: npt.ArrayLike
) -> FloatArray:
    """Z_i = max over the horizon grid of S_i^(T') / gamma_{T'}; +inf stays +inf."""
    p = nan_to_inf(prefix)
    if p.ndim != 2:
        raise ValueError("prefix scores must have shape (n, T)")
    cols = _horizon_columns(horizons, p.shape[1])
    g = _f64(scales)
    if g.shape != (len(cols),) or not np.all(g > 0.0):
        raise ValueError(
            "horizon scales must be positive, one per horizon (all-zero fit scores?)"
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.asarray(p[:, cols] / g[None, :], dtype=np.float64)
    ratio[np.isnan(ratio)] = np.inf
    return np.asarray(ratio.max(axis=1), dtype=np.float64)


def horizon_thresholds(
    prefix_fit: npt.ArrayLike,
    prefix_cal: npt.ArrayLike,
    horizons: Sequence[int],
    alpha: float,
) -> HorizonThresholds:
    """Fit gamma_{T'} on D_fit, calibrate the meta-score on D_cal, return q_{T'} = q_Z gamma_{T'}.

    A test trajectory with Z <= q_Z has S^(T') <= q_{T'} for every T' on the grid, so the grid is
    covered simultaneously at probability >= 1 - alpha with no union-bound loss.
    """
    scales = horizon_scales(prefix_fit, horizons, alpha)
    z = horizon_meta_scores(prefix_cal, horizons, scales)
    q_z = calibrate(z, alpha)
    with np.errstate(invalid="ignore"):
        thresholds = np.asarray(q_z * scales, dtype=np.float64)
    thresholds[np.isposinf(scales) | np.isposinf(q_z)] = np.inf
    return HorizonThresholds(
        horizons=tuple(int(h) for h in horizons),
        scales=scales,
        meta_threshold=q_z,
        thresholds=thresholds,
    )
