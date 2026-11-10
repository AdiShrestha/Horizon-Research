"""Conformal calibration core: float64 NumPy arrays on the CPU only.

One trajectory is one exchangeable unit. Every function here receives one score (or one
vector of per-step statistics) per trajectory and returns plain float64 results. Statistics
produced on the GPU in float32 are converted exactly to float64 on entry.

Non-finite values. A trajectory whose rollout contains a non-finite value is
never dropped: NaN in a score or statistic is read as +inf, so the trajectory is uncovered by
every finite threshold and still counts in every denominator. Weights that are non-finite or
not positive become +inf (an infinite band at that cell) before any ratio is formed.
Thresholds are +inf when the rank exceeds the number of calibration trajectories.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

FloatArray = npt.NDArray[np.float64]

__all__ = [
    "FloatArray",
    "HorizonThresholds",
    "calibrate",
    "conformal_rank",
    "covered",
    "crc_threshold",
    "ess_is_adequate",
    "excursion_fraction",
    "field_score",
    "horizon_meta_scores",
    "horizon_scales",
    "horizon_thresholds",
    "infinite_cell_fraction",
    "kish_ess",
    "lp_time_profile",
    "lp_time_weights",
    "nan_to_inf",
    "normalized_residuals",
    "prefix_scores",
    "prefix_thresholds",
    "sanitize_weights",
    "step_maxima",
    "step_quantiles",
    "sup_score_from_steps",
    "weighted_calibrate",
]


# --------------------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------------------


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
        raise ValueError(f"time profile must have shape ({steps},), got {profile.shape}")
    if not np.all(np.isfinite(profile) & (profile > 0.0)):
        raise ValueError("time profile must be finite and positive")
    return profile


# --------------------------------------------------------------------------------------
# Split-conformal threshold
# --------------------------------------------------------------------------------------


def conformal_rank(n: int, alpha: float) -> int:
    """k = ceil((n + 1)(1 - alpha)) without floating-point drift. k > n means q = +inf."""
    return math.ceil(round((n + 1) * (1.0 - alpha), 12))


def calibrate(scores: npt.ArrayLike, alpha: float) -> float:
    """Split-conformal threshold: the k-th smallest calibration score, +inf if k > n."""
    _check_alpha(alpha)
    s = nan_to_inf(scores)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("scores must be a non-empty vector with one score per trajectory")
    k = conformal_rank(s.size, alpha)
    if k > s.size:
        return math.inf
    s = np.sort(s, kind="stable")
    return float(s[k - 1])


def covered(scores: npt.ArrayLike, threshold: float) -> npt.NDArray[np.bool_]:
    """Coverage indicator of the score event {S <= q}. A +inf score is covered only by q = +inf."""
    return nan_to_inf(scores) <= threshold


# --------------------------------------------------------------------------------------
# Scores from per-step statistics
# --------------------------------------------------------------------------------------


def sup_score_from_steps(m: npt.ArrayLike, a: npt.ArrayLike | None = None) -> FloatArray:
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
    return np.array([calibrate(p[:, j], alpha) for j in range(p.shape[1])], dtype=np.float64)


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
        raise ValueError("weights must broadcast to the error array without enlarging it")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(e / weights, dtype=np.float64)
    r[np.isnan(r)] = np.inf
    return r


def _fraction_rank(gamma: float, size: int) -> int:
    """j = ceil((1 - gamma) * size): the j-th smallest value is the (1 - gamma)-quantile."""
    return math.ceil(round((1.0 - gamma) * size, 12))


def field_score(
    err: npt.ArrayLike, w: npt.ArrayLike, gamma: float = 0.0, per_step: bool = False
) -> FloatArray:
    """err, w: (n, T, *space). gamma = 0 gives the sup score; gamma > 0 gives the space-time
    fraction score (STF) or, with per_step=True, the per-step fraction score (PSF)."""
    if not 0.0 <= gamma < 1.0:
        raise ValueError(f"gamma must lie in [0, 1), got {gamma!r}")
    r = normalized_residuals(err, w)
    n, steps = r.shape[:2]
    if per_step:
        r = r.reshape(n, steps, -1)
        if gamma == 0.0:
            return np.asarray(r.max(axis=2).max(axis=1), dtype=np.float64)
        j = _fraction_rank(gamma, r.shape[2])
        return np.asarray(np.partition(r, j - 1, axis=2)[:, :, j - 1].max(axis=1), dtype=np.float64)
    r = r.reshape(n, -1)
    if gamma == 0.0:
        return np.asarray(r.max(axis=1), dtype=np.float64)
    j = _fraction_rank(gamma, r.shape[1])
    return np.asarray(np.partition(r, j - 1, axis=1)[:, j - 1], dtype=np.float64)


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


def weighted_calibrate(
    scores: npt.ArrayLike, lr_cal: npt.ArrayLike, lr_test: float, alpha: float
) -> float:
    """Covariate-shift weighted threshold (Tibshirani et al. 2019): the (1 - alpha)-quantile of
    sum_i p_i delta_{S_i} + p_test delta_{+inf}."""
    _check_alpha(alpha)
    s = nan_to_inf(scores)
    p = _f64(lr_cal)
    if s.ndim != 1 or s.size == 0 or p.shape != s.shape:
        raise ValueError("scores and likelihood ratios must be vectors of the same length")
    ratios_ok = bool(np.all(np.isfinite(p)) and np.all(p >= 0.0))
    if not (ratios_ok and math.isfinite(lr_test) and lr_test >= 0.0):
        raise ValueError("likelihood ratios must be finite and non-negative")
    total = float(p.sum()) + float(lr_test)
    if total <= 0.0:
        raise ValueError("likelihood ratios sum to zero")
    order = np.argsort(s, kind="stable")
    s, p = s[order], p[order]
    cdf = np.round(np.cumsum(p) / total, 12)
    idx = int(np.searchsorted(cdf, round(1.0 - alpha, 12), side="left"))
    return math.inf if idx >= s.size else float(s[idx])


def kish_ess(lr: npt.ArrayLike) -> float:
    """Kish effective sample size (sum r)^2 / sum r^2 of the likelihood ratios."""
    x = _f64(lr)
    if x.size == 0 or not np.all(np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("likelihood ratios must be a non-empty finite non-negative vector")
    squares = float(np.sum(x**2))
    return 0.0 if squares == 0.0 else float(np.sum(x) ** 2 / squares)


def ess_is_adequate(lr: npt.ArrayLike, alpha: float) -> bool:
    """The monitoring rule of Algorithm 4: flag the calibration when n_eff < 5 / alpha."""
    _check_alpha(alpha)
    return kish_ess(lr) >= 5.0 / alpha


# --------------------------------------------------------------------------------------
# Time-profile LP (Cleaveland et al., AAAI 2024)
# --------------------------------------------------------------------------------------


def lp_time_weights(r: npt.ArrayLike, a_min: float = 1e-6) -> FloatArray:
    """LP relaxation of Cleaveland et al. (AAAI 2024). r: (n1, T) nonnegative per-step
    statistics on D_fit. Returns a (T,) summing to 1; score S_i = max_t a_t r_it and the
    band radius at step t is q / a_t."""
    stats = _f64(r)
    if stats.ndim != 2 or stats.size == 0:
        raise ValueError("per-step statistics must have shape (n1, T)")
    if not np.all(np.isfinite(stats)) or np.any(stats < 0.0):
        raise ValueError("the LP needs finite non-negative statistics (see the D_fit rule)")
    n1, steps = stats.shape
    c = np.concatenate([np.zeros(steps), np.ones(n1)])
    i_idx, t_idx = np.meshgrid(np.arange(n1), np.arange(steps), indexing="ij")
    rows = (i_idx * steps + t_idx).ravel()
    data = np.concatenate([stats.ravel(), -np.ones(n1 * steps)])
    cols = np.concatenate([t_idx.ravel(), steps + i_idx.ravel()])
    a_ub = coo_matrix(
        (data, (np.concatenate([rows, rows]), cols)), shape=(n1 * steps, steps + n1)
    ).tocsr()
    a_eq = np.concatenate([np.ones(steps), np.zeros(n1)])[None, :]
    bounds = [(a_min / steps, None)] * steps + [(0.0, None)] * n1
    res = linprog(
        c,
        A_ub=a_ub,
        b_ub=np.zeros(n1 * steps),
        A_eq=a_eq,
        b_eq=[1.0],
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(str(res.message))
    return np.asarray(res.x[:steps], dtype=np.float64)


def lp_time_profile(r: npt.ArrayLike, a_min: float = 1e-6) -> FloatArray:
    """Band profile b_t = 1 / a_t of the LP weights, for ``sup_score_from_steps`` and
    ``prefix_scores``: the score is max_t m_it / b_t = max_t a_t m_it and the band radius at
    step t is q * b_t. ``lp_time_weights`` returns the score multipliers a_t instead."""
    return np.asarray(1.0 / lp_time_weights(r, a_min), dtype=np.float64)


# --------------------------------------------------------------------------------------
# Conformal risk control (Angelopoulos et al., ICLR 2024)
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
        raise ValueError(f"horizons must be increasing integers in [1, {steps}], got {grid}")
    return [h - 1 for h in grid]


def horizon_scales(prefix_fit: npt.ArrayLike, horizons: Sequence[int], alpha: float) -> FloatArray:
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
        raise ValueError("horizon scales must be positive, one per horizon (all-zero fit scores?)")
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
