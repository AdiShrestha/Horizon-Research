"""CPU conformal calibration and one-score-per-trajectory reductions."""

from __future__ import annotations

import math
from fractions import Fraction
from operator import index

import numpy as np
from jaxtyping import Float64
from numpy.typing import NDArray

from .lp import lp_time_weights as lp_time_weights

# Ruff treats jaxtyping's shape strings as Python forward references.


def _probability(value: float, name: str) -> Fraction:
    if not 0 <= value < 1 or (name == "alpha" and value == 0):
        raise ValueError(f"{name} must be in {'(0, 1)' if name == 'alpha' else '[0, 1)'}")
    # Interpret the supplied decimal level exactly at integer rank boundaries.
    return Fraction(str(value))


def _nonnegative(array: NDArray[np.float64]) -> NDArray[np.float64]:
    if not isinstance(array, np.ndarray) or array.dtype != np.float64:
        raise TypeError("expected a float64 NumPy array")
    if np.any(np.isnan(array)) or np.any(array < 0):
        raise ValueError("expected nonnegative values without NaNs")
    return array


def _positive_weights(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    weights = _nonnegative(weights)
    if not np.all(np.isfinite(weights)) or np.any(weights == 0):
        raise ValueError("weights must be finite and strictly positive")
    return weights


def conformal_rank(n: int, alpha: float) -> int:
    """Return the one-based split-conformal rank, including the infinity slot."""
    n = index(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    return math.ceil((n + 1) * (1 - _probability(alpha, "alpha")))


def calibrate(scores: Float64[np.ndarray, "n"], alpha: float) -> float:  # noqa: F821
    """Return the k-th smallest score, with one extra score at infinity."""
    scores = _nonnegative(scores)
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    k = conformal_rank(scores.size, alpha)
    if k > scores.size:
        return math.inf
    return float(np.partition(scores, k - 1)[k - 1])


def sup_score_from_steps(
    m: Float64[np.ndarray, "n t"],  # noqa: F722
    a: Float64[np.ndarray, "t"] | None = None,  # noqa: F821
) -> Float64[np.ndarray, "n"]:  # noqa: F821
    """Reduce time-wise statistics by dividing by the optional envelope a."""
    m = _nonnegative(m)
    if m.ndim != 2 or m.shape[1] == 0:
        raise ValueError("step statistics must have shape (n, T) with T > 0")
    if a is not None:
        a = _positive_weights(a)
        if a.shape != (m.shape[1],):
            raise ValueError("the time envelope must have shape (T,)")
        m = m / a
    return m.max(axis=1)


def field_score(
    err: Float64[np.ndarray, "n t *space"],  # noqa: F722
    w: Float64[np.ndarray, "..."],
    gamma: float = 0.0,
    per_step: bool = False,
) -> Float64[np.ndarray, "n"]:  # noqa: F821
    """Compute SUP, space–time fraction (STF), or per-step fraction (PSF).

    Weights must broadcast to the error shape. For a shared time envelope
    over a spatial field, give it singleton spatial axes.
    """
    miss_fraction = _probability(gamma, "gamma")
    if not isinstance(err, np.ndarray) or err.dtype != np.float64:
        raise TypeError("expected a float64 NumPy error array")
    if err.ndim < 2 or 0 in err.shape[1:] or np.any(np.isnan(err)):
        raise ValueError("errors need nonempty time and space axes and no NaNs")
    normalized = np.abs(err) / np.broadcast_to(_positive_weights(w), err.shape)
    if miss_fraction == 0:
        return normalized.max(axis=tuple(range(1, err.ndim)))
    if per_step:
        normalized = normalized.reshape(*err.shape[:2], math.prod(err.shape[2:]))
    else:
        normalized = normalized.reshape(err.shape[0], math.prod(err.shape[1:]))
    j = math.ceil((1 - miss_fraction) * normalized.shape[-1]) - 1
    quantiles = np.partition(normalized, j, axis=-1)[..., j]
    return quantiles.max(axis=1) if per_step else quantiles


def band_radius(q: float, w: Float64[np.ndarray, "..."]) -> Float64[np.ndarray, "..."]:
    """Return the half-width of the band for threshold q and envelope w."""
    if not q >= 0:
        raise ValueError("q must be nonnegative")
    return np.asarray(q * _positive_weights(w), dtype=np.float64)


def weighted_calibrate(
    scores: Float64[np.ndarray, "n"],  # noqa: F821
    lr_cal: Float64[np.ndarray, "n"],  # noqa: F821
    lr_test: float,
    alpha: float,
) -> float:
    """Return the weighted quantile with the test ratio assigned to infinity."""
    level = 1 - _probability(alpha, "alpha")
    scores = _nonnegative(scores)
    lr_cal = _nonnegative(lr_cal)
    if scores.ndim != 1 or scores.shape != lr_cal.shape:
        raise ValueError("scores and calibration ratios need matching vector shapes")
    lr_test = float(lr_test)
    if not np.all(np.isfinite(lr_cal)) or not math.isfinite(lr_test) or lr_test < 0:
        raise ValueError("likelihood ratios must be finite and nonnegative")

    order = np.argsort(scores, kind="stable")
    order = order[lr_cal[order] > 0]
    if order.size == 0:
        if lr_test == 0:
            raise ValueError("total likelihood-ratio mass must be positive")
        return math.inf
    ratios = lr_cal[order]
    scale = max(float(ratios.max()), lr_test)
    cumulative = np.cumsum(ratios / scale)
    cdf = cumulative / (cumulative[-1] + lr_test / scale)
    cutoff = float(level)
    idx = int(np.searchsorted(cdf, cutoff, side="left"))
    # Resolve comparisons within a bound on float64 summation error exactly.
    tolerance = 4 * (ratios.size + 2) * np.finfo(np.float64).eps
    neighbors = cdf[max(0, idx - 1):min(cdf.size, idx + 1)]
    if np.any(np.abs(neighbors - cutoff) <= tolerance):
        masses = [Fraction(float(ratio)) for ratio in ratios]
        target = level * (sum(masses, Fraction(0)) + Fraction(lr_test))
        cumulative_mass = Fraction(0)
        for score_index, mass in zip(order, masses):
            cumulative_mass += mass
            if cumulative_mass >= target:
                return float(scores[score_index])
        return math.inf
    return math.inf if idx == order.size else float(scores[order[idx]])


def kish_ess(lr: Float64[np.ndarray, "n"]) -> float:  # noqa: F821
    """Return the Kish effective sample size of the calibration ratios."""
    lr = _nonnegative(lr)
    if lr.ndim != 1 or not np.all(np.isfinite(lr)):
        raise ValueError("likelihood ratios must be a finite vector")
    scale = float(lr.max(initial=0))
    if scale == 0:
        raise ValueError("total likelihood-ratio mass must be positive")
    scaled = lr / scale
    return float(scaled.sum() ** 2 / np.dot(scaled, scaled))
