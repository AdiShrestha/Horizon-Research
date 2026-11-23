"""Fit and select W4 time profiles on weight-fitting trajectories only."""
import math

import numpy as np
from scipy.optimize import isotonic_regression, least_squares

from trajcp.conformal.core import calibrate, conformal_rank, sup_score_from_steps
from trajcp.conformal.lp import lp_time_weights

BETAS = (.50, .80, .90, .95, .99)
SETTINGS = tuple((s, eta) for s in range(4) for eta in (.02, .05, .10))


def step_maxima(error, w3):
    """Compute the float64 SUP sufficient statistics for a field envelope."""
    if error.shape != w3.shape or error.ndim != 3 or np.any(w3 <= 0):
        raise ValueError("expected aligned (n, T, x) error and positive W3 weights")
    return np.max(np.abs(error) / w3, axis=-1)


def fit_profile(steps, method, *, beta=None, epsilon0, eta):
    steps = np.asarray(steps, dtype=np.float64)
    if steps.ndim != 2 or steps.size == 0 or not np.isfinite(steps).all() or np.any(steps < 0):
        raise ValueError("expected nonnegative finite D_fit step maxima")
    if method == "lp":
        # The LP returns multipliers in the score, so its profile is exactly reciprocal.
        return 1. / lp_time_weights(steps)
    if method == "q":
        if beta not in BETAS:
            raise ValueError("unknown W4-q beta")
        profile = isotonic_regression(np.quantile(steps, 1 - beta, axis=0)).x
    elif method == "par":
        t = np.arange(1, steps.shape[1] + 1) / steps.shape[1]
        target = np.log(np.maximum(np.median(steps, axis=0), epsilon0))
        # Fit the median step maxima in log space. Scaling t by T rescales c1 only.
        def residual(params):
            log_c0, c1, c2 = params
            return log_c0 + c2 * np.log1p(c1 * t) - target
        result = least_squares(residual, [np.mean(target), 1., 1.],
                               bounds=([-700., 0., 0.], [700., 1000., 30.]))
        if not result.success:
            raise RuntimeError("W4 parametric fit failed")
        profile = np.exp(residual(result.x) + target)
    else:
        raise ValueError("unknown W4 profile family")
    return np.maximum(profile, max(epsilon0, eta * profile[0]))


def select_profile(steps, relative_w3, folds, *, alpha, epsilon0, eta):
    """Compare mean full relative width after pseudo-calibrating each held-out D_fit fold."""
    a, b = (np.asarray(fold) for fold in folds)
    if (a.ndim != 1 or b.ndim != 1 or a.dtype.kind not in "iu" or b.dtype.kind not in "iu"
            or not np.array_equal(np.sort(np.concatenate((a, b))), np.arange(len(steps)))):
        raise ValueError("two folds must partition D_fit exactly")
    if any(conformal_rank(len(fold), alpha) > len(fold) for fold in (a, b)):
        raise ValueError("both D_fit folds must admit finite pseudo-calibration ranks")
    if (steps.shape != relative_w3.shape or not np.isfinite(relative_w3).all()
            or np.any(relative_w3 <= 0)):
        raise ValueError("expected positive W3 relative-width coefficients on D_fit")
    candidates = [("q", beta) for beta in BETAS] + [("lp", None), ("par", None)]
    results = []
    for method, beta in candidates:
        total = []
        for fitting, pseudo in ((a, b), (b, a)):
            profile = fit_profile(steps[fitting], method, beta=beta, epsilon0=epsilon0, eta=eta)
            q = calibrate(sup_score_from_steps(steps[pseudo], profile), alpha)
            total.extend(np.mean(2 * q * relative_w3[pseudo] * profile, axis=1).tolist())
        results.append({"method": method, "beta": beta, "mean_relative_width": math.fsum(total) / len(steps)})
    chosen = min(results, key=lambda row: row["mean_relative_width"])
    full = fit_profile(steps, chosen["method"], beta=chosen["beta"], epsilon0=epsilon0, eta=eta)
    return full, chosen, results
