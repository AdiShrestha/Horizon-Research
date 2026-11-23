"""Sparse CPU linear program for fitting inverse time envelopes."""

import numpy as np
from jaxtyping import Float64
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


# Ruff treats jaxtyping's shape strings as Python forward references.
def lp_time_weights(
    r: Float64[np.ndarray, "n t"],  # noqa: F722
    a_min: float = 1e-6,
) -> Float64[np.ndarray, "t"]:  # noqa: F821
    """Fit positive coefficients b for max_t b[t] * r[i, t].

    Fit on weight-fitting trajectories. The division-based score and the
    band use the reciprocal envelope 1 / b.
    """
    if not isinstance(r, np.ndarray) or r.dtype != np.float64:
        raise TypeError("expected float64 NumPy step statistics")
    if r.ndim != 2 or r.size == 0 or not np.all(np.isfinite(r)) or np.any(r < 0):
        raise ValueError("r must be a nonempty (n, T) array of finite nonnegative statistics")
    if not 0 < a_min <= 1:
        raise ValueError("a_min must be in (0, 1]")

    n, horizon = r.shape
    rows = np.arange(n * horizon)
    time_columns = np.tile(np.arange(horizon), n)
    slack_columns = horizon + np.repeat(np.arange(n), horizon)
    # Each row represents b[t] * r[i, t] - R[i] <= 0.
    constraints = coo_matrix(
        (
            np.concatenate((r.ravel(), -np.ones(rows.size))),
            (np.tile(rows, 2), np.concatenate((time_columns, slack_columns))),
        ),
        shape=(rows.size, horizon + n),
    ).tocsr()
    objective = np.concatenate((np.zeros(horizon), np.ones(n)))
    equality = np.concatenate((np.ones(horizon), np.zeros(n)))[None, :]
    bounds = [(a_min / horizon, None)] * horizon + [(0.0, None)] * n
    result = linprog(
        objective,
        A_ub=constraints,
        b_ub=np.zeros(rows.size),
        A_eq=equality,
        b_eq=[1.0],
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Time-envelope LP failed: {result.message}")
    return np.asarray(result.x[:horizon], dtype=np.float64)
