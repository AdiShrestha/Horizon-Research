"""Streaming discrepancy reductions for the finite §7.5 audit cohort."""

import numpy as np


def location(shape, flat):
    step, cell = map(int, np.unravel_index(flat, shape))
    return {"macro_step": step + 1, "cell": cell}


class WeightedAudit:
    """Track matched-row maxima and a pointwise cross-trajectory conservative maximum."""

    def __init__(self):
        self.rows = []
        self._max_error = self._min_weight = None

    def update(self, ids, discrepancy, weights):
        """Discrepancy and weights have shape (batch, positive macro-step, projected cell)."""
        ids = np.asarray(ids, dtype=np.int64)
        error = np.asarray(discrepancy, dtype=np.float64)
        weight = np.broadcast_to(weights, error.shape)
        if (error.ndim != 3 or error.size == 0 or ids.shape != (len(error),)
                or not np.isfinite(error).all() or np.any(error < 0)
                or not np.isfinite(weight).all() or np.any(weight <= 0)):
            raise ValueError("need finite nonnegative discrepancies and finite positive weights")
        expected = None if self._max_error is None else self._max_error.shape
        if expected is not None and error.shape[1:] != expected:
            raise ValueError("macro-step and cell dimensions must stay fixed")
        n, *shape = error.shape
        lo, hi = np.argmin(weight, axis=0, keepdims=True), np.argmax(error, axis=0, keepdims=True)
        current_min = np.take_along_axis(weight, lo, axis=0)[0]
        current_max = np.take_along_axis(error, hi, axis=0)[0]
        if expected is None:
            self._min_weight = np.full(shape, np.inf)
            self._max_error = np.full(shape, -np.inf)
            self._min_id = np.zeros(shape, dtype=np.int64)
            self._max_id = np.zeros(shape, dtype=np.int64)
        lower = current_min < self._min_weight
        higher = current_max > self._max_error
        self._min_id[lower], self._max_id[higher] = ids[lo[0]][lower], ids[hi[0]][higher]
        np.minimum(self._min_weight, current_min, out=self._min_weight)
        np.maximum(self._max_error, current_max, out=self._max_error)
        normalized = (error / weight).reshape(n, -1)
        for i, peak in enumerate(np.argmax(normalized, axis=1)):
            place = np.unravel_index(peak, shape)
            self.rows.append({
                "id": int(ids[i]), **location(shape, peak), "epsilon": float(normalized[i, peak]),
                "discrepancy": float(error[i][place]), "weight": float(weight[i][place]),
            })

    def report(self, threshold):
        if not self.rows or not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("the audit requires trajectories and a finite positive calibration threshold")
        values = self._max_error / self._min_weight
        peak = np.argmax(values)
        at = np.unravel_index(peak, values.shape)
        epsilon = float(values[at])
        trajectories = [dict(row, epsilon_over_q=row["epsilon"] / threshold) for row in self.rows]
        worst = max(trajectories, key=lambda row: row["epsilon"])
        if epsilon < worst["epsilon"]:
            raise AssertionError("the conservative audit maximum must dominate the matched-row maximum")
        return {
            "threshold": threshold,
            "matched_maximum": worst,
            "trajectories": trajectories,
            "conservative_maximum": {
                **location(values.shape, peak), "epsilon": epsilon, "epsilon_over_q": epsilon / threshold,
                "discrepancy": float(self._max_error[at]), "weight": float(self._min_weight[at]),
                "discrepancy_id": int(self._max_id[at]), "weight_id": int(self._min_id[at]),
            },
            "passes_strict_1e_3": epsilon / threshold < 1e-3,
        }
