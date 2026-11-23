"""Study-side §10.1 trajectory-band coverage and relative widths."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from trajcp.conformal.core import band_radius, calibrate, conformal_rank, field_score

FloatArray = NDArray[np.float64]
TrajectoryID = str | int


@dataclass(frozen=True)
class TrajectoryBatch:
    """Aligned physical-unit truth and forecast, with shape (N, T, *field)."""

    ids: Sequence[TrajectoryID]
    truth: FloatArray
    prediction: FloatArray


@dataclass(frozen=True)
class BandMetrics:
    """Held-out metrics with per-trajectory rows aligned to test_ids."""

    test_ids: tuple[TrajectoryID, ...]
    covered: NDArray[np.bool_]
    relative_width_by_step: FloatArray
    simultaneous_coverage: float
    mean_relative_width: float
    relative_width_at_quarter: float | None
    relative_width_at_end: float


@dataclass(frozen=True)
class SplitBandEvaluation:
    calibration_ids: tuple[TrajectoryID, ...]
    calibration_scores: FloatArray
    rank: int
    threshold: float
    metrics: BandMetrics


def _validate_batch(batch: TrajectoryBatch, *, allow_empty: bool = False) -> tuple[TrajectoryID, ...]:
    ids = tuple(batch.ids)
    for field in (batch.truth, batch.prediction):
        if not isinstance(field, np.ndarray) or field.dtype != np.float64:
            raise TypeError("truth and prediction must be float64 NumPy arrays")
        if not np.isfinite(field).all():
            raise ValueError("truth and prediction must be finite")
    shape = batch.truth.shape
    if len(shape) < 3 or 0 in shape[1:] or batch.prediction.shape != shape:
        raise ValueError("truth and prediction need matching (N, T, *field) shapes")
    if shape[0] != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("trajectory IDs must be unique and match the number of rows")
    if not ids and not allow_empty:
        raise ValueError("the evaluation batch must contain at least one trajectory")
    return ids


def _band_metrics(
    batch: TrajectoryBatch, ids: tuple[TrajectoryID, ...], threshold: float,
    quadrature: FloatArray, weights: FloatArray,
) -> BandMetrics:
    if not isinstance(quadrature, np.ndarray) or quadrature.dtype != np.float64:
        raise TypeError("quadrature must be a float64 NumPy array")
    if not np.isfinite(quadrature).all() or np.any(quadrature <= 0):
        raise ValueError("quadrature weights must be finite and strictly positive")
    sqrt_quadrature = np.sqrt(np.broadcast_to(quadrature, batch.truth.shape[2:]))
    shape = batch.truth.shape
    radius = np.broadcast_to(band_radius(threshold, weights), shape)
    with np.errstate(over="ignore"):
        lower, upper = batch.prediction - radius, batch.prediction + radius
        covered = ((lower <= batch.truth) & (batch.truth <= upper)).reshape(shape[0], -1).all(axis=1)
        width = 2.0 * radius
        truth_norm = np.hypot.reduce((batch.truth * sqrt_quadrature).reshape(*shape[:2], -1), axis=-1)
        width_norm = np.hypot.reduce((width * sqrt_quadrature).reshape(*shape[:2], -1), axis=-1)
        relative = np.divide(
            width_norm, truth_norm, out=np.full_like(truth_norm, np.nan), where=truth_norm > 0,
        )
    horizon = shape[1]
    return BandMetrics(
        ids, covered, relative, float(covered.mean()), float(relative.mean()),
        float(relative[:, horizon // 4 - 1].mean()) if horizon % 4 == 0 else None,
        float(relative[:, -1].mean()),
    )


def evaluate_band(
    test: TrajectoryBatch, threshold: float, quadrature: FloatArray, weights: FloatArray | None = None,
) -> BandMetrics:
    """Evaluate a frozen symmetric band; weights broadcast to the rollout shape.

    Quadrature broadcasts to the field axes, for example spatial weights
    shared across channels. A zero reference norm makes the corresponding
    relative width and any aggregate containing it NaN. Coverage still uses
    every trajectory. The quarter-step metric is None if T/4 is off-grid.
    """
    ids = _validate_batch(test)
    return _band_metrics(test, ids, threshold, quadrature, np.array(1.0) if weights is None else weights)


def evaluate_split_band(
    calibration: TrajectoryBatch, test: TrajectoryBatch, alpha: float, quadrature: FloatArray, *,
    calibration_weights: FloatArray | None = None, test_weights: FloatArray | None = None,
) -> SplitBandEvaluation:
    """Calibrate one SUP score per trajectory and evaluate the held-out band.

    Omit both envelopes for B01. For W4, supply already frozen positive
    envelopes on each split. A shared time profile of shape (T, 1) suffices
    for a scalar field with one spatial axis. Calibration never reads test
    truth or fits weights.
    """
    calibration_ids = _validate_batch(calibration, allow_empty=True)
    if (calibration_weights is None) != (test_weights is None):
        raise ValueError("supply both calibration and test weights or omit both")
    with np.errstate(over="ignore"):
        scores = field_score(
            calibration.truth - calibration.prediction,
            np.array(1.0) if calibration_weights is None else calibration_weights,
        )
    threshold = calibrate(scores, alpha)
    test_ids = _validate_batch(test)
    if set(calibration_ids).intersection(test_ids):
        raise ValueError("calibration and test trajectory IDs must be disjoint")
    if calibration.truth.shape[1:] != test.truth.shape[1:]:
        raise ValueError("calibration and test need the same time and field axes")
    metrics = _band_metrics(
        test, test_ids, threshold, quadrature, np.array(1.0) if test_weights is None else test_weights,
    )
    return SplitBandEvaluation(
        calibration_ids, scores, conformal_rank(len(calibration_ids), alpha), threshold, metrics,
    )
