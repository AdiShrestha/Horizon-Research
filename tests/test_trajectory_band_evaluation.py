import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from trajcp.evaluation.metrics import TrajectoryBatch, evaluate_band, evaluate_split_band


@pytest.fixture
def fixture():
    cal_truth = np.full((3, 4, 2), 8.0)
    cal_error = np.zeros_like(cal_truth)
    cal_error[0, 0, 0] = 1
    cal_error[1, 3, 1] = -3
    cal_error[2, 2, 0] = 4
    calibration = TrajectoryBatch(("cal-a", "cal-b", "cal-c"), cal_truth, cal_truth - cal_error)

    truth = np.repeat(np.array([[4, 8, 8, 16], [8, 8, 16, 16]], dtype=float)[..., None], 2, axis=2)
    error = np.zeros_like(truth)
    error[0, 0] = [2, -2]  # Upper and lower W4 endpoints.
    error[:, 3] = [4, -4]  # Upper and lower B01 endpoints.
    error[1, 0, 0] = 3     # The only W4 miss.
    test = TrajectoryBatch(("test-a", "test-b"), truth, truth - error)
    return calibration, test, np.array([1.0, 3.0]), np.array([[1.0], [1.0], [2.0], [4.0]])


def test_calibration_to_held_out_b01_and_frozen_w4(fixture):
    calibration, test, quadrature, w4 = fixture
    b01 = evaluate_split_band(calibration, test, 0.25, quadrature)
    adaptive = evaluate_split_band(
        calibration, test, 0.25, quadrature, calibration_weights=w4, test_weights=w4,
    )
    assert b01.calibration_ids == adaptive.calibration_ids == calibration.ids
    assert b01.rank == adaptive.rank == 3
    assert_array_equal(b01.calibration_scores, [1, 3, 4])
    assert_array_equal(adaptive.calibration_scores, [1, 0.75, 2])
    assert b01.threshold == 4
    assert adaptive.threshold == 2

    uniform = b01.metrics
    weighted = adaptive.metrics
    assert uniform.test_ids == weighted.test_ids == test.ids
    assert_array_equal(uniform.covered, [True, True])
    assert_array_equal(weighted.covered, [True, False])
    assert (uniform.simultaneous_coverage, weighted.simultaneous_coverage) == (1, 0.5)
    # Full widths are 8 for B01 and [4, 4, 8, 16] for W4.
    assert_array_equal(uniform.relative_width_by_step, [[2, 1, 1, 0.5], [1, 1, 0.5, 0.5]])
    assert_array_equal(weighted.relative_width_by_step, [[1, 0.5, 1, 1], [0.5, 0.5, 0.5, 1]])
    assert (uniform.mean_relative_width, weighted.mean_relative_width) == (0.9375, 0.75)
    assert (uniform.relative_width_at_quarter, weighted.relative_width_at_quarter) == (1.5, 0.75)
    assert (uniform.relative_width_at_end, weighted.relative_width_at_end) == (0.5, 1)


def test_nonuniform_quadrature_and_per_trajectory_envelopes():
    truth = np.broadcast_to(np.array([1.0, 2.0]), (2, 4, 2))
    test = TrajectoryBatch((1, 2), truth, truth)
    weights = np.array([[[2.0, 1.0]], [[2.0, 2.0]]])
    metrics = evaluate_band(test, 0.5, np.array([1.0, 3.0]), weights)
    expected = np.sqrt(np.array([7.0, 16.0]) / 13)
    assert_allclose(metrics.relative_width_by_step, np.broadcast_to(expected[:, None], (2, 4)))
    assert metrics.mean_relative_width == pytest.approx(expected.mean())
    assert metrics.relative_width_at_quarter == pytest.approx(expected.mean())
    assert metrics.relative_width_at_end == pytest.approx(expected.mean())


def test_batch_order_invariance_and_id_alignment(fixture):
    calibration, test, quadrature, w4 = fixture

    def reorder(batch, order):
        return TrajectoryBatch(tuple(batch.ids[i] for i in order), batch.truth[order], batch.prediction[order])

    first = evaluate_split_band(calibration, test, 0.25, quadrature, calibration_weights=w4, test_weights=w4)
    changed = evaluate_split_band(
        reorder(calibration, [2, 0, 1]), reorder(test, [1, 0]), 0.25, quadrature,
        calibration_weights=w4, test_weights=w4,
    )
    assert changed.rank == first.rank and changed.threshold == first.threshold
    assert dict(zip(first.calibration_ids, first.calibration_scores)) == dict(
        zip(changed.calibration_ids, changed.calibration_scores)
    )
    assert changed.metrics.test_ids == ("test-b", "test-a")
    assert_array_equal(changed.metrics.covered, first.metrics.covered[::-1])
    assert_array_equal(changed.metrics.relative_width_by_step, first.metrics.relative_width_by_step[::-1])
    for name in ("simultaneous_coverage", "mean_relative_width", "relative_width_at_quarter", "relative_width_at_end"):
        assert getattr(changed.metrics, name) == getattr(first.metrics, name)


def test_rank_past_calibration_set_means_full_coverage_and_unbounded_width(fixture):
    calibration, test, quadrature, w4 = fixture
    result = evaluate_split_band(
        calibration, test, 0.1, quadrature, calibration_weights=w4, test_weights=w4,
    )
    assert result.rank == 4 and result.threshold == math.inf
    assert result.metrics.simultaneous_coverage == 1
    assert np.isposinf(result.metrics.relative_width_by_step).all()
    assert result.metrics.mean_relative_width == math.inf
    assert result.metrics.relative_width_at_quarter == math.inf
    assert result.metrics.relative_width_at_end == math.inf


@pytest.mark.parametrize("threshold", [0.0, 1.0, math.inf])
def test_zero_truth_norm_is_undefined_without_losing_coverage(threshold):
    truth = np.array([[[0.0], [1.0], [2.0], [4.0]]])
    test = TrajectoryBatch(("a",), truth, truth)
    metrics = evaluate_band(test, threshold, np.array(1.0))
    assert metrics.simultaneous_coverage == 1
    assert math.isnan(metrics.relative_width_by_step[0, 0])
    assert math.isnan(metrics.mean_relative_width)
    assert math.isnan(metrics.relative_width_at_quarter)
    assert metrics.relative_width_at_end == threshold / 2


def test_exact_quarter_is_undefined_when_off_grid():
    truth = np.ones((1, 5, 1))
    metrics = evaluate_band(TrajectoryBatch((7,), truth, truth), 1, np.array(1.0))
    assert metrics.relative_width_at_quarter is None
    assert metrics.mean_relative_width == metrics.relative_width_at_end == 2


def test_quarter_and_end_are_one_based_predicted_steps():
    truth = np.ones((1, 8, 1))
    profile = np.arange(1, 9, dtype=np.float64)[:, None]
    metrics = evaluate_band(TrajectoryBatch((7,), truth, truth), 0.5, np.array(1.0), profile)
    assert metrics.relative_width_at_quarter == 2
    assert metrics.relative_width_at_end == 8


def test_test_residuals_do_not_affect_calibration(fixture):
    calibration, test, quadrature, w4 = fixture
    shifted = TrajectoryBatch(test.ids, test.truth, test.prediction + 100)
    result = evaluate_split_band(
        calibration, shifted, 0.25, quadrature, calibration_weights=w4, test_weights=w4,
    )
    assert_array_equal(result.calibration_scores, [1, 0.75, 2])
    assert result.rank == 3 and result.threshold == 2
    assert result.metrics.simultaneous_coverage == 0


def test_split_rejects_shared_or_duplicate_ids(fixture):
    calibration, test, quadrature, _ = fixture
    overlap = TrajectoryBatch(("cal-c", "test-b"), test.truth, test.prediction)
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_split_band(calibration, overlap, 0.25, quadrature)
    duplicate = TrajectoryBatch(("test-a", "test-a"), test.truth, test.prediction)
    with pytest.raises(ValueError, match="unique"):
        evaluate_split_band(calibration, duplicate, 0.25, quadrature)


def test_weight_validation(fixture):
    calibration, test, quadrature, w4 = fixture
    with pytest.raises(ValueError, match="both"):
        evaluate_split_band(calibration, test, 0.25, quadrature, calibration_weights=w4)
    with pytest.raises(ValueError, match="strictly positive"):
        evaluate_band(test, 1, quadrature, np.array(0.0))
    with pytest.raises(ValueError, match="quadrature"):
        evaluate_band(test, 1, np.array([1.0, 0.0]))
