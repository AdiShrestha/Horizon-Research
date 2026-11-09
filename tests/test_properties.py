"""Properties of the calibration threshold (hypothesis)."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from trajcp.conformal import core

ALPHAS = st.sampled_from([0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.9])
FINITE = {"allow_nan": False, "allow_infinity": False}
SCORES = hnp.arrays(
    np.float64,
    st.integers(1, 60),
    elements=st.floats(0.0, 1e6, **FINITE),
)


@given(SCORES, ALPHAS)
def test_calibration_set_inclusion(scores: np.ndarray, alpha: float) -> None:
    n = scores.size
    k = core.conformal_rank(n, alpha)
    q = core.calibrate(scores, alpha)
    if k > n:
        assert math.isinf(q)
    else:
        assert q in scores
        assert np.count_nonzero(scores <= q) >= k  # at least ceil((n + 1)(1 - alpha)) / n


@given(
    st.integers(1, 12).flatmap(
        lambda n: st.tuples(
            hnp.arrays(np.float64, (n, 4, 5), elements=st.floats(-1e3, 1e3, **FINITE)),
            hnp.arrays(np.float64, (n, 4, 5), elements=st.floats(1e-3, 1e3, **FINITE)),
        )
    ),
    st.floats(1e-3, 1e3, **FINITE),
    ALPHAS,
)
def test_rescaling_the_weights_leaves_the_band_unchanged(
    data: tuple[np.ndarray, np.ndarray], c: float, alpha: float
) -> None:
    err, w = data
    q = core.calibrate(core.field_score(err, w), alpha)
    q_scaled = core.calibrate(core.field_score(err, c * w), alpha)
    if math.isinf(q) or math.isinf(q_scaled):
        assert math.isinf(q) and math.isinf(q_scaled)
    else:
        np.testing.assert_allclose(q_scaled * (c * w), q * w, rtol=1e-14, atol=0.0)


@given(SCORES, ALPHAS, ALPHAS)
def test_thresholds_are_nested_in_alpha(scores: np.ndarray, a1: float, a2: float) -> None:
    low, high = sorted((a1, a2))
    assert core.calibrate(scores, low) >= core.calibrate(scores, high)


@given(
    hnp.arrays(
        np.float64,
        st.tuples(st.integers(1, 40), st.integers(1, 30)),
        elements=st.floats(0.0, 1e6, **FINITE),
    ),
    ALPHAS,
)
def test_threshold_is_monotone_in_the_horizon(m: np.ndarray, alpha: float) -> None:
    thresholds = core.prefix_thresholds(core.prefix_scores(m), alpha)
    assert np.all(thresholds[1:] >= thresholds[:-1])  # inf >= inf holds, unlike np.diff


@given(st.floats(0.0, 1e6, **FINITE), st.integers(1, 80), ALPHAS)
def test_equal_scores_give_full_coverage(value: float, n: int, alpha: float) -> None:
    scores = np.full(n, value)
    q = core.calibrate(scores, alpha)
    assert q == value or math.isinf(q)
    assert bool(core.covered([value], q)[0])
    assert np.all(core.covered(scores, q))
