import math

import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from trajcp.conformal.core import band_radius, calibrate, conformal_rank, field_score


@st.composite
def fields(draw):
    shape = (draw(st.integers(1, 25)), draw(st.integers(1, 8)), draw(st.integers(1, 8)))
    errors = draw(arrays(np.float64, shape, elements=st.integers(-20, 20)))
    weights = draw(arrays(np.float64, shape, elements=st.integers(1, 20)))
    return errors, weights


@settings(max_examples=150, derandomize=True, database=None)
@given(
    fields(),
    st.sampled_from([0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.9]),
    st.floats(min_value=0.125, max_value=8, allow_nan=False),
)
def test_core_6_conformal_properties(data, alpha, scale):
    errors, weights = data
    n, horizon, _ = errors.shape
    scores = field_score(errors, weights)
    rank = conformal_rank(n, alpha)
    q = calibrate(scores, alpha)
    included = np.count_nonzero(scores <= q)
    if rank <= n:
        assert included >= rank
    else:
        assert q == math.inf
        assert included == n
        assert np.all(np.isposinf(band_radius(q, weights)))

    scaled_weights = scale * weights
    q_scaled = calibrate(field_score(errors, scaled_weights), alpha)
    assert_allclose(
        band_radius(q_scaled, scaled_weights), band_radius(q, weights), rtol=1e-14, atol=0
    )
    assert calibrate(scores, alpha / 2) >= q >= calibrate(scores, (alpha + 1) / 2)

    prefix_thresholds = [
        calibrate(field_score(errors[:, :t], weights[:, :t]), alpha)
        for t in range(1, horizon + 1)
    ]
    assert prefix_thresholds == sorted(prefix_thresholds)
    tied_scores = np.full(n, 3.5)
    assert np.all(tied_scores <= calibrate(tied_scores, alpha))
