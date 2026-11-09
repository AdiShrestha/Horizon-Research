"""The threshold agrees with independent implementations."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from trajcp.conformal import core

SEED = 31415926  # theory-check seed; not one of the registered data streams


def counting_oracle(scores: list[float], alpha: float) -> float:
    """Smallest score v with #{S <= v} >= ceil((n + 1)(1 - alpha)), in exact arithmetic."""
    k = math.ceil((len(scores) + 1) * (1 - Fraction(str(alpha))))
    for value in sorted(set(scores)):
        if sum(1 for s in scores if s <= value) >= k:
            return value
    return math.inf


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
def test_matches_an_exact_counting_oracle(alpha: float) -> None:
    rng = np.random.default_rng([SEED, 71, round(alpha * 100)])
    for n in [*range(1, 60), 99, 100, 101, 599, 600, 601]:
        for rounded in (False, True):  # rounding creates ties
            scores = rng.standard_exponential(n)
            if rounded:
                scores = np.round(scores, 1)
            assert core.calibrate(scores, alpha) == counting_oracle(scores.tolist(), alpha), (n,)


@pytest.mark.oracle
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
@pytest.mark.parametrize("n", [8, 9, 19, 20, 50, 99, 100, 300, 600])
def test_matches_mapie_split_conformal_on_one_step_slices(n: int, alpha: float) -> None:
    pytest.importorskip("mapie")
    from mapie.regression import SplitConformalRegressor
    from sklearn.dummy import DummyRegressor

    rng = np.random.default_rng([SEED, 72, n, round(alpha * 100)])
    y = 2.0 * rng.standard_normal(n) + 1.0
    x = np.zeros((n, 1))
    # A constant predictor of 0: the absolute residual of every calibration point is |y|.
    model = DummyRegressor(strategy="constant", constant=0.0).fit(x, y)
    mapie = SplitConformalRegressor(
        estimator=model, confidence_level=1.0 - alpha, conformity_score="absolute", prefit=True
    )
    expected = core.calibrate(np.abs(y), alpha)
    try:
        mapie.conformalize(x, y)
        _, intervals = mapie.predict_interval(np.zeros((1, 1)), allow_infinite_bounds=True)
    except ValueError as error:
        # MAPIE refuses (instead of returning +inf) exactly when the rank exceeds n.
        assert math.isinf(expected), error
        assert "too low" in str(error)
        return
    upper = float(intervals[0, 1, 0])
    if math.isinf(expected):
        assert math.isinf(upper)
    else:
        assert abs(upper - expected) <= 1e-12, (upper, expected)
