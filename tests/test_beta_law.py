"""Realized coverage follows the Beta-Binomial law implied by exchangeability.

Each split draws n_cal calibration and n_test test scores independently from a score model,
calibrates, and counts covered test points. For continuous scores the count is exactly
Beta-Binomial(n_test, k, n_cal + 1 - k) with k = ceil((n_cal + 1)(1 - alpha)). Two checks:

* the mean coverage over splits lies inside the 99.9% band around k / (n_cal + 1), with the
  variance of the Beta-Binomial law (a plain binomial band is exact only for n_test = 1), and
* the empirical distribution of the counts passes a Kolmogorov-Smirnov test against that law,
  with a Monte Carlo null because the law is discrete.

A negative control shows that an off-by-one rank is detected. The default run uses 400 splits;
the nightly run uses 10,000.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats

from trajcp.conformal import core
from trajcp.theory import fields

SEED = 31415926  # theory-check seed; not one of the registered data streams
ALPHA = 0.1
Z_BAND = float(stats.norm.ppf(0.9995))  # two-sided 99.9% band
FALSE_ALARM = 0.001

Generator = Callable[[np.random.Generator, int], npt.NDArray[np.float64]]
MODELS: dict[str, Generator] = {
    "uniform": fields.uniform_scores,
    "ar1": lambda rng, n: fields.sup_scores(rng, n, length=0.0),
    "gp": lambda rng, n: fields.sup_scores(rng, n, length=3.0),
}


def covered_counts(
    scores: npt.NDArray[np.float64], n_cal: int, rank_shift: int = 0
) -> npt.NDArray[np.int64]:
    """Covered test points per split; rank_shift != 0 deliberately breaks the rank (control)."""
    counts = np.empty(scores.shape[0], dtype=np.int64)
    k = core.conformal_rank(n_cal, ALPHA) + rank_shift
    for split, row in enumerate(scores):
        if rank_shift:
            threshold = float(np.sort(row[:n_cal])[k - 1])
        else:
            threshold = core.calibrate(row[:n_cal], ALPHA)
        counts[split] = np.count_nonzero(core.covered(row[n_cal:], threshold))
    return counts


def ks_statistic(counts: npt.NDArray[np.int64], law: stats.rv_frozen, n_test: int) -> float:
    support = np.arange(n_test + 1)
    empirical = np.cumsum(np.bincount(counts, minlength=n_test + 1)) / counts.size
    return float(np.max(np.abs(empirical - law.cdf(support))))


def ks_pvalue(
    counts: npt.NDArray[np.int64],
    law: stats.rv_frozen,
    n_test: int,
    rng: np.random.Generator,
    draws: int = 2000,
) -> float:
    """Monte Carlo p-value of the KS statistic under the discrete law."""
    observed = ks_statistic(counts, law, n_test)
    null = np.array(
        [
            ks_statistic(np.asarray(law.rvs(size=counts.size, random_state=rng)), law, n_test)
            for _ in range(draws)
        ]
    )
    return (1 + int(np.count_nonzero(null >= observed))) / (1 + draws)


def law_checks(
    counts: npt.NDArray[np.int64], n_cal: int, n_test: int, rng: np.random.Generator
) -> tuple[bool, float]:
    """(mean coverage inside the 99.9% band, KS p-value) against the Beta-Binomial law."""
    splits = counts.size
    k = core.conformal_rank(n_cal, ALPHA)
    law = stats.betabinom(n_test, k, n_cal + 1 - k)
    mean = counts.sum() / (splits * n_test)
    band = Z_BAND * math.sqrt(float(law.var()) / splits) / n_test
    return abs(mean - k / (n_cal + 1)) <= band, ks_pvalue(counts, law, n_test, rng)


def run(model: str, n_cal: int, n_test: int, splits: int, *key: int) -> tuple[bool, float]:
    rng = np.random.default_rng([SEED, 31, *key])
    scores = MODELS[model](rng, splits * (n_cal + n_test)).reshape(splits, n_cal + n_test)
    return law_checks(covered_counts(scores, n_cal), n_cal, n_test, rng)


@pytest.mark.parametrize("model", ["uniform", "ar1", "gp"])
@pytest.mark.parametrize("n_cal", [20, 100])
def test_coverage_follows_the_beta_binomial_law(model: str, n_cal: int) -> None:
    in_band, p_value = run(model, n_cal, 100, 400, list(MODELS).index(model), n_cal)
    assert in_band
    assert p_value > FALSE_ALARM, p_value


@pytest.mark.parametrize("shift", [-1, 1])
def test_an_off_by_one_rank_is_detected(shift: int) -> None:
    n_cal, n_test, splits = 20, 100, 400
    rng = np.random.default_rng([SEED, 32, shift + 2])
    scores = MODELS["uniform"](rng, splits * (n_cal + n_test)).reshape(splits, n_cal + n_test)
    counts = covered_counts(scores, n_cal, rank_shift=shift)
    in_band, p_value = law_checks(counts, n_cal, n_test, rng)
    assert not in_band or p_value <= FALSE_ALARM


@pytest.mark.nightly
@pytest.mark.parametrize("n_cal", [20, 50, 100, 600])
def test_nightly_uniform_scores(n_cal: int) -> None:
    in_band, p_value = run("uniform", n_cal, 100, 10_000, 40, n_cal)
    assert in_band
    assert p_value > FALSE_ALARM, p_value


@pytest.mark.nightly
@pytest.mark.parametrize("model", ["ar1", "gp"])
@pytest.mark.parametrize("n_cal", [20, 50])
def test_nightly_space_time_fields(model: str, n_cal: int) -> None:
    in_band, p_value = run(model, n_cal, 50, 10_000, 41, list(MODELS).index(model), n_cal)
    assert in_band
    assert p_value > FALSE_ALARM, p_value
