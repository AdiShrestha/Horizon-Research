import os

import numpy as np
import pytest
from scipy.signal import lfilter
from scipy.stats import betabinom, binom, ks_2samp

from trajcp.conformal.core import calibrate, conformal_rank, field_score


def _gp_factors(horizon, cells):
    def factor(size, length_scale):
        distances = np.subtract.outer(np.arange(size), np.arange(size))
        covariance = np.exp(-0.5 * (distances / length_scale) ** 2)
        return np.linalg.cholesky(covariance + 1e-10 * np.eye(size))

    return factor(horizon, 2.0), factor(cells, 2.5)


@pytest.mark.parametrize(("model", "seed"), [("AR1", 1247), ("GP", 1248)])
def test_core_3_beta_law(model, seed):
    splits = 10_000 if os.environ.get("TRAJCP_NIGHTLY") == "1" else 400
    horizon, cells, n_cal, n_test = 50, 64, 19, 24
    alpha = 0.1
    rank = conformal_rank(n_cal, alpha)
    expected = rank / (n_cal + 1)
    rng = np.random.default_rng(seed)
    reference_rng = np.random.default_rng(seed + 10_000)

    growth = np.geomspace(1.0, 6.0, horizon)[:, None]
    rho = 0.8
    time_factor, space_factor = _gp_factors(horizon, cells)
    counts = np.empty(splits, dtype=int)
    first_test_included = np.empty(splits, dtype=bool)
    for i in range(splits):
        fields = rng.normal(size=(n_cal + n_test, horizon, cells))
        if model == "AR1":
            # A stationary AR(1) in time at each spatial cell.
            fields[:, 1:, :] *= np.sqrt(1 - rho**2)
            fields = lfilter([1.0], [1.0, -rho], fields, axis=1)
        else:
            fields = time_factor @ fields @ space_factor.T
        scores = field_score(fields * growth, growth)
        q = calibrate(scores[:n_cal], alpha)
        included = scores[n_cal:] <= q
        first_test_included[i] = included[0]
        counts[i] = included.sum()

    # One preselected test trajectory per independent split gives a binomial
    # count. Check the full-cohort mean against the same acceptance band.
    lower, upper = np.array(binom.interval(0.999, splits, expected)) / splits
    coverage = counts.mean() / n_test
    assert lower <= first_test_included.mean() <= upper
    assert lower <= coverage <= upper

    distribution = betabinom(n_test, rank, n_cal + 1 - rank)
    reference = distribution.rvs(size=splits, random_state=reference_rng)
    # Calibrate the two-sample KS statistic under the discrete null. The
    # continuous KS p-value does not model the ties in covered counts.
    observed = ks_2samp(counts, reference).statistic
    probabilities = distribution.pmf(np.arange(n_test + 1))
    null_histograms = reference_rng.multinomial(splits, probabilities, size=(1999, 2))
    null_ks = np.max(
        abs(np.cumsum(null_histograms[:, 0] - null_histograms[:, 1], axis=1)), axis=1
    )
    observed_ks = round(observed * splits)
    p_value = (1 + np.count_nonzero(null_ks >= observed_ks)) / (len(null_ks) + 1)
    print(
        f"{model} {splits=:,} mean={coverage:.6f} expected={expected:.6f} "
        f"binomial band=[{lower:.6f}, {upper:.6f}] "
        f"independent sentinel mean={first_test_included.mean():.6f} "
        f"KS={observed:.5f} p={p_value:.4f}"
    )
    assert p_value >= 0.001
