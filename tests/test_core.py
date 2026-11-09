"""Conformal core: ranks, weighted calibration, reductions, scores, LP, CRC and horizons."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from trajcp.conformal import core

SEED = 31415926  # theory-check seed; not one of the registered data streams


def rng_for(*key: int) -> np.random.Generator:
    return np.random.default_rng([SEED, *key])


# --------------------------------------------------------------------------------------
# Rank and threshold edge cases
# --------------------------------------------------------------------------------------

ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5]


def exact_rank(n: int, alpha: float) -> int:
    return math.ceil((n + 1) * (1 - Fraction(str(alpha))))


def test_rank_matches_exact_rational_arithmetic() -> None:
    for alpha in ALPHAS:
        for n in range(1, 2001):
            assert core.conformal_rank(n, alpha) == exact_rank(n, alpha), (n, alpha)


def test_threshold_is_infinite_exactly_when_the_rank_exceeds_n() -> None:
    assert math.isinf(core.calibrate(np.arange(8.0), 0.1))
    assert core.calibrate(np.arange(9.0), 0.1) == 8.0  # n = 9 is the smallest finite case
    assert math.isinf(core.calibrate(np.arange(18.0), 0.05))
    assert core.calibrate(np.arange(19.0), 0.05) == 18.0
    for alpha in ALPHAS:
        for n in range(1, 120):
            finite = math.isfinite(core.calibrate(np.arange(float(n)), alpha))
            assert finite == (exact_rank(n, alpha) <= n)


def test_calibrate_returns_the_kth_smallest_and_leaves_input_alone() -> None:
    scores = np.array([5.0, 1.0, 4.0, 2.0, 3.0, 9.0, 8.0, 7.0, 6.0, 10.0])
    before = scores.copy()
    assert core.calibrate(scores, 0.2) == 9.0  # k = ceil(11 * 0.8) = 9
    np.testing.assert_array_equal(scores, before)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, math.nan])
def test_alpha_outside_the_open_unit_interval_is_rejected(alpha: float) -> None:
    with pytest.raises(ValueError):
        core.calibrate(np.arange(10.0), alpha)


def test_bonferroni_wall_matches_the_closed_form_and_the_plan_table() -> None:
    plan_table = {
        0.10: {16: 159, 32: 319, 64: 639, 100: 999, 250: 2499, 500: 4999},
        0.05: {16: 319, 32: 639, 64: 1279, 100: 1999, 250: 4999, 500: 9999},
    }
    for alpha, row in plan_table.items():
        for steps, expected in row.items():
            n = 1
            while core.conformal_rank(n, alpha / steps) > n:
                n += 1
            assert n == expected == round(steps / alpha) - 1, (alpha, steps)


# --------------------------------------------------------------------------------------
# Weighted calibration with unit ratios equals unweighted
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [9, 19, 20, 30, 50, 99, 100, 200, 300, 600])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.3])
def test_unit_ratios_reproduce_unweighted_calibration(n: int, alpha: float) -> None:
    scores = rng_for(2, n, round(alpha * 100)).standard_exponential(n)
    ones = np.ones(n)
    assert core.weighted_calibrate(scores, ones, 1.0, alpha) == core.calibrate(scores, alpha)
    tied = np.round(scores, 1)
    assert core.weighted_calibrate(tied, ones, 1.0, alpha) == core.calibrate(tied, alpha)


def exact_weighted_threshold(
    scores: list[float], ratios: list[int], test_ratio: int, alpha: float
) -> float:
    """Definition of Algorithm 4 in exact rational arithmetic."""
    total = sum(ratios) + test_ratio
    target = 1 - Fraction(str(alpha))
    for value in sorted(set(scores)):
        mass = sum(Fraction(r, total) for s, r in zip(scores, ratios, strict=True) if s <= value)
        if mass >= target:
            return value
    return math.inf


@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.3, 0.5])
def test_weighted_threshold_matches_exact_definition(alpha: float) -> None:
    rng = rng_for(3, round(alpha * 100))
    for _ in range(200):
        n = int(rng.integers(1, 25))
        scores = np.round(rng.standard_exponential(n), 1)
        ratios = rng.integers(0, 8, size=n)
        test_ratio = int(rng.integers(1, 8))
        expected = exact_weighted_threshold(scores.tolist(), ratios.tolist(), test_ratio, alpha)
        assert core.weighted_calibrate(scores, ratios.astype(float), float(test_ratio), alpha) == (
            expected
        )


def test_kish_ess_and_its_monitoring_rule() -> None:
    assert core.kish_ess(np.ones(50)) == pytest.approx(50.0)
    assert core.kish_ess([1.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert core.kish_ess([1.0, 1.0, 0.0, 0.0]) == pytest.approx(2.0)
    assert core.kish_ess([2.0, 1.0, 1.0]) == pytest.approx(16.0 / 6.0)
    assert core.kish_ess([0.0, 0.0]) == 0.0
    assert core.ess_is_adequate(np.ones(50), 0.1)  # 50 >= 5 / 0.1
    assert not core.ess_is_adequate(np.ones(49), 0.1)
    for bad in ([], [-1.0, 2.0], [math.inf, 1.0], [math.nan]):
        with pytest.raises(ValueError):
            core.kish_ess(bad)


# --------------------------------------------------------------------------------------
# Step-maximum and step-quantile reductions equal the full-field scores
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fields() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = rng_for(4)
    n, steps, cells = 40, 30, 64
    err = rng.standard_normal((n, steps, cells)) * np.exp(0.05 * np.arange(steps))[None, :, None]
    w3 = np.exp(0.3 * rng.standard_normal((n, steps, cells)))
    a = 0.7 * np.exp(0.04 * np.arange(steps))
    return err, w3, a


def test_step_maxima_reproduce_the_sup_score(
    fields: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    err, w3, a = fields
    full = core.field_score(err, a[None, :, None] * w3)
    reduced = core.sup_score_from_steps(core.step_maxima(err, w3), a)
    np.testing.assert_allclose(reduced, full, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("gamma", [0.001, 0.005, 0.01, 0.05, 0.1])
def test_step_quantiles_reproduce_the_per_step_fraction_score(
    fields: tuple[np.ndarray, np.ndarray, np.ndarray], gamma: float
) -> None:
    err, w3, a = fields
    full = core.field_score(err, a[None, :, None] * w3, gamma, per_step=True)
    reduced = core.sup_score_from_steps(core.step_quantiles(err, w3, gamma), a)
    np.testing.assert_allclose(reduced, full, rtol=1e-14, atol=0.0)


def test_prefix_scores_match_truncated_fields(
    fields: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    err, w3, a = fields
    prefix = core.prefix_scores(core.step_maxima(err, w3), a)
    assert prefix.shape == err.shape[:2]
    assert np.all(prefix[:, 1:] >= prefix[:, :-1])
    sup = core.sup_score_from_steps(core.step_maxima(err, w3), a)
    np.testing.assert_array_equal(prefix[:, -1], sup)
    for steps in (1, 7, 16, 30):
        truncated = core.field_score(err[:, :steps], (a[None, :, None] * w3)[:, :steps])
        np.testing.assert_allclose(prefix[:, steps - 1], truncated, rtol=1e-14, atol=0.0)


def test_one_threshold_covers_every_prefix_at_once(
    fields: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    err, w3, a = fields
    prefix = core.prefix_scores(core.step_maxima(err, w3), a)
    q = core.calibrate(prefix[:, -1], 0.2)
    final_ok = prefix[:, -1] <= q
    every_prefix_ok = np.all(prefix <= q, axis=1)
    np.testing.assert_array_equal(final_ok, every_prefix_ok)
    thresholds = core.prefix_thresholds(prefix, 0.2)
    assert np.all(thresholds[1:] >= thresholds[:-1])
    assert thresholds[-1] == q


# --------------------------------------------------------------------------------------
# Semantics of the fraction level gamma
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("gamma", [0.001, 0.005, 0.01, 0.05, 0.1, 0.3])
def test_space_time_fraction_score_is_the_smallest_valid_level(gamma: float) -> None:
    rng = rng_for(5, round(gamma * 1000))
    err = rng.standard_normal((25, 12, 40))
    score = core.field_score(err, np.ones_like(err), gamma)
    r = np.abs(err).reshape(25, -1)
    cells = r.shape[1]
    for i in range(25):
        assert np.sum(r[i] > score[i]) <= gamma * cells  # at most gamma of the points exceed it
        ordered = np.sort(r[i])
        j = math.ceil(round((1 - gamma) * cells, 12))
        assert score[i] == ordered[j - 1]
        if j >= 2:  # one step lower and more than gamma of the points would exceed
            assert np.sum(r[i] > ordered[j - 2]) > gamma * cells


@pytest.mark.parametrize("gamma", [0.001, 0.01, 0.05, 0.1, 0.3])
def test_per_step_fraction_score_covers_every_step(gamma: float) -> None:
    rng = rng_for(6, round(gamma * 1000))
    err = rng.standard_normal((20, 15, 50))
    score = core.field_score(err, np.ones_like(err), gamma, per_step=True)
    r = np.abs(err)
    for i in range(20):
        exceed = np.sum(r[i] > score[i], axis=1)  # exceedances per step
        assert np.all(exceed <= gamma * 50)
    assert np.all(score <= core.field_score(err, np.ones_like(err)))
    np.testing.assert_array_equal(
        core.field_score(err, np.ones_like(err), 0.0, per_step=True),
        core.field_score(err, np.ones_like(err), 0.0),
    )


# --------------------------------------------------------------------------------------
# The LP recovers an exponential growth profile
# --------------------------------------------------------------------------------------


def test_lp_recovers_the_exponential_growth_profile() -> None:
    steps, n_fit, true_ratio = 100, 300, 20.0
    growth = math.log(true_ratio) / (steps - 1)
    rng = rng_for(8)
    scale = rng.lognormal(0.0, 0.5, size=(n_fit, 1))
    noise = rng.lognormal(0.0, 0.25, size=(n_fit, steps))
    stats = scale * np.exp(growth * np.arange(steps))[None, :] * noise
    a = core.lp_time_weights(stats)
    assert a.shape == (steps,)
    assert a.sum() == pytest.approx(1.0)
    assert np.all(a > 0.0)
    measured = a[0] / a[-1]  # radius at the last step over radius at the first step
    assert true_ratio / 2 <= measured <= true_ratio * 2, f"radius ratio {measured:.2f}"


def test_lp_rejects_non_finite_statistics() -> None:
    stats = np.ones((5, 4))
    stats[2, 1] = np.inf
    with pytest.raises(ValueError):
        core.lp_time_weights(stats)


# --------------------------------------------------------------------------------------
# Conformal risk control
# --------------------------------------------------------------------------------------


def crc_by_definition(r: np.ndarray, alpha: float) -> float:
    """inf{lambda >= 0: (sum_i L_i(lambda) + 1) / (n + 1) <= alpha} over the candidate values."""
    n = r.shape[0]
    flat = r.reshape(n, -1)
    cells = flat.shape[1]
    target = Fraction(str(alpha))
    for lam in sorted({0.0, *flat.ravel().tolist()}):
        total = sum(Fraction(int(np.sum(flat[i] > lam)), cells) for i in range(n))
        if (total + 1) / (n + 1) <= target:
            return lam
    return math.inf


@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.3, 0.5])
def test_crc_threshold_matches_its_definition(alpha: float) -> None:
    rng = rng_for(9, round(alpha * 100))
    for _ in range(150):
        n = int(rng.integers(1, 30))
        cells = int(rng.integers(1, 12))
        r = np.round(rng.standard_exponential((n, cells)), 1)
        assert core.crc_threshold(r, alpha) == crc_by_definition(r, alpha), (n, cells, alpha)


def test_crc_threshold_is_infinite_when_no_lambda_qualifies() -> None:
    r = np.ones((5, 4))
    assert math.isinf(core.crc_threshold(r, 0.1))  # (5 + 1) * 0.1 < 1
    assert core.crc_threshold(np.zeros((30, 4)), 0.1) == 0.0


def test_crc_controls_the_expected_excursion_fraction() -> None:
    alpha, n, cells, repeats = 0.2, 50, 20, 4000
    rng = rng_for(10)
    losses = np.empty(repeats)
    for j in range(repeats):
        draws = rng.standard_t(3, size=(n + 1, cells))
        lam = core.crc_threshold(np.abs(draws[:n]), alpha)
        losses[j] = core.excursion_fraction(np.abs(draws[n:]), lam)[0]
    standard_error = losses.std(ddof=1) / math.sqrt(repeats)
    assert losses.mean() <= alpha + 4 * standard_error
    assert losses.mean() > alpha / 2  # not vacuous


# --------------------------------------------------------------------------------------
# Horizon-specific thresholds
# --------------------------------------------------------------------------------------


def test_horizon_thresholds_have_the_stated_structure() -> None:
    rng = rng_for(11)
    steps, grid = 40, (4, 10, 25, 40)
    growth = np.exp(0.03 * np.arange(steps))
    fit = core.prefix_scores(rng.standard_exponential((60, steps)) * growth)
    cal = core.prefix_scores(rng.standard_exponential((80, steps)) * growth)
    result = core.horizon_thresholds(fit, cal, grid, 0.1)
    assert result.horizons == grid
    columns = [h - 1 for h in grid]
    expected_scales = np.quantile(fit[:, columns], 0.9, axis=0, method="higher")
    np.testing.assert_array_equal(result.scales, expected_scales)
    z = core.horizon_meta_scores(cal, grid, result.scales)
    assert result.meta_threshold == core.calibrate(z, 0.1)
    expected = result.meta_threshold * result.scales
    np.testing.assert_allclose(result.thresholds, expected, rtol=0, atol=0)
    inside = z <= result.meta_threshold
    for column, h in enumerate(grid):
        assert np.all(cal[inside, h - 1] <= result.thresholds[column])


def test_horizon_grid_is_simultaneously_valid_without_a_union_bound() -> None:
    alpha, n_fit, n_cal, steps, repeats = 0.1, 40, 40, 64, 3000
    grid = (4, 16, 32, 64)
    rng = rng_for(12)
    hits = np.empty(repeats, dtype=bool)
    profile = np.exp(0.02 * np.arange(steps))
    for j in range(repeats):
        raw = np.abs(rng.standard_normal((n_fit + n_cal + 1, steps))) * profile
        prefix = core.prefix_scores(raw)
        result = core.horizon_thresholds(prefix[:n_fit], prefix[n_fit : n_fit + n_cal], grid, alpha)
        test = prefix[-1]
        hits[j] = all(test[h - 1] <= result.thresholds[c] for c, h in enumerate(grid))
    k = core.conformal_rank(n_cal, alpha)
    expected = k / (n_cal + 1)  # exact for continuous scores
    standard_error = math.sqrt(expected * (1 - expected) / repeats)
    assert abs(hits.mean() - expected) <= 3.5 * standard_error, (hits.mean(), expected)


def test_infinite_horizon_scale_gives_an_infinite_threshold() -> None:
    fit = np.ones((10, 6))
    fit[:5, 3:] = np.inf  # half of D_fit blew up from step 4 on
    cal = np.ones((20, 6))
    result = core.horizon_thresholds(fit, cal, (2, 6), 0.1)
    assert math.isfinite(result.thresholds[0])
    assert math.isinf(result.thresholds[1])
    with pytest.raises(ValueError):
        core.horizon_thresholds(np.zeros((10, 6)), cal, (2, 6), 0.1)  # all-zero fit scores


@pytest.mark.parametrize("grid", [(), (0, 3), (3, 3), (5, 3), (2, 99)])
def test_invalid_horizon_grids_are_rejected(grid: tuple[int, ...]) -> None:
    prefix = np.ones((10, 8))
    with pytest.raises(ValueError):
        core.horizon_thresholds(prefix, prefix, grid, 0.1)


# --------------------------------------------------------------------------------------
# Non-finite handling: failed rollouts are never dropped
# --------------------------------------------------------------------------------------


def test_failed_rollouts_stay_in_the_sample_as_infinite_scores() -> None:
    base = np.arange(1.0, 11.0)  # n = 10, k = 9 at alpha = 0.2
    one_failure = base.copy()
    one_failure[3] = np.nan
    two_failures = base.copy()
    two_failures[[3, 6]] = np.nan
    assert core.calibrate(base, 0.2) == 9.0
    assert core.calibrate(one_failure, 0.2) == 10.0  # one +inf pushes the 9th value up by one
    assert math.isinf(core.calibrate(two_failures, 0.2))  # more than alpha failed: infinite
    assert not core.covered([np.nan], 1e12)[0]
    assert core.covered([np.nan], math.inf)[0]
    assert core.covered([1.0, 2.0, 3.0], 2.0).tolist() == [True, True, False]


def test_nan_in_step_statistics_gives_an_infinite_sup_score() -> None:
    m = np.ones((3, 4))
    m[1, 2] = np.nan
    scores = core.sup_score_from_steps(m)
    assert scores[0] == 1.0 and scores[2] == 1.0 and math.isinf(scores[1])
    assert math.isinf(core.prefix_scores(m)[1, 3]) and core.prefix_scores(m)[1, 1] == 1.0
    for bad_profile in (np.array([1.0, 0.0, 1.0, 1.0]), np.array([1.0, np.nan, 1.0, 1.0])):
        with pytest.raises(ValueError):
            core.sup_score_from_steps(m, bad_profile)


def test_non_finite_weights_become_infinite_bands_and_non_finite_rollouts_score_infinity() -> None:
    err = np.ones((1, 2, 3))
    weights = np.ones((1, 2, 3))
    weights[0, 0, 0] = 0.0
    weights[0, 0, 1] = np.nan
    weights[0, 1, 2] = -2.0
    assert core.infinite_cell_fraction(weights) == pytest.approx(3 / 6)
    # Infinite bands cover their cells: those cells contribute nothing to the score.
    assert core.field_score(err, weights)[0] == 1.0
    assert core.field_score(err, np.full((1, 2, 3), np.inf))[0] == 0.0
    # A non-finite state gives +inf whatever the weight is.
    err_failed = err.copy()
    err_failed[0, 1, 0] = np.nan
    assert math.isinf(core.field_score(err_failed, weights)[0])
    err_failed[0, 1, 0] = np.inf
    assert math.isinf(core.field_score(err_failed, np.full((1, 2, 3), np.inf))[0])
    with pytest.raises(ValueError):
        core.field_score(err, np.ones((2, 2, 3)))  # weights may not enlarge the error array


def test_float32_statistics_are_converted_to_float64() -> None:
    m32 = np.array([[0.1, 0.2], [0.3, 0.1]], dtype=np.float32)
    result = core.sup_score_from_steps(m32)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, m32.astype(np.float64).max(axis=1))
