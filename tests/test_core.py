import math
from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from trajcp.conformal.core import (
    band_radius,
    calibrate,
    conformal_rank,
    field_score,
    kish_ess,
    sup_score_from_steps,
    weighted_calibrate,
)
from trajcp.conformal.lp import lp_time_weights


@pytest.mark.parametrize(
    ("n", "alpha", "rank"),
    [
        (0, 0.1, 1),
        (8, 0.1, 9),
        (9, np.nextafter(0.1, 0), 10),
        (9, 0.1, 9),
        (9, np.nextafter(0.1, 1), 9),
        (18, 0.05, 19),
        (19, 0.05, 19),
        (19, 0.2, 16),
        (2, np.nextafter(1.0, 0), 1),
    ],
)
def test_core_1_rank_boundary(n, alpha, rank):
    scores = np.arange(n, dtype=np.float64)[::-1]
    assert conformal_rank(n, alpha) == rank
    assert calibrate(scores, alpha) == (rank - 1 if rank <= n else math.inf)
    radius = band_radius(calibrate(scores, alpha), np.array(2.0))
    assert isinstance(radius, np.ndarray) and radius.shape == ()
    assert radius == (2 * (rank - 1) if rank <= n else math.inf)


@pytest.mark.parametrize(
    ("n", "alpha"),
    tuple(product((1, 2, 8, 9, 10, 18, 19, 50, 99, 600), (0.01, 0.05, 0.1, 0.2))),
)
def test_core_2_equal_ratios(n, alpha):
    scores = np.random.default_rng(n).integers(0, 40, size=n).astype(np.float64)
    assert weighted_calibrate(scores, np.ones(n), 1.0, alpha) == calibrate(scores, alpha)


def test_core_2_test_mass_and_ties():
    scores = np.array([7.0, 2.0, 7.0, 12.0, 1.0])
    ratios = np.array([1.0, 2.0, 3.0, 2.0, 0.0])
    assert weighted_calibrate(scores, ratios, 2.0, 0.4) == 7.0
    assert weighted_calibrate(scores, ratios, 2.0, 0.2) == 12.0
    assert weighted_calibrate(scores, ratios, np.nextafter(2.0, 0), 0.2) == 12.0
    assert weighted_calibrate(scores, ratios, np.nextafter(2.0, 3), 0.2) == math.inf
    assert weighted_calibrate(scores, ratios, 3.0, 0.2) == math.inf
    assert weighted_calibrate(scores, ratios, 0.0, 0.95) == 2.0
    assert weighted_calibrate(scores, ratios * 1e300, 2e300, 0.4) == 7.0
    assert weighted_calibrate(scores, np.zeros_like(ratios), 1.0, 0.2) == math.inf
    assert weighted_calibrate(scores, ratios, 0.0, np.nextafter(1.0, 0)) == 2.0
    with pytest.raises(ValueError, match="positive"):
        weighted_calibrate(scores, np.zeros_like(ratios), 0.0, 0.2)
    assert kish_ess(np.ones(9)) == 9
    assert kish_ess(ratios * 1e300) == pytest.approx(64 / 18)


def test_core_4_cached_sup_and_psf():
    rng = np.random.default_rng(199)
    err = rng.normal(size=(8, 5, 4, 6))
    w3 = rng.uniform(0.25, 2, size=err.shape)
    envelope = np.geomspace(1, 9, num=err.shape[1])
    w4 = envelope[None, :, None, None] * w3
    normalized = (abs(err) / w3).reshape(8, 5, 24)

    cached_maxima = normalized.max(axis=-1)
    assert_allclose(sup_score_from_steps(cached_maxima, envelope), field_score(err, w4))

    # A positive-gamma PSF cache stores the spatial order statistic per step.
    gamma = 0.25
    cached_psf = np.sort(normalized, axis=-1)[..., 17]
    expected = field_score(err, w4, gamma=gamma, per_step=True)
    assert_allclose(sup_score_from_steps(cached_psf, envelope), expected)
    assert np.all(sup_score_from_steps(cached_maxima, envelope) > expected)


def test_core_5_fraction_semantics():
    # Concentrate both misses in one step. STF may drop both, PSF only one.
    field = np.array([[[100.0, 90.0, 2.0, 2.0], [2.0, 2.0, 1.0, 1.0]]])
    unit = np.array(1.0)
    stf = field_score(field, unit, gamma=0.25)
    psf = field_score(field, unit, gamma=0.25, per_step=True)
    assert_array_equal(stf, [2.0])
    assert_array_equal(psf, [90.0])
    assert (field > stf[0]).mean() == 0.25
    assert (field > stf[0]).mean(axis=-1).max() == 0.5
    assert (field > psf[0]).mean(axis=-1).max() == 0.25
    for per_step in (False, True):
        assert_array_equal(field_score(field, unit, per_step=per_step), [100.0])

    rng = np.random.default_rng(189)
    field = rng.integers(0, 6, size=(21, 5, 8)).astype(np.float64)
    for gamma in (0.001, 0.025, 0.1, 0.25, 0.99):
        stf = field_score(field, unit, gamma=gamma)
        limit = math.floor(Fraction(str(gamma)) * 40)
        assert np.all((field > stf[:, None, None]).sum(axis=(1, 2)) <= limit)
        assert np.all((field >= stf[:, None, None]).sum(axis=(1, 2)) > limit)
        psf = field_score(field, unit, gamma=gamma, per_step=True)
        limit = math.floor(Fraction(str(gamma)) * 8)
        assert np.all((field > psf[:, None, None]).sum(axis=2) <= limit)


@pytest.mark.parametrize(
    ("function", "args", "error"),
    [
        (calibrate, (np.array([1], dtype=np.float32), 0.2), TypeError),
        (calibrate, (np.array([np.nan]), 0.2), ValueError),
        (calibrate, (np.array([1.0]), 1.0), ValueError),
        (field_score, (np.ones((1, 1, 1)), np.array(0.0)), ValueError),
        (sup_score_from_steps, (np.ones((2, 3)), np.ones(2)), ValueError),
        (lp_time_weights, (np.array([[math.inf]]),), ValueError),
    ],
)
def test_core_input_contract(function, args, error):
    with pytest.raises(error):
        function(*args)


def test_core_8_lp_exponential_growth_end_to_end():
    rng = np.random.default_rng(800)
    horizon = 100
    growth = np.geomspace(1, 20, horizon)
    sensitivity = rng.uniform(0.75, 1.75, size=(128, horizon, 5))
    # Add modest noise to separable growth and ensure cell maxima are nontrivial.
    errors = (
        growth[None, :, None]
        * rng.uniform(0.2, 2.0, size=(128, 1, 1))
        * rng.uniform(0.8, 1.2, size=(128, horizon, 1))
        * rng.uniform(0.25, 1.0, size=sensitivity.shape)
        * sensitivity
    )
    steps = (abs(errors) / sensitivity).max(axis=-1)
    coefficients = lp_time_weights(steps[:64])
    assert coefficients.shape == (horizon,)
    assert coefficients.dtype == np.float64
    assert np.all(coefficients >= 1e-6 / horizon)
    assert coefficients.sum() == pytest.approx(1.0, abs=1e-12)
    envelope = np.reciprocal(coefficients)
    assert_allclose(envelope / envelope[0], growth, rtol=0.5)
    ratio = envelope[-1] / envelope[0]
    assert 10 <= ratio <= 40

    held_out_steps = steps[64:]
    multiplied = (held_out_steps * coefficients).max(axis=1)
    divided = sup_score_from_steps(held_out_steps, envelope)
    w4 = sensitivity[64:] * envelope[None, :, None]
    full_field = field_score(errors[64:], w4)
    assert_allclose(multiplied, divided, rtol=1e-14)
    assert_allclose(full_field, divided, rtol=1e-14)

    q_product = calibrate(multiplied[:32], 0.1)
    q_division = calibrate(divided[:32], 0.1)
    q_field = calibrate(full_field[:32], 0.1)
    assert math.isfinite(q_division)
    assert_allclose([q_product, q_field], q_division, rtol=1e-14)
    width_from_coefficients = 2 * q_product / coefficients[None, :, None] * sensitivity[96:]
    width_from_envelope = 2 * band_radius(q_division, w4[32:])
    assert_allclose(width_from_coefficients, width_from_envelope, rtol=1e-14)
    truth_in_band = (abs(errors[96:]) <= width_from_envelope / 2).all(axis=(1, 2))
    assert_array_equal(truth_in_band, divided[32:] <= q_division)
    assert 0 < truth_in_band.sum() < truth_in_band.size
    time_radius = band_radius(q_division, envelope)
    assert time_radius[-1] / time_radius[0] == pytest.approx(ratio)
    print(f"LP envelope T=100/T=1 ratio={ratio:.6f}, held-out band inclusions="
          f"{truth_in_band.sum()}/{truth_in_band.size}")
