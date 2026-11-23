import numpy as np
import pytest

from trajcp.conformal.core import calibrate, field_score, sup_score_from_steps
from trajcp.conformal.lp import lp_time_weights
from trajcp.conformal.weights import fit_profile, select_profile, step_maxima
from trajcp.rollout.sensitivity import intrinsic_weights, perturbed_initials, raw_sensitivity


def test_low_mode_unit_rms_reproducible_in_any_scope():
    x = 2 * np.pi * np.arange(64) / 64
    initial = np.asarray([np.sin(x), 3 * np.cos(2 * x), np.zeros_like(x)], dtype=np.float32)
    params = np.array([[.001], [.002], [.003]], dtype=np.float32)
    perturbed = perturbed_initials(initial, params, seed=81)
    np.testing.assert_array_equal(perturbed[0], initial)
    np.testing.assert_array_equal(perturbed[:, 1::-1], perturbed_initials(initial[1::-1], params[1::-1], seed=81))
    assert not np.array_equal(perturbed[1], perturbed[2])
    assert not np.array_equal(perturbed, perturbed_initials(initial, params, seed=82))
    delta = perturbed[1:, :2].astype(float) - initial[None, :2]
    np.testing.assert_allclose(np.sqrt(np.mean(delta**2, axis=-1)) / np.sqrt(np.mean(initial[:2]**2, axis=-1)),
                               np.full((8, 2), 1e-3), atol=4e-8, rtol=0)
    # High modes may contain float32 rounding from the addition to the input.
    rounding_bound = np.sum(np.abs(np.spacing(perturbed[1:, :2])), axis=-1, keepdims=True) / 2
    assert np.all(np.abs(np.fft.rfft(delta)[..., 9:]) <= rounding_bound + 1e-13)
    np.testing.assert_array_equal(perturbed[:, 2], 0)
    # This synthetic emulator doubles the initial field at the second step.
    rollout = np.stack([perturbed, 2 * perturbed], axis=2)
    result = raw_sensitivity(rollout)
    np.testing.assert_array_equal(result[:, 1], 2 * result[:, 0])


def test_periodic_smoothing_and_intrinsic_floor():
    spikes = np.zeros((1, 2, 19), dtype=float)
    spikes[0, 0, 0] = 4
    spikes[0, 1, :12] = 2
    smoothed = intrinsic_weights(spikes, 1, 0., .01)
    assert smoothed[0, 0, -1] == pytest.approx(smoothed[0, 0, 1])
    np.testing.assert_allclose(smoothed[0, 0].sum(), 4.19)
    w = intrinsic_weights(spikes, 0, .1, .01)
    assert w[0, 0, 1] == .01
    assert w[0, 1, -1] == pytest.approx(.21)
    np.testing.assert_array_equal(w, intrinsic_weights(np.concatenate([spikes, 50 * spikes]), 0, .1, .01)[:1])


def test_twofold_selection_requires_finite_ranks_and_exercises_all_families():
    growth = (1 + 2 * np.arange(1, 9) / 8)**1.2
    steps = np.arange(1, 21)[:, None] * growth
    one = np.ones_like(steps)
    with pytest.raises(ValueError, match="finite"):
        select_profile(steps, one, (np.arange(8), np.arange(8, 20)), alpha=.1, epsilon0=1e-6, eta=.02)
    with pytest.raises(ValueError, match="partition"):
        select_profile(steps, one, (np.arange(10), np.arange(9, 19)), alpha=.1, epsilon0=1e-6, eta=.02)
    a, b = np.arange(0, 20, 2), np.arange(1, 20, 2)
    profile, chosen, results = select_profile(steps, one, (a, b), alpha=.1, epsilon0=1e-6, eta=.02)
    assert len(results) == 7 and {row["method"] for row in results} == {"q", "lp", "par"}
    assert chosen["mean_relative_width"] == min(row["mean_relative_width"] for row in results)
    assert (profile > 0).all()
    parametric = fit_profile(steps, "par", epsilon0=1e-6, eta=.02)
    np.testing.assert_allclose(parametric, np.median(steps, axis=0), rtol=1e-5)
    q_profile = fit_profile(steps[:, ::-1], "q", beta=.5, epsilon0=1e-6, eta=.02)
    np.testing.assert_allclose(q_profile, np.mean(np.median(steps, axis=0)))


def test_lp_reciprocal_field_cache_and_global_rescaling_through_bands():
    rng = np.random.default_rng(7)
    error = rng.normal(size=(32, 10, 7)) * np.arange(1., 11.)[None, :, None]
    w3 = rng.uniform(.1, 2., size=error.shape)
    steps = step_maxima(error, w3)
    fit = slice(0, 12)
    cal = slice(12, 24)
    test = slice(24, None)
    inverse = lp_time_weights(steps[fit])
    profile = fit_profile(steps[fit], "lp", epsilon0=1e-6, eta=.02)
    np.testing.assert_array_equal(profile, 1 / inverse)
    cached = (steps * inverse).max(axis=1)
    weights = w3 * profile[None, :, None]
    np.testing.assert_allclose(sup_score_from_steps(steps, profile), cached, rtol=1e-14)
    np.testing.assert_allclose(field_score(error, weights), cached, rtol=1e-14)
    q = calibrate(cached[cal], .1)
    alpha_q = calibrate(field_score(error[cal], weights[cal]), .1)
    scaled_q = calibrate(field_score(error[cal], 7.3 * weights[cal]), .1)
    radius = alpha_q * weights[test]
    np.testing.assert_allclose(radius, scaled_q * 7.3 * weights[test], rtol=1e-14)
    np.testing.assert_allclose(2 * radius, 2 * q * w3[test] / inverse[None, :, None], rtol=1e-14)
    np.testing.assert_array_equal((np.abs(error[test]) <= radius).all(axis=(1, 2)), cached[test] <= q)
