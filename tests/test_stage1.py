"""Stage 1 runner and reductions: key discipline, non-finite rule and statistic definitions."""

from __future__ import annotations

import math

import numpy as np

from trajcp.rollout import reduce as rd
from trajcp.rollout import runner


def test_perturbations_are_unit_rms_and_prefix_stable() -> None:
    idx = np.array([3, 4])
    big = runner.perturbation_fields("AUX", "p2", idx, 16, 64)
    small = runner.perturbation_fields("AUX", "p2", idx, 8, 64)
    np.testing.assert_allclose(np.sqrt(np.mean(big**2, axis=-1)), 1.0, rtol=1e-12)
    np.testing.assert_array_equal(big[:, :8], small)
    assert not np.allclose(big[0], big[1])  # different trajectories, different keys


def test_low_mode_perturbations_have_no_energy_above_the_cutoff() -> None:
    xi = runner.perturbation_fields("AUX", "p2", np.array([0]), 4, 64, cutoff=8)
    spec = np.abs(np.fft.rfft(xi, axis=-1))
    assert spec[..., 9:].max() < 1e-10 * spec.max()


def test_reduce_weight_matches_definitions_and_never_drops_nonfinite() -> None:
    rng = np.random.default_rng(0)
    err = rng.standard_normal((3, 5, 256))
    w = np.exp(0.2 * rng.standard_normal((3, 5, 256)))
    stats = rd.reduce_weight(err, w)
    r = np.abs(err) / w
    np.testing.assert_allclose(stats["m"], r.max(-1), rtol=1e-6)
    j = math.ceil(round((1 - 0.01) * 256, 12))
    np.testing.assert_allclose(stats["p0.01"], np.sort(r, -1)[..., j - 1], rtol=1e-6)
    err[1, 2, 7] = np.nan
    w[2, 3, 5] = 0.0  # infinite band at that cell
    stats = rd.reduce_weight(err, w)
    assert math.isinf(stats["m"][1, 2]) and np.isfinite(stats["m"][2, 3])
    assert stats["m"].shape == (3, 5)


def test_w3_field_gives_infinite_bands_on_nonfinite_steps() -> None:
    raw = np.ones((2, 4, 32))
    raw[0, 2, 5] = np.inf
    w = rd.w3_field(raw, 3, 0.02, 1e-6)
    assert np.isposinf(w[0, 2]).all() and np.isfinite(w[0, [0, 1, 3]]).all()
    assert np.isfinite(w[1]).all()
