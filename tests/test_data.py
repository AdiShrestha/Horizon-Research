"""Reference-solution generator: seeds, initial conditions, batching and projection."""

from __future__ import annotations

import numpy as np
import pytest

from trajcp.data import burgers
from trajcp.data.seeds import trajectory_key


def test_confirmation_stream_is_refused() -> None:
    with pytest.raises(PermissionError):
        trajectory_key("CONFIRM", "p2", 0)


def test_keys_are_stateless_and_distinct() -> None:
    a = burgers.fourier_ic(trajectory_key("AUX", "p2", 3), 64, 1.0)[0]
    b = burgers.fourier_ic(trajectory_key("AUX", "p2", 3), 64, 1.0)[0]
    c = burgers.fourier_ic(trajectory_key("AUX", "p2", 4), 64, 1.0)[0]
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    assert np.abs(burgers.fourier_ic(trajectory_key("AUX", "p2", 3), 8192, 0.5)[0]).max() == (
        pytest.approx(0.5)
    )


def test_initial_condition_is_the_same_function_on_every_grid() -> None:
    fine = burgers.fourier_ic(trajectory_key("AUX", "p2", 1), 1024, 1.0)[0]
    coarse = burgers.fourier_ic(trajectory_key("AUX", "p2", 1), 256, 1.0)[0]
    np.testing.assert_allclose(fine[::4], coarse, rtol=0, atol=1e-13)
    np.testing.assert_allclose(burgers.project(fine[None], 256)[0], coarse, rtol=0, atol=1e-12)


def test_batched_viscosities_match_single_solves() -> None:
    keys = [trajectory_key("AUX", "p2", i) for i in range(3)]
    u0 = np.stack([burgers.fourier_ic(key, 128, 0.5)[0] for key in keys])
    nus = np.array([2e-3, 3e-3, 5e-3])
    many = burgers.solve_many(u0, nus, 0.01, 5, 4)
    for i, nu in enumerate(nus):
        single = burgers.solve(u0[i : i + 1], nu, 0.01, 5, 4)[0]
        np.testing.assert_allclose(many[i], single, atol=1e-12)
