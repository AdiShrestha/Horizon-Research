from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from trajcp.emulators.fno import FNO
from trajcp.emulators.training import (
    Normalization,
    atomic_safetensors,
    learning_rate,
    relative_l2,
    tail_criterion,
)


def test_learning_rate_schedule() -> None:
    kw = dict(peak=1e-3, floor=1e-6, warmup=5, total=500)
    assert learning_rate(0, **kw) == 2e-4
    assert abs(learning_rate(5, **kw) - 1e-3) < 1e-15
    assert abs(learning_rate(500, **kw) - 1e-6) < 1e-15
    values = [learning_rate(e, **kw) for e in range(5, 501)]
    assert all(a >= b for a, b in itertools.pairwise(values))


def test_tail_criterion_windows() -> None:
    epochs = list(range(5, 501, 5))
    flat = tail_criterion(epochs, [1.0] * len(epochs), 500)
    assert flat["pass"] and flat["relative_change"] == 0.0
    falling = [1.0 if e <= 450 else 0.9 for e in epochs]
    assert not tail_criterion(epochs, falling, 500)["pass"]
    assert not tail_criterion(epochs[:80], [1.0] * 80, 500)["pass"]


def test_normalization_and_loss() -> None:
    rng = np.random.default_rng(0)
    u = rng.normal(2.0, 3.0, (4, 11, 8))
    norm = Normalization.from_training(u, np.array([1.0, 2.0, 3.0, 4.0]))
    assert abs(norm.state_mean - u.mean()) < 1e-12
    assert abs(norm.increment_sd - np.diff(u, axis=1).std()) < 1e-12
    assert norm.mu_mean == 2.5
    a = torch.ones(3, 5)
    assert torch.allclose(relative_l2(2 * a, a), torch.ones(3))


def test_film_starts_as_identity_and_safetensors_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    plain = FNO(1, 2, 1, width=8, modes=4)
    torch.manual_seed(0)
    film = FNO(1, 2, 1, width=8, modes=4, cond_dim=1)
    x = torch.randn(3, 2, 32)
    assert torch.allclose(plain(x), film(x, torch.randn(3, 1)))
    atomic_safetensors(film, tmp_path / "m.safetensors", {"epoch": "1"})
    again = FNO(1, 2, 1, width=8, modes=4, cond_dim=1)
    again.load_state_dict(load_file(str(tmp_path / "m.safetensors")))
    c = torch.randn(3, 1)
    assert torch.equal(film(x, c), again(x, c))


def test_rollout_error_summary_sup_and_nonfinite() -> None:
    from trajcp.emulators.training import rollout_error_summary

    truth = np.ones((3, 10, 4)) * 2.0
    pred = truth.copy()
    pred[0, 0, 1] += 0.5
    pred[2, 9, 0] = np.nan
    s = rollout_error_summary(pred, truth)
    assert s["non_finite_trajectories"] == 1
    assert s["p90_sup"]["1"] > 0.0 and s["median_sup"]["1"] == 0.0
    assert s["p90_sup_rel"]["1"] <= 0.25
    assert np.isinf(s["p90_sup"]["10"])
