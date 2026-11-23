"""SafeTensors resumable state: exact round trip of model and AdamW state, and no pickle."""

from __future__ import annotations

import torch

from trajcp.emulators.checkpoint import load_state, save_state
from trajcp.emulators.e1 import make_optimizer, scaled_peak_lr, train_epoch
from trajcp.emulators.training import Normalization


def _fixture():
    g = torch.Generator().manual_seed(0)
    u = torch.randn(4, 6, 32, generator=g)
    mu = torch.randn(4, generator=g)
    norm = Normalization(0.0, 1.0, 1.0, 0.0, 1.0)
    return u, mu, norm


def _small_model():
    from trajcp.emulators.fno import FNO

    torch.manual_seed(0)
    return FNO(1, 2, 1, width=8, modes=4, layers=2, cond_dim=1)


def test_scaled_peak_lr_follows_a06() -> None:
    assert scaled_peak_lr(64) == 1e-3
    assert abs(scaled_peak_lr(256) - 2e-3) < 1e-12
    assert scaled_peak_lr(1024) == 3e-3  # 4e-3 capped


def test_resume_is_identical_to_an_uninterrupted_run(tmp_path) -> None:
    u, mu, norm = _fixture()
    kw = {"total_epochs": 4, "seed": 3, "batch": 8, "peak_lr": 1e-3}

    straight, opt_s = _small_model(), None
    opt_s = make_optimizer(straight)
    for e in range(4):
        train_epoch(straight, opt_s, u, mu, norm, epoch=e, **kw)

    resumed = _small_model()
    opt_r = make_optimizer(resumed)
    for e in range(2):
        train_epoch(resumed, opt_r, u, mu, norm, epoch=e, **kw)
    save_state(tmp_path, resumed, opt_r, {"epoch": 2, "curves": {}})
    assert not list(tmp_path.glob("*.pt"))

    fresh = _small_model()
    opt_f = make_optimizer(fresh)
    meta = load_state(tmp_path, fresh, opt_f, torch.device("cpu"))
    assert meta is not None and meta["epoch"] == 2
    for e in range(2, 4):
        train_epoch(fresh, opt_f, u, mu, norm, epoch=e, **kw)
    for a, b in zip(straight.parameters(), fresh.parameters(), strict=True):
        assert torch.equal(a, b)
