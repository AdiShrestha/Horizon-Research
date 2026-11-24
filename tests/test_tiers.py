"""Tier trainer (A18): E1 equals the engineering E1 loop on P2; E2 and E3 run and resume."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("safetensors")

from trajcp.emulators import e1, tiers  # noqa: E402
from trajcp.emulators.training import Normalization  # noqa: E402


def _toy(n: int = 6, steps: int = 12, points: int = 32, k: int = 1, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, points, endpoint=False)
    u = np.stack(
        [
            np.stack([np.sin(x + 0.1 * t + i) * np.exp(-0.01 * t) for t in range(steps + 1)])
            for i in range(n)
        ]
    ).astype(np.float32)
    u += 0.01 * rng.standard_normal(u.shape).astype(np.float32)
    cond = rng.uniform(-1, 1, (n, k))
    return u, cond


def test_e1_matches_engineering_loop_first_20_steps():
    u, cond = _toy(n=8, steps=40, points=64)
    norm_old = Normalization.from_training(u, cond[:, 0])
    norm_new = tiers.CondNormalization.from_training(u, cond)
    assert norm_new.state_mean == norm_old.state_mean
    assert norm_new.increment_sd == norm_old.increment_sd
    mu_old = torch.tensor((cond[:, 0] - norm_old.mu_mean) / norm_old.mu_sd, dtype=torch.float32)
    c_new = torch.tensor(norm_new.cond(cond), dtype=torch.float32)
    assert torch.equal(mu_old[:, None], c_new)
    ut = torch.from_numpy(u)

    losses_old, losses_new = [], []
    torch.manual_seed(3)
    m_old = e1.build_model()
    o_old = e1.make_optimizer(m_old)
    torch.manual_seed(3)
    m_new = tiers.build("E1", 1)
    o_new = tiers.make_optimizer(m_new, tiers.PEAK_LR["E1"])
    e1.train_epoch(
        m_old, o_old, ut, mu_old, norm_old, epoch=0, total_epochs=500, seed=3, batch=8,
        peak_lr=1e-3, on_step=lambda i, v: losses_old.append(float(v)),
        stop=lambda: len(losses_old) >= 20,
    )
    tiers.train_epoch(
        "E1", m_new, o_new, ut, c_new, norm_new, epoch=0, total_epochs=500, seed=3,
        peak_lr=1e-3, batch=8, on_step=lambda i, v: losses_new.append(float(v)),
        stop=lambda: len(losses_new) >= 20,
    )
    assert len(losses_old) == len(losses_new) == 20
    assert losses_old == losses_new
    for a, b in zip(m_old.state_dict().values(), m_new.state_dict().values(), strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("tier", ["E2", "E3"])
def test_other_tiers_train_and_are_deterministic(tier):
    u, cond = _toy(n=6, steps=24, points=32, k=3)
    norm = tiers.CondNormalization.from_training(u, cond)
    ut = torch.from_numpy(u)
    c = torch.tensor(norm.cond(cond), dtype=torch.float32)
    runs = []
    for _ in range(2):
        torch.manual_seed(1)
        model = tiers.build(tier, 3)
        opt = tiers.make_optimizer(model, tiers.PEAK_LR[tier])
        out = []
        for epoch in range(2):
            loss, steps, phase = tiers.train_epoch(
                tier, model, opt, ut, c, norm, epoch=epoch, total_epochs=2, seed=1,
                peak_lr=tiers.PEAK_LR[tier], batch=16,
            )
            out.append((loss, steps, phase))
        runs.append(out)
    assert runs[0] == runs[1]
    assert all(np.isfinite(r[0]) for r in runs[0])
    if tier == "E3":
        # 10% of 2 epochs rounds to 0: both epochs unroll, N_pairs // m starts
        assert runs[0][0][2] == "unroll"
        assert runs[0][0][1] == -(-(6 * 24 // tiers.E3_UNROLL) // 16)
    else:
        assert runs[0][0][2] == "pushforward"
        assert runs[0][0][1] == -(-(6 * (24 - tiers.E2_PUSHFORWARD + 1)) // 16)


def test_e3_warm_phase_is_one_step():
    u, cond = _toy(n=4, steps=20, points=32)
    norm = tiers.CondNormalization.from_training(u, cond)
    model = tiers.build("E3", 1)
    opt = tiers.make_optimizer(model, 1e-3)
    _, _, phase0 = tiers.train_epoch(
        "E3", model, opt, torch.from_numpy(u), torch.tensor(norm.cond(cond), dtype=torch.float32),
        norm, epoch=0, total_epochs=100, seed=0, peak_lr=1e-3, batch=16,
    )
    _, _, phase10 = tiers.train_epoch(
        "E3", model, opt, torch.from_numpy(u), torch.tensor(norm.cond(cond), dtype=torch.float32),
        norm, epoch=10, total_epochs=100, seed=0, peak_lr=1e-3, batch=16,
    )
    assert (phase0, phase10) == ("one-step", "unroll")


def test_unet_size_and_shape():
    model = tiers.build("E2", 3)
    params = sum(p.numel() for p in model.parameters())
    assert 200_000 <= params <= 800_000
    y = model(torch.randn(2, 4, 256), torch.randn(2, 3))
    assert y.shape == (2, 1, 256)


def test_growth_slope_recovers_power_law():
    t = np.arange(1, 201, dtype=float)
    assert tiers.growth_slope(0.01 * t**1.5) == pytest.approx(1.5)


def test_aux_tiers_train_and_wang_labels_are_fixed():
    u, cond = _toy(n=6, steps=20, points=32)
    norm = tiers.CondNormalization.from_training(u, cond)
    ut = torch.from_numpy(u)
    c = torch.tensor(norm.cond(cond), dtype=torch.float32)
    a = tiers.wang_labels(ut, 0, norm.state_sd)
    b = tiers.wang_labels(ut, 0, norm.state_sd)
    assert torch.equal(a, b)
    rel = float((a - ut).std() / norm.state_sd)
    assert 0.04 < rel < 0.06
    for tier, labels in (("E1D", None), ("E1W", a)):
        torch.manual_seed(0)
        model = tiers.build(tier, 1)
        opt = tiers.make_optimizer(model, 1e-3)
        loss, _, phase = tiers.train_epoch(
            tier, model, opt, ut, c, norm, epoch=0, total_epochs=10, seed=0, peak_lr=1e-3,
            batch=16, labels=labels,
        )
        assert phase == "one-step" and np.isfinite(loss)


def test_dropout_zero_fno_is_the_e1_model():
    torch.manual_seed(5)
    a = tiers.build("E1", 2)
    torch.manual_seed(5)
    b = tiers.build("E1D", 2)
    for x, y in zip(a.state_dict().values(), b.state_dict().values(), strict=True):
        assert torch.equal(x, y)
    b.eval()
    x = torch.randn(3, 3, 64)
    cc = torch.randn(3, 2)
    assert torch.equal(a(x, cc), b(x, cc))
