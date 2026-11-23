"""P3 solver validation (IC-02 B3 step 2): sign conventions, then lambda_max at L = 22, nu4 = 1.

1. Linear growth: a small single mode grows at k^2 - nu4 k^4 under the Exponax stepper (sign of
   the u_xx and u_xxxx terms).
2. Nonlinearity: Exponax and an independent ETDRK4 of u_t = -u_xx - nu4 u_xxxx - u u_x agree on a
   short chaotic-free window (relative difference reported).
3. Benettin lambda_max at L = 22, nu4 = 1 (64 points); published value about 0.043, gate +-15%.
Tolerances fixed before the first run. Writes <out>/p3_validation.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from trajcp.data import ks


def linear_growth() -> dict[str, object]:
    extent, n, nu4, dt, steps = 22.0, 64, 1.0, 0.01, 200
    rows = []
    for mode in (1, 2, 3, 4, 5):
        k = 2 * np.pi * mode / extent
        x = np.arange(n) * extent / n
        u0 = 1e-8 * np.cos(k * x)
        u = ks.advance(u0[None], nu4, extent, dt, steps)[0]
        amp0 = np.abs(np.fft.rfft(u0)[mode])
        amp = np.abs(np.fft.rfft(u)[mode])
        measured = float(np.log(amp / amp0) / (dt * steps))
        expected = float(k**2 - nu4 * k**4)
        rows.append(
            {
                "mode": mode,
                "measured": measured,
                "expected": expected,
                "abs_diff": abs(measured - expected),
            }
        )
    return {"rows": rows, "pass": all(r["abs_diff"] < 1e-3 for r in rows)}


def nonlinear_agreement() -> dict[str, object]:
    extent, n, nu4, dt, steps = 22.0, 64, 1.0, 0.005, 400  # t = 2
    x = np.arange(n) * extent / n
    u0 = 0.5 * np.cos(2 * np.pi * x / extent) + 0.3 * np.sin(4 * np.pi * x / extent)
    a = ks.advance(u0[None], nu4, extent, dt, steps)[0]
    b = ks.etdrk4_reference(u0, nu4, extent, dt, steps)
    wrong = ks.etdrk4_reference(-u0, nu4, extent, dt, steps)  # sign-flipped control
    diff = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    return {
        "rel_diff": diff,
        "pass": diff < 1e-6,
        "control_sign_flip_rel_diff": float(
            np.linalg.norm(a + wrong) / np.linalg.norm(a)
        ),
    }


def benettin(
    extent: float,
    n: int,
    nu4: float,
    n_traj: int,
    burn: float,
    horizon: float,
    dt: float,
    renorm: float,
    seed_first: int,
    perturbation: float = 1e-8,
):
    from trajcp.data import burgers
    from trajcp.data.seeds import component_key, trajectory_key

    u0 = np.stack(
        [
            burgers.fourier_ic(
                component_key(trajectory_key("AUX", "p3", seed_first + i), "ic"),
                n,
                1.0,
                extent=extent,
            )[0]
            for i in range(n_traj)
        ]
    )
    sub = round(renorm / dt)
    u = ks.advance(u0, nu4, extent, dt, round(burn / dt))
    rng = np.random.default_rng(20261106 + seed_first)
    d = rng.standard_normal(u.shape)
    d *= perturbation / np.linalg.norm(d, axis=-1, keepdims=True)
    v = u + d
    total = np.zeros(n_traj)
    n_ren = round(horizon / renorm)
    for _ in range(n_ren):
        u = ks.advance(u, nu4, extent, dt, sub)
        v = ks.advance(v, nu4, extent, dt, sub)
        sep = np.linalg.norm(v - u, axis=-1)
        total += np.log(sep / perturbation)
        v = u + (v - u) * (perturbation / sep)[:, None]
    lam = total / (n_ren * renorm)
    return lam


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "linear_growth": linear_growth(),
        "nonlinear_agreement": nonlinear_agreement(),
    }
    print(json.dumps(report, indent=1), flush=True)
    start = time.perf_counter()
    lam = benettin(
        22.0, 64, 1.0, 50, burn=200.0, horizon=1000.0, dt=0.05, renorm=1.0, seed_first=0
    )
    mean, se = float(lam.mean()), float(lam.std(ddof=1) / np.sqrt(lam.size))
    report["lambda_max_L22"] = {
        "mean": mean,
        "se": se,
        "published": 0.043,
        "tolerance": 0.15,
        "pass": abs(mean - 0.043) <= 0.15 * 0.043,
        "trajectories": 50,
        "horizon": 1000.0,
        "seconds": time.perf_counter() - start,
    }
    (args.out / "p3_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["lambda_max_L22"]), flush=True)


if __name__ == "__main__":
    main()
