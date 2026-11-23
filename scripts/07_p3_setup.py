"""P3 (Kuramoto-Sivashinsky) setup under IC-02 B3: lambda_max, macro step, burn-in, resolution audit.

L = 32 pi, nu4 in U[0.9, 1.1], emulator grid 256, solver grid 512 (ETDRK4, float64, CPU), solver step
h = 0.01 on every grid. AUX trajectory indices (task p3, one operation each, A02):
  0-49 lambda_max at L = 22 (06_p3_validate.py); 100-149 lambda_max at L = 32 pi, nu4 = 1;
  200-219 stationarity; 300-319 audit at nu4 = 0.9; 320-339 audit at nu4 = 1.1.
Rules fixed before the first run:
  * lambda_max: Benettin, 50 trajectories, 200 time-unit provisional burn-in, 1,000 time units.
  * macro step: dt = n h with n = round(9 / (500 lambda_max) / h); tau_L = 1 / lambda_max.
  * burn-in B in {200, 400, 800} macro steps, the smallest passing: the mean energy and each of the
    dyadic shell energies (modes 1-4, 5-8, 9-16, 17-32, 33-64, 65-128 of the 256 grid), averaged over
    the last 10 macro steps of the burn-in and 20 trajectories, are within 5% of their long-run
    value (20 trajectories x 1,000 time units after step 800).
  * resolution audit on 512, 1024, 2048 points from the same post-burn-in state: criterion (a)
    max_t ||P u_N - P u_2N||_inf / max_t ||u||_inf <= 1e-5, criterion (b) top-third spectral energy
    <= 1e-6 of the total; error growth over t reported.
Writes <out>/p3_setup.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from trajcp.data import burgers, ks
from trajcp.data.seeds import component_key, trajectory_key

EXTENT, H, EMULATOR = 32 * math.pi, 0.01, 256
SHELLS = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128)]


def initial(first: int, n: int, points: int) -> np.ndarray:
    return np.stack(
        [
            burgers.fourier_ic(
                component_key(trajectory_key("AUX", "p3", first + i), "ic"),
                points,
                1.0,
                extent=EXTENT,
            )[0]
            for i in range(n)
        ]
    )


def pad(u: np.ndarray, points: int) -> np.ndarray:
    """Exact Fourier interpolation to more points (zero-padded spectrum), amplitude preserved."""
    n = u.shape[-1]
    spec = np.fft.rfft(u, axis=-1)
    out = np.zeros((*u.shape[:-1], points // 2 + 1), dtype=complex)
    out[..., : spec.shape[-1] - 1] = spec[..., :-1]
    return np.fft.irfft(out, n=points, axis=-1) * (points / n)


def shell_energy(u: np.ndarray) -> np.ndarray:
    """(..., 1 + len(SHELLS)): mean energy then the shell energies of the 256-point projection."""
    p = burgers.project(u.reshape(-1, u.shape[-1]), EMULATOR).reshape(
        *u.shape[:-1], EMULATOR
    )
    spec = np.abs(np.fft.rfft(p, axis=-1) / EMULATOR) ** 2
    shells = [spec[..., a : b + 1].sum(-1) for a, b in SHELLS]
    return np.stack([np.mean(p**2, axis=-1), *shells], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"extent": EXTENT, "solver_step": H}

    # lambda_max
    t0 = time.perf_counter()
    u0 = initial(100, 50, 512)
    u = ks.advance(u0, 1.0, EXTENT, H, round(200.0 / H))
    lam = ks.benettin_lambda(u, 1.0, EXTENT, H, horizon=1000.0, seed=20261106 + 100)
    mean, se = float(lam.mean()), float(lam.std(ddof=1) / np.sqrt(lam.size))
    n_sub = max(1, round(9.0 / (500.0 * mean) / H))
    macro = n_sub * H
    report["lambda_max"] = {
        "mean": mean,
        "se": se,
        "trajectories": 50,
        "horizon": 1000.0,
        "burn_in_time_units": 200.0,
        "seconds": time.perf_counter() - t0,
    }
    report["tau_L"] = 1.0 / mean
    report["macro_dt"] = macro
    report["substeps"] = n_sub
    report["horizon_in_tau_L"] = 500 * macro * mean
    print(
        json.dumps(
            {k: report[k] for k in ("lambda_max", "tau_L", "macro_dt", "substeps")}
        ),
        flush=True,
    )

    # stationarity
    t0 = time.perf_counter()
    state = ks.advance(
        initial(200, 20, 512), 1.0, EXTENT, macro / n_sub, 0
    )  # no-op copy
    series = [shell_energy(state)]
    stepper_state = state
    long_steps = round(1000.0 / macro)
    for _ in range(800 + long_steps):
        stepper_state = ks.advance(stepper_state, 1.0, EXTENT, H, n_sub)
        series.append(shell_energy(stepper_state))
    series_a = np.stack(series)  # (steps + 1, 20, 7)
    long_run = series_a[800 : 800 + long_steps + 1].mean(axis=(0, 1))
    checks = {}
    chosen = None
    for burn in (200, 400, 800):
        end = series_a[burn - 10 : burn].mean(axis=(0, 1))
        rel = np.abs(end - long_run) / long_run
        checks[str(burn)] = {
            "relative_difference": rel.tolist(),
            "pass": bool(rel.max() <= 0.05),
        }
        if chosen is None and checks[str(burn)]["pass"]:
            chosen = burn
    report["stationarity"] = {
        "labels": ["energy", *map(str, SHELLS)],
        "long_run": long_run.tolist(),
        "checks": checks,
        "burn_in_macro_steps": chosen,
        "seconds": time.perf_counter() - t0,
    }
    print(json.dumps(report["stationarity"]), flush=True)
    burn = chosen or 800

    # resolution audit
    audit = []
    for nu4, first in ((0.9, 300), (1.1, 320)):
        t0 = time.perf_counter()
        base = ks.advance(initial(first, 20, 512), nu4, EXTENT, H, burn * n_sub)
        proj, finest = {}, None
        for grid in (512, 1024, 2048):
            start = base if grid == 512 else pad(base, grid)
            traj = ks.solve(start, nu4, EXTENT, macro, 500, n_sub)
            proj[grid] = burgers.project(traj, EMULATOR)
            if grid == 2048:
                finest = traj
        scale = np.abs(finest).max(axis=(1, 2))
        pairs = {
            f"{a}-{b}": np.abs(proj[a] - proj[b]).max(-1)
            for a, b in ((512, 1024), (1024, 2048))
        }
        spec = np.abs(np.fft.rfft(proj[2048], axis=-1)[..., : EMULATOR // 2]) ** 2
        top = spec[..., EMULATOR // 2 - EMULATOR // 6 :].sum(-1) / spec[..., 1:].sum(-1)
        marks = [1, 10, 50, 100, 200, 300, 400, 500]
        audit.append(
            {
                "nu4": nu4,
                "trajectories": 20,
                "seconds": time.perf_counter() - t0,
                "criterion_a_worst": {
                    k: float((v.max(1) / scale).max()) for k, v in pairs.items()
                },
                "criterion_a_pass": {
                    k: bool((v.max(1) / scale).max() <= 1e-5) for k, v in pairs.items()
                },
                "top_third_energy_worst": float(top.max()),
                "criterion_b_energy_pass": bool(top.max() <= 1e-6),
                "error_growth_median_over_trajectories": {
                    k: {str(t): float(np.median(v[:, t] / scale)) for t in marks}
                    for k, v in pairs.items()
                },
            }
        )
        print(json.dumps(audit[-1]), flush=True)
    report["resolution_audit"] = audit
    (args.out / "p3_setup.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
