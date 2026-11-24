"""P3 burn-in under A15: stationarity on 100 fresh AUX trajectories, then the resolution re-audit.

Rules fixed in ARCHITECT.md A15 before this script ran:
  * AUX indices 400-499 (task p3), each with its own nu4 ~ U[0.9, 1.1] from the "mu" component key and
    a Fourier initial condition (modes 1-8, slope N(3, 0.5^2), sup norm 1) from the "ic" key, solved on
    512 points with solver step h = 0.01 and macro step 0.2 (20 substeps), for 1,600 + 500 macro steps.
  * Statistics: mean energy and six dyadic shell energies of the 256-point projection.
  * Long-run reference: ensemble mean over the final 500 macro steps (1,601..2,100).
  * At each burn-in length B in (200, 400, 800, 1600) the ensemble mean over macro steps B-9..B is
    compared with the reference.
  * Primary rule: every statistic within 5%; burn-in = the shortest passing B.
  * Fallback, used only if no B passes the primary rule: every statistic within max(5%, 2 SE), SE the
    standard error over the 100 trajectories of the per-trajectory difference (window mean minus
    long-run mean), relative to the reference; burn-in = the shortest passing B. Otherwise stop.
  * Resolution re-audit at the chosen B on AUX 500-519 (nu4 = 0.9) and 520-539 (nu4 = 1.1), grids
    512/1024/2048, criteria as in 07_p3_setup.py.
Writes <out>/p3_burnin.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import jax
import numpy as np

from trajcp.data import burgers, ks
from trajcp.data.seeds import component_key, trajectory_key

EXTENT, H, SUB, EMULATOR, SOLVER = 32 * math.pi, 0.01, 20, 256, 512
NU4_LOW, NU4_HIGH = 0.9, 1.1
LADDER, LONG, WINDOW = (200, 400, 800, 1600), 500, 10
SHELLS = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128)]


def draw(stream: str, index: int, points: int = SOLVER) -> tuple[np.ndarray, float, float]:
    """Initial condition (sup norm 1, before burn-in), its slope and nu4 for one trajectory."""
    key = trajectory_key(stream, "p3", index)
    u0, slope = burgers.fourier_ic(component_key(key, "ic"), points, 1.0, extent=EXTENT)
    unit = float(jax.random.uniform(component_key(key, "mu")))
    return u0, slope, NU4_LOW + unit * (NU4_HIGH - NU4_LOW)


def shell_energy(u: np.ndarray) -> np.ndarray:
    p = burgers.project(u.reshape(-1, u.shape[-1]), EMULATOR).reshape(*u.shape[:-1], EMULATOR)
    spec = np.abs(np.fft.rfft(p, axis=-1) / EMULATOR) ** 2
    shells = [spec[..., a : b + 1].sum(-1) for a, b in SHELLS]
    return np.stack([np.mean(p**2, axis=-1), *shells], axis=-1)


def pad(u: np.ndarray, points: int) -> np.ndarray:
    n = u.shape[-1]
    spec = np.fft.rfft(u, axis=-1)
    out = np.zeros((*u.shape[:-1], points // 2 + 1), dtype=complex)
    out[..., : spec.shape[-1] - 1] = spec[..., :-1]
    return np.fft.irfft(out, n=points, axis=-1) * (points / n)


def stationarity() -> dict[str, object]:
    t0 = time.perf_counter()
    draws = [draw("AUX", 400 + i) for i in range(100)]
    state = np.stack([d[0] for d in draws])
    nu4 = np.array([d[2] for d in draws])
    series = [shell_energy(state)]
    for _ in range(LADDER[-1] + LONG):
        state = ks.advance_many(state, nu4, EXTENT, H, SUB)
        series.append(shell_energy(state))
    s = np.stack(series)  # (2101, 100, 7)
    if not np.isfinite(s).all():
        raise SystemExit("non-finite energies in the stationarity run")
    long_traj = s[LADDER[-1] + 1 :].mean(axis=0)  # (100, 7)
    reference = long_traj.mean(axis=0)
    checks, primary, fallback = {}, None, None
    for burn in LADDER:
        window = s[burn - WINDOW + 1 : burn + 1].mean(axis=0)  # (100, 7)
        rel = np.abs(window.mean(0) - reference) / reference
        diff = window - long_traj
        se = diff.std(axis=0, ddof=1) / math.sqrt(diff.shape[0]) / reference
        tol = np.maximum(0.05, 2.0 * se)
        checks[str(burn)] = {
            "relative_difference": rel.tolist(),
            "relative_se": se.tolist(),
            "primary_pass": bool(rel.max() <= 0.05),
            "fallback_pass": bool((rel <= tol).all()),
        }
        if primary is None and checks[str(burn)]["primary_pass"]:
            primary = burn
        if fallback is None and checks[str(burn)]["fallback_pass"]:
            fallback = burn
    rule = "primary" if primary is not None else ("fallback" if fallback is not None else None)
    return {
        "aux_indices": [400, 499],
        "nu4_range": [float(nu4.min()), float(nu4.max())],
        "labels": ["energy", *map(str, SHELLS)],
        "reference": reference.tolist(),
        "checks": checks,
        "rule": rule,
        "burn_in_macro_steps": primary if primary is not None else fallback,
        "seconds": time.perf_counter() - t0,
    }


def audit(burn: int) -> list[dict[str, object]]:
    out = []
    for nu4, first in ((0.9, 500), (1.1, 520)):
        t0 = time.perf_counter()
        u0 = np.stack([draw("AUX", first + i)[0] for i in range(20)])
        base = ks.advance(u0, nu4, EXTENT, H, burn * SUB)
        proj, finest = {}, None
        for grid in (512, 1024, 2048):
            traj = ks.solve(base if grid == 512 else pad(base, grid), nu4, EXTENT, H * SUB, 500, SUB)
            proj[grid] = burgers.project(traj, EMULATOR)
            finest = traj
        scale = np.abs(finest).max(axis=(1, 2))
        pairs = {
            f"{a}-{b}": np.abs(proj[a] - proj[b]).max(-1) for a, b in ((512, 1024), (1024, 2048))
        }
        spec = np.abs(np.fft.rfft(proj[2048], axis=-1)[..., : EMULATOR // 2]) ** 2
        top = spec[..., EMULATOR // 2 - EMULATOR // 6 :].sum(-1) / spec[..., 1:].sum(-1)
        marks = [1, 10, 50, 100, 200, 300, 400, 500]
        out.append(
            {
                "nu4": nu4,
                "aux_indices": [first, first + 19],
                "seconds": time.perf_counter() - t0,
                "criterion_a_worst": {k: float((v.max(1) / scale).max()) for k, v in pairs.items()},
                "criterion_a_pass": {
                    k: bool((v.max(1) / scale).max() <= 1e-5) for k, v in pairs.items()
                },
                "top_third_energy_worst": float(top.max()),
                "criterion_b_energy_pass": bool(top.max() <= 1e-6),
                "error_growth_median": {
                    k: {str(t): float(np.median(v[:, t] / scale)) for t in marks}
                    for k, v in pairs.items()
                },
            }
        )
        print(json.dumps(out[-1]), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"decision": "A15", "macro_dt": H * SUB, "substeps": SUB}
    report["stationarity"] = st = stationarity()
    print(json.dumps(st), flush=True)
    (args.out / "p3_burnin.json").write_text(json.dumps(report, indent=2))
    if st["burn_in_macro_steps"] is None:
        raise SystemExit("A15: no burn-in length passes either rule; stop and report")
    report["resolution_audit"] = res = audit(int(st["burn_in_macro_steps"]))
    report["audit_pass"] = all(
        all(r["criterion_a_pass"].values()) and r["criterion_b_energy_pass"] for r in res
    )
    (args.out / "p3_burnin.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
