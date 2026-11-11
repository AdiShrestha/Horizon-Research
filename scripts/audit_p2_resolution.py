"""Resolution and representability audit of the 1D Burgers reference solutions.

For each (viscosity, amplitude) configuration, solve the same initial conditions on solver grids
512, 1024, 2048 and 4096, project to the 256-point emulator grid and check, before any bulk
generation:

  (a) solver convergence: max_t ||P u(N) - P u(2N)||_inf / max_t ||u||_inf <= 1e-5;
  (b) representability on the emulator grid, using the finest solution:
      spectral energy in the top third of the 128 retained modes <= 1e-6 of the total, at every
      step; and overshoot of the projected field beyond the fine field's extrema <= 1e-4 of
      max_t ||u||_inf;
  decay: ||u_T||_2 / ||u_0||_2 and ||u_{T/2}||_2 / ||u_0||_2, flagged when the ratio at T is
      below 0.1 (trajectories near zero well before the horizon).

Tolerances were fixed before the first run. Initial conditions use auxiliary-stream keys only.
Macro step 0.01, T = 200 (final time 2) on the unit domain; ETDRK4 substeps from a Courant
limit of 0.2. Results: JSON and figures in --out.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from trajcp.data import burgers
from trajcp.data.seeds import component_key, trajectory_key

GRIDS = (512, 1024, 2048, 4096)
EMULATOR = 256
MACRO_DT, STEPS, COURANT = 0.01, 200, 0.2
TOL_CONVERGENCE, TOL_ENERGY, TOL_OVERSHOOT, DECAY_FLAG = 1e-5, 1e-6, 1e-4, 0.1


def initial_conditions(n: int, points: int, amplitude: float, first: int) -> np.ndarray:
    return np.stack(
        [
            burgers.fourier_ic(
                component_key(trajectory_key("AUX", "p2", first + i), "ic"), points, amplitude
            )[0]
            for i in range(n)
        ]
    )


def audit(nu: float, amplitude: float, n: int, first: int) -> dict[str, object]:
    projected: dict[int, np.ndarray] = {}
    timing: dict[int, float] = {}
    finest = None
    for grid in GRIDS:
        u0 = initial_conditions(n, grid, amplitude, first)
        substeps = burgers.substeps_for(grid, MACRO_DT, amplitude, COURANT)
        start = time.perf_counter()
        u = burgers.solve(u0, nu, MACRO_DT, STEPS, substeps)
        timing[grid] = time.perf_counter() - start
        projected[grid] = burgers.project(u, EMULATOR)
        if grid == GRIDS[-1]:
            finest = u
    assert finest is not None
    scale = np.abs(finest).max(axis=(1, 2))  # max_t ||u||_inf per trajectory
    convergence = {
        f"{a}-{b}": (np.abs(projected[a] - projected[b]).max(axis=(1, 2)) / scale).tolist()
        for a, b in itertools.pairwise(GRIDS)
    }
    spectrum = np.abs(np.fft.rfft(projected[GRIDS[-1]], axis=-1)[..., : EMULATOR // 2]) ** 2
    top = spectrum[..., EMULATOR // 2 - EMULATOR // 6 :].sum(-1) / spectrum[..., 1:].sum(-1)
    p = projected[GRIDS[-1]]
    over = np.maximum(p.max(-1) - finest.max(-1), finest.min(-1) - p.min(-1)).max(-1) / scale
    norms = np.linalg.norm(finest, axis=-1)
    decay_end = (norms[:, -1] / norms[:, 0]).tolist()
    worst_conv = {k: max(v) for k, v in convergence.items()}
    return {
        "nu": nu,
        "amplitude": amplitude,
        "trajectories": n,
        "substeps": {g: burgers.substeps_for(g, MACRO_DT, amplitude, COURANT) for g in GRIDS},
        "solve_seconds": timing,
        "convergence_per_trajectory": convergence,
        "convergence_worst": worst_conv,
        "criterion_a_pass_by_pair": {k: v <= TOL_CONVERGENCE for k, v in worst_conv.items()},
        "top_third_energy_worst": float(top.max()),
        "overshoot_worst": float(over.max()),
        "criterion_b_pass": bool(top.max() <= TOL_ENERGY and over.max() <= TOL_OVERSHOOT),
        "decay_ratio_end": decay_end,
        "decay_ratio_mid": (norms[:, STEPS // 2] / norms[:, 0]).tolist(),
        "decay_flag": bool(min(decay_end) < DECAY_FLAG),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nu", type=float, nargs="+", default=[5e-4, 1e-3, 2e-3, 5e-3])
    parser.add_argument("--amplitude", type=float, nargs="+", default=[1.0])
    parser.add_argument("--trajectories", type=int, default=20)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for amplitude in args.amplitude:
        for nu in args.nu:
            print(f"nu={nu:g} amplitude={amplitude:g}", flush=True)
            results.append(audit(nu, amplitude, args.trajectories, first=0))
            (args.out / "audit.json").write_text(json.dumps(results, indent=2))
            r = results[-1]
            print(
                f"  conv {r['convergence_worst']}  top {r['top_third_energy_worst']:.2e}"
                f"  over {r['overshoot_worst']:.2e}  decay min {min(r['decay_ratio_end']):.3f}"
                f"  seconds {r['solve_seconds']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
