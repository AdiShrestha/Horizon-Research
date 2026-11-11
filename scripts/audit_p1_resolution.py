"""Resolution, representability and dynamics audit of the linear P1 reference solutions.

P1: u_t + c u_x = nu u_xx + delta u_xxx on the unit periodic domain, macro step 0.01, T = 500
(final time 5). For each corner (c, nu, delta) of a candidate prior box, 20 auxiliary-stream
initial conditions (random Fourier series, modes 1-8, slope ~ N(3, 0.5^2), sup norm 1; also the
S3 slope N(2.2, 0.5^2)) are solved exactly by Fourier multipliers on the 256-point grid and with
Exponax on grids 256, 512 and 1024 (projected to 256). Checks, tolerances fixed before the first
run (same as the P2 audit):

  (a) max_t ||P u_exponax(N) - u_exact||_inf / max_t ||u||_inf <= 1e-5 for every N, and
      between successive grids;
  (b) top-third spectral energy <= 1e-6 of the total and overshoot <= 1e-4 (against the exact
      solution evaluated on 4,096 points);
  decay: ||u_T||_2 / ||u_0||_2 flagged below 0.1.
Reported, not gated: the per-macro-step phase rotation |Im m(k)| dt of the highest initial mode
(advection plus dispersion) and its dispersive part alone, and the decay at T/2.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from trajcp.data import burgers, linear
from trajcp.data.seeds import component_key, trajectory_key

EMULATOR, FINE, GRIDS = 256, 4096, (256, 512, 1024)
MACRO_DT, STEPS, AMPLITUDE, MAX_MODE = 0.01, 500, 1.0, 8
TOL_CONVERGENCE, TOL_ENERGY, TOL_OVERSHOOT, DECAY_FLAG = 1e-5, 1e-6, 1e-4, 0.1


def initial_condition(index: int, points: int, slope_mean: float) -> np.ndarray:
    key = component_key(trajectory_key("AUX", "p1", index), "ic")
    return burgers.fourier_ic(key, points, AMPLITUDE, max_mode=MAX_MODE, slope_mean=slope_mean)[0]


def audit(c: float, nu: float, delta: float, n: int, slope_mean: float) -> dict[str, object]:
    conv, conv_pairs, top, over, end, mid = [], [], [], [], [], []
    for i in range(n):
        u0 = initial_condition(i, EMULATOR, slope_mean)
        truth = linear.exact(u0, c, nu, delta, MACRO_DT, STEPS)
        fine = linear.exact(initial_condition(i, FINE, slope_mean), c, nu, delta, MACRO_DT, STEPS)
        scale = np.abs(fine).max()
        solved = {
            g: burgers.project(
                linear.solve_exponax(
                    initial_condition(i, g, slope_mean), c, nu, delta, MACRO_DT, STEPS
                ),
                EMULATOR,
            )
            if g != EMULATOR
            else linear.solve_exponax(u0, c, nu, delta, MACRO_DT, STEPS)
            for g in GRIDS
        }
        conv.append(max(np.abs(solved[g] - truth).max() for g in GRIDS) / scale)
        conv_pairs.append(
            max(np.abs(solved[a] - solved[b]).max() for a, b in itertools.pairwise(GRIDS)) / scale
        )
        spectrum = np.abs(np.fft.rfft(truth, axis=-1)[..., : EMULATOR // 2]) ** 2
        top.append(
            float(
                (
                    spectrum[..., EMULATOR // 2 - EMULATOR // 6 :].sum(-1)
                    / spectrum[..., 1:].sum(-1)
                ).max()
            )
        )
        over.append(
            float(
                np.maximum(truth.max(-1) - fine.max(-1), fine.min(-1) - truth.min(-1)).max() / scale
            )
        )
        norms = np.linalg.norm(truth, axis=-1)
        end.append(float(norms[-1] / norms[0]))
        mid.append(float(norms[STEPS // 2] / norms[0]))
    k = 2.0 * np.pi * MAX_MODE
    return {
        "c": c,
        "nu": nu,
        "delta": delta,
        "slope_mean": slope_mean,
        "trajectories": n,
        "a_vs_exact_worst": max(conv),
        "a_between_grids_worst": max(conv_pairs),
        "a_pass": max(conv) <= TOL_CONVERGENCE and max(conv_pairs) <= TOL_CONVERGENCE,
        "top_third_energy_worst": max(top),
        "overshoot_worst": max(over),
        "b_pass": max(top) <= TOL_ENERGY and max(over) <= TOL_OVERSHOOT,
        "decay_end_min": min(end),
        "decay_end_median": float(np.median(end)),
        "decay_mid_min": min(mid),
        "decay_flag": min(end) < DECAY_FLAG,
        "phase_per_step_top_mode": abs(c * k + delta * k**3) * MACRO_DT,
        "dispersive_phase_per_step_top_mode": delta * k**3 * MACRO_DT,
        "damping_per_step_top_mode": nu * k**2 * MACRO_DT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--c", type=float, nargs="+", required=True)
    parser.add_argument("--nu", type=float, nargs="+", required=True)
    parser.add_argument("--delta", type=float, nargs="+", required=True)
    parser.add_argument("--slope-mean", type=float, nargs="+", default=[3.0])
    parser.add_argument("--trajectories", type=int, default=20)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for c, nu, delta, slope in itertools.product(args.c, args.nu, args.delta, args.slope_mean):
        row = audit(c, nu, delta, args.trajectories, slope)
        rows.append(row)
        print(
            f"c={c:g} nu={nu:g} delta={delta:g} slope={slope:g} "
            f"a={row['a_vs_exact_worst']:.1e}/{row['a_between_grids_worst']:.1e} "
            f"top={row['top_third_energy_worst']:.1e} over={row['overshoot_worst']:.1e} "
            f"decay={row['decay_end_min']:.3f} phase={row['phase_per_step_top_mode']:.3f}",
            flush=True,
        )
    (args.out / "audit.json").write_text(json.dumps(rows, indent=2, default=lambda o: o.item()))


if __name__ == "__main__":
    main()
