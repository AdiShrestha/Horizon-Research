"""Time the generation of 1D Burgers trajectories and project the storage of the 1D tasks.

Generates --trajectories trajectories from auxiliary-stream keys (indices starting at --first),
viscosity log-uniform on [--nu-low, --nu-high], fixed initial sup norm --amplitude, on solver grid
--grid, projected to 256 points. Compilation is done on a separate small batch before timing; the
timed runs are repeated. Writes a JSON report.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from trajcp.data import burgers
from trajcp.data.seeds import trajectory_key

MACRO_DT, COURANT, EMULATOR = 0.01, 0.2, 256
BYTES_PER_VALUE = 4  # float32 storage


def inputs(
    first: int, n: int, grid: int, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray]:
    u0, nus = [], []
    for i in range(first, first + n):
        key = trajectory_key("AUX", "p2", i)
        u0.append(burgers.fourier_ic(key, grid, args.amplitude)[0])
        draw = float(jax.random.uniform(jax.random.fold_in(key, 1)))
        nus.append(float(np.exp(np.log(args.nu_low) + draw * np.log(args.nu_high / args.nu_low))))
    return np.stack(u0), np.array(nus)


def storage() -> dict[str, object]:
    """float32 bytes per 1D task: D_train 1,000, development pool 1,500, confirmation pool 1,500,
    500 per shift set (S1-S4; P3 also S5)."""
    out: dict[str, object] = {}
    for task, steps, shifts in (("p1", 500, 4), ("p2", 200, 4), ("p3", 500, 5)):
        trajectories = 1000 + 1500 + 1500 + 500 * shifts
        size = trajectories * (steps + 1) * EMULATOR * BYTES_PER_VALUE
        out[task] = {
            "trajectories": trajectories,
            "steps": steps,
            "bytes": size,
            "GiB": size / 2**30,
        }
    out["total_GiB"] = sum(v["GiB"] for v in out.values() if isinstance(v, dict))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=1024)
    parser.add_argument("--nu-low", type=float, default=1.75e-3)
    parser.add_argument("--nu-high", type=float, default=5e-3)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--trajectories", type=int, default=100)
    parser.add_argument("--first", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    substeps = burgers.substeps_for(args.grid, MACRO_DT, args.amplitude, COURANT)

    u0, nus = inputs(args.first, args.trajectories, args.grid, args)
    burgers.solve_many(u0[:2], nus[:2], MACRO_DT, args.steps, substeps)  # compile once
    seconds = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        u = burgers.solve_many(u0, nus, MACRO_DT, args.steps, substeps)
        projected = burgers.project(u, EMULATOR).astype(np.float32)
        seconds.append(time.perf_counter() - start)
    report = {
        "grid": args.grid,
        "substeps": substeps,
        "trajectories": args.trajectories,
        "nu_range": [args.nu_low, args.nu_high],
        "amplitude": args.amplitude,
        "raw_seconds": seconds,
        "median_seconds": float(np.median(seconds)),
        "all_finite": bool(np.isfinite(projected).all()),
        "load_average": list(__import__("os").getloadavg()),
        "storage_projection": storage(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("median_seconds", "raw_seconds", "all_finite")}))
    print(json.dumps(report["storage_projection"]))


if __name__ == "__main__":
    main()
