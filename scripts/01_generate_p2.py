"""Generate 1D Burgers (P2) trajectories for one seed stream.

Prior: unit periodic domain, macro step 0.01, T = 200; initial condition = random Fourier series
(modes 1-8, slope ~ N(3, 0.5^2)) rescaled to sup norm 0.5; viscosity log-uniform on
[1.75e-3, 5e-3]. Solver: ETDRK4 on 1,024 points (Courant 0.2, 2/3 dealiasing), float64 on the CPU,
exact Fourier projection to 256 points, stored as float32. Trajectory i uses the stateless key of
(stream, task p2, i), so it does not depend on batching; the initial condition and the viscosity
draw from separate component keys.

Writes <out>/<split>.npy (float32, shape (n, 201, 256)) and <out>/<split>_meta.parquet.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import time
from pathlib import Path

import jax
import numpy as np
import polars as pl

from trajcp.data import burgers
from trajcp.data.seeds import STREAMS, component_key, trajectory_key

EXTENT, MACRO_DT, STEPS, EMULATOR, GRID, COURANT = 1.0, 0.01, 200, 256, 1024, 0.2
AMPLITUDE, NU_LOW, NU_HIGH = 0.5, 1.75e-3, 5e-3


def draw(stream: str, index: int) -> tuple[np.ndarray, float, float]:
    key = trajectory_key(stream, "p2", index)
    u0, slope = burgers.fourier_ic(component_key(key, "ic"), GRID, AMPLITUDE)
    unit = float(jax.random.uniform(component_key(key, "mu")))
    nu = float(np.exp(np.log(NU_LOW) + unit * np.log(NU_HIGH / NU_LOW)))
    return u0, slope, nu


def summaries(u0: np.ndarray) -> dict[str, float]:
    """phi(u0) on the emulator grid: L2 norm, gradient L2 norm, sup norm, spectral centroid."""
    k = np.fft.rfftfreq(u0.size, d=1.0 / u0.size)
    power = np.abs(np.fft.rfft(u0)) ** 2
    grad = np.fft.irfft(2j * np.pi * k * np.fft.rfft(u0), n=u0.size)
    return {
        "ic_l2": float(np.sqrt(np.mean(u0**2))),
        "ic_grad_l2": float(np.sqrt(np.mean(grad**2))),
        "ic_linf": float(np.abs(u0).max()),
        "ic_spec_centroid": float((k[1:] * power[1:]).sum() / power[1:].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stream", choices=["TRAIN", "DEV", "AUX"], required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=100)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.split}.npy"
    if path.exists():
        raise SystemExit(f"{path} exists; never overwrite generated data")
    data = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(args.n, STEPS + 1, EMULATOR)
    )
    substeps = burgers.substeps_for(GRID, MACRO_DT, AMPLITUDE, COURANT)
    rows = []
    start = time.perf_counter()
    for first in range(0, args.n, args.batch):
        idx = range(first, min(first + args.batch, args.n))
        draws = [draw(args.stream, i) for i in idx]
        u = burgers.solve_many(
            np.stack([d[0] for d in draws]),
            np.array([d[2] for d in draws]),
            MACRO_DT,
            STEPS,
            substeps,
        )
        projected = burgers.project(u, EMULATOR)
        if not np.isfinite(projected).all():
            raise SystemExit(f"non-finite reference solution in batch starting at {first}")
        data[first : first + len(idx)] = projected.astype(np.float32)
        for j, i in enumerate(idx):
            rows.append(
                {
                    "index": i,
                    "stream": args.stream,
                    "nu": draws[j][2],
                    "log_nu": float(np.log(draws[j][2])),
                    "slope": draws[j][1],
                    "amplitude": AMPLITUDE,
                    **summaries(projected[j, 0]),
                }
            )
        print(f"{first + len(idx)}/{args.n} {time.perf_counter() - start:.1f}s", flush=True)
    data.flush()
    pl.DataFrame(rows).write_parquet(args.out / f"{args.split}_meta.parquet")
    settings = {
        "split": args.split,
        "stream": args.stream,
        "master_seed": STREAMS[args.stream],
        "task": "p2",
        "n": args.n,
        "extent": EXTENT,
        "macro_dt": MACRO_DT,
        "steps": STEPS,
        "solver_grid": GRID,
        "emulator_grid": EMULATOR,
        "substeps": substeps,
        "courant": COURANT,
        "scheme": "ETDRK4 conservative, 2/3 dealiasing",
        "amplitude": AMPLITUDE,
        "nu_range": [NU_LOW, NU_HIGH],
        "dtype_solver": "float64",
        "dtype_store": "float32",
        "versions": {p: md.version(p) for p in ("exponax", "jax", "numpy")},
        "seconds": time.perf_counter() - start,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (args.out / f"{args.split}_settings.json").write_text(json.dumps(settings, indent=2))
    print(json.dumps({k: settings[k] for k in ("n", "seconds", "sha256")}))


if __name__ == "__main__":
    main()
