"""Generate Kuramoto-Sivashinsky (P3) trajectories for one seed stream.

u_t + u u_x + u_xx + nu4 u_xxxx = 0 on L = 32 pi; nu4 ~ U[0.9, 1.1] from the "mu" component key;
initial condition = Fourier series (modes 1-8, slope N(3, 0.5^2), sup norm 1) from the "ic" key, on
the 512-point solver grid, then a burn-in of B macro steps (A15, read from p3_burnin.json), then
T = 500 macro steps of dt = 0.2 (20 ETDRK4 substeps of 0.01, float64, CPU), projected to 256 points by
exact Fourier truncation and stored as float32. Trajectory i depends only on (stream, p3, i).

Writes <out>/<split>.npy (n, 501, 256), <out>/<split>_meta.parquet, <out>/<split>_settings.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from trajcp.data import burgers, ks
from trajcp.data.seeds import STREAMS

_spec = importlib.util.spec_from_file_location(
    "p3_burnin", Path(__file__).with_name("09_p3_burnin.py")
)
assert _spec is not None and _spec.loader is not None
p3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p3)

STEPS = 500


def summaries(u0: np.ndarray, extent: float) -> dict[str, float]:
    k = np.fft.rfftfreq(u0.size, d=1.0 / u0.size)
    power = np.abs(np.fft.rfft(u0)) ** 2
    grad = np.fft.irfft(2j * np.pi * k / extent * np.fft.rfft(u0), n=u0.size)
    return {
        "ic_l2": float(np.sqrt(np.mean(u0**2))),
        "ic_grad_l2": float(np.sqrt(np.mean(grad**2))),
        "ic_linf": float(np.abs(u0).max()),
        "ic_spec_centroid": float((k[1:] * power[1:]).sum() / power[1:].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stream", choices=["TRAIN", "DEV"], required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--burnin-report", type=Path, default=Path("artifacts/audit/p3/p3_burnin.json"))
    parser.add_argument("--setup-report", type=Path, default=Path("artifacts/audit/p3/p3_setup.json"))
    parser.add_argument("--batch", type=int, default=100)
    args = parser.parse_args()
    burn_report = json.loads(args.burnin_report.read_text())
    if not burn_report.get("audit_pass"):
        raise SystemExit("A15 burn-in or its resolution re-audit has not passed")
    burn = int(burn_report["stationarity"]["burn_in_macro_steps"])
    setup = json.loads(args.setup_report.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.split}.npy"
    if path.exists():
        raise SystemExit(f"{path} exists; never overwrite generated data")
    data = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(args.n, STEPS + 1, p3.EMULATOR)
    )
    rows = []
    start = time.perf_counter()
    for first in range(0, args.n, args.batch):
        idx = range(first, min(first + args.batch, args.n))
        draws = [p3.draw(args.stream, i) for i in idx]
        u0 = np.stack([d[0] for d in draws])
        nu4 = np.array([d[2] for d in draws])
        on = ks.advance_many(u0, nu4, p3.EXTENT, p3.H, burn * p3.SUB)
        u = ks.solve_many(on, nu4, p3.EXTENT, p3.H * p3.SUB, STEPS, p3.SUB)
        projected = burgers.project(u, p3.EMULATOR)
        if not np.isfinite(projected).all():
            raise SystemExit(f"non-finite reference solution in batch starting at {first}")
        data[first : first + len(idx)] = projected.astype(np.float32)
        for j, i in enumerate(idx):
            rows.append(
                {
                    "index": i,
                    "stream": args.stream,
                    "nu4": float(nu4[j]),
                    "slope": draws[j][1],
                    "energy_mean": float(np.mean(projected[j] ** 2)),
                    **summaries(projected[j, 0], p3.EXTENT),
                }
            )
        print(f"{first + len(idx)}/{args.n} {time.perf_counter() - start:.1f}s", flush=True)
    data.flush()
    meta = pl.DataFrame(rows)
    meta.write_parquet(args.out / f"{args.split}_meta.parquet")
    lam = setup["lambda_max"]["mean"]
    settings = {
        "split": args.split,
        "stream": args.stream,
        "master_seed": STREAMS[args.stream],
        "task": "p3",
        "n": args.n,
        "extent": p3.EXTENT,
        "macro_dt": p3.H * p3.SUB,
        "substeps": p3.SUB,
        "solver_step": p3.H,
        "steps": STEPS,
        "solver_grid": p3.SOLVER,
        "emulator_grid": p3.EMULATOR,
        "scheme": "ETDRK4 KuramotoSivashinskyConservative, 2/3 dealiasing",
        "nu4_range": [p3.NU4_LOW, p3.NU4_HIGH],
        "ic": "Fourier modes 1-8, slope N(3, 0.5^2), sup norm 1, then burn-in",
        "burn_in_macro_steps": burn,
        "burn_in_rule": burn_report["stationarity"]["rule"],
        "lambda_max": lam,
        "lambda_max_se": setup["lambda_max"]["se"],
        "tau_L": 1.0 / lam,
        "horizon_in_tau_L": STEPS * p3.H * p3.SUB * lam,
        "dtype_solver": "float64",
        "dtype_store": "float32",
        "versions": {p: md.version(p) for p in ("exponax", "jax", "numpy")},
        "seconds": time.perf_counter() - start,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checks": {
            "finite": True,
            "nu4_observed": [float(meta["nu4"].min()), float(meta["nu4"].max())],
            "max_abs_spatial_mean": float(np.abs(np.asarray(data).mean(-1)).max()),
            "energy_mean_range": [float(meta["energy_mean"].min()), float(meta["energy_mean"].max())],
        },
    }
    (args.out / f"{args.split}_settings.json").write_text(json.dumps(settings, indent=2))
    print(json.dumps({k: settings[k] for k in ("n", "seconds", "sha256", "burn_in_macro_steps")}))


if __name__ == "__main__":
    main()
