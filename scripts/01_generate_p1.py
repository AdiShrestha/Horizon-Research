# Ported from the parallel Horizon workspace, commit a1500f9.
"""Generate linear advection-diffusion-dispersion (P1) trajectories for one seed stream.

Prior: unit periodic domain, macro step 0.01, T = 500; initial condition = random Fourier series
(modes 1-8, slope ~ N(3, 0.5^2)) rescaled to sup norm 1; c ~ U[0.5, 1.5], nu log-uniform on
[2e-4, 1e-3], delta log-uniform on [1e-5, 1e-4], drawn as one draw of three uniforms on the "mu"
component key. Solution: exact Fourier multiplier on the 256-point grid (the initial condition is
band-limited to mode 8, so this is exact up to round-off), float64, stored as float32.

Writes <out>/<split>.npy (float32, shape (n, 501, 256)), <out>/<split>_meta.parquet and
<out>/<split>_settings.json (including summary checks for the data card).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from trajcp.data import burgers, linear
from trajcp.data.seeds import STREAMS, component_key, trajectory_key

EXTENT, MACRO_DT, STEPS, EMULATOR, AMPLITUDE, DECAY_FLAG = 1.0, 0.01, 500, 256, 1.0, 0.1


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
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.split}.npy"
    if path.exists():
        raise SystemExit(f"{path} exists; never overwrite generated data")
    data = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(args.n, STEPS + 1, EMULATOR)
    )
    rows = []
    start = time.perf_counter()
    for i in range(args.n):
        key = trajectory_key(args.stream, "p1", i)
        u0, slope = burgers.fourier_ic(component_key(key, "ic"), EMULATOR, AMPLITUDE)
        c, nu, delta = linear.draw_parameters(component_key(key, "mu"))
        u = linear.exact(u0, c, nu, delta, MACRO_DT, STEPS, EXTENT)
        if not np.isfinite(u).all():
            raise SystemExit(f"non-finite solution at index {i}")
        data[i] = u.astype(np.float32)
        norms = np.linalg.norm(u, axis=-1)
        rows.append(
            {
                "index": i,
                "stream": args.stream,
                "c": c,
                "nu": nu,
                "log_nu": float(np.log(nu)),
                "delta": delta,
                "log_delta": float(np.log(delta)),
                "slope": slope,
                "amplitude": AMPLITUDE,
                "decay_mid": float(norms[STEPS // 2] / norms[0]),
                "decay_end": float(norms[-1] / norms[0]),
                **summaries(u0),
            }
        )
    data.flush()
    meta = pl.DataFrame(rows)
    meta.write_parquet(args.out / f"{args.split}_meta.parquet")
    params = ("c", "log_nu", "log_delta")
    spearman = {
        f"{a}~{b}": float(stats.spearmanr(meta[a], meta[b]).statistic)
        for j, a in enumerate(params)
        for b in [*params[j + 1 :], "slope", "ic_grad_l2", "ic_spec_centroid"]
    }
    checks = {
        "finite": True,
        "ranges": {p: [float(meta[p].min()), float(meta[p].max())] for p in ("c", "nu", "delta")},
        "slope_mean": float(meta["slope"].mean()),
        "slope_sd": float(meta["slope"].std()),
        "max_abs_spatial_mean": float(np.abs(np.asarray(data).mean(-1)).max()),
        "decay_end_min": float(meta["decay_end"].min()),
        "decay_end_median": float(meta["decay_end"].median()),
        "decay_mid_min": float(meta["decay_mid"].min()),
        "below_decay_flag": int((meta["decay_end"] < DECAY_FLAG).sum()),
        "spearman": spearman,
        "max_abs_spearman": max(abs(v) for v in spearman.values()),
    }
    settings = {
        "split": args.split,
        "stream": args.stream,
        "master_seed": STREAMS[args.stream],
        "task": "p1",
        "n": args.n,
        "extent": EXTENT,
        "macro_dt": MACRO_DT,
        "steps": STEPS,
        "emulator_grid": EMULATOR,
        "solver": "exact Fourier multiplier, float64",
        "amplitude": AMPLITUDE,
        "c_range": linear.C_RANGE,
        "nu_range": linear.NU_RANGE,
        "delta_range": linear.DELTA_RANGE,
        "dtype_store": "float32",
        "versions": {p: md.version(p) for p in ("jax", "numpy")},
        "seconds": time.perf_counter() - start,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checks": checks,
    }
    (args.out / f"{args.split}_settings.json").write_text(json.dumps(settings, indent=2))
    print(json.dumps({k: settings[k] for k in ("n", "seconds", "sha256", "checks")}, indent=2))


if __name__ == "__main__":
    main()
