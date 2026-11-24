"""Stage 1 for a task pool (P1-P3; tier runs from 02_train.py or the engineering E1 run): roll out one emulator, reduce to sufficient statistics (IC-02 WP-E).

Writes <out>/stats.parquet, one row per trajectory with fixed-size float32 arrays of length T:
M, u_l2, e_l2 and, for f in {w0, w3}: m_f, p{gamma}_f, wbar_f, wl2_f; plus index, nu, log_nu, phi(u0)
columns and the perturbation-key identity (stream, task, index; key = component_key(
trajectory_key(stream, task, index), "perturbation")). With --fields the error field and the default
W3 field are stored as float32 NPY under <out>/fields/. With --variants (E1 seed 0 only) the W3
ablation variants are written to <out>/variants/<name>.parquet. <out>/stats_meta.json records the
command, interpreter, code commit, device, timing and blow-up counts.
Default W3 (A09): M = 8 low-mode unit-RMS perturbations, eps_p = 1e-3 rms(u0), finite differences,
smoothing sigma 3, relative floor 0.02 of the per-step median, absolute floor 1e-6 x the training
field SD (normalization state_sd of the run).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

from trajcp.emulators import device as dev  # noqa: E402
from trajcp.emulators import tiers  # noqa: E402
from trajcp.rollout import reduce as rd  # noqa: E402
from trajcp.rollout import runner  # noqa: E402

DEFAULT = {"s": 3, "eta": 0.02, "M": 8, "eps": 1e-3, "kind": "lowmode"}
GRID_S, GRID_ETA = (0, 1, 2, 3), (0.02, 0.05, 0.10)


def git_commit() -> dict[str, object]:
    root = Path(__file__).resolve().parent.parent

    def run(*c: str) -> str:
        return subprocess.run(
            ["git", *c], cwd=root, capture_output=True, text=True
        ).stdout

    return {
        "commit": run("rev-parse", "HEAD").strip(),
        "dirty": bool(
            run(
                "status", "--porcelain", "--untracked-files=no", "src", "scripts"
            ).strip()
        ),
    }


def frame(
    columns: dict[str, np.ndarray], scalars: dict[str, np.ndarray]
) -> pl.DataFrame:
    data = {k: pl.Series(k, v) for k, v in scalars.items()}
    for name, arr in columns.items():
        data[name] = pl.Series(name, arr, dtype=pl.Array(pl.Float32, arr.shape[1]))
    return pl.DataFrame(data)


def stat_columns(prefix: str, stats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{k}_{prefix}": v for k, v in stats.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None, help="default: the task horizon")
    parser.add_argument("--fields", action="store_true")
    parser.add_argument("--variants", action="store_true")
    parser.add_argument("--weights", default="best.safetensors")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev.configure(args.device)
    model, norm, settings = runner.load_emulator(args.run, args.device, args.weights)
    task = settings.get("task", "p2")
    raw = np.load(args.data / f"{args.split}.npy", mmap_mode="r")
    meta = pl.read_parquet(args.data / f"{args.split}_meta.parquet").sort("index")
    n = args.limit or raw.shape[0]
    steps = args.steps or (tiers.TASKS[task].steps if "tier" in settings else 200)
    points = raw.shape[-1]
    stream = meta["stream"][0]
    floor = 1e-6 * norm.state_sd
    if isinstance(norm, tiers.CondNormalization):
        mu_all = norm.cond(np.stack([meta[c].to_numpy() for c in settings["cond"]], axis=1))
    else:
        mu_all = (meta["log_nu"].to_numpy() - norm.mu_mean) / norm.mu_sd
    err_store = w3_store = None
    if args.fields:
        (args.out / "fields").mkdir(exist_ok=True)
        err_store = np.lib.format.open_memmap(
            args.out / "fields/error.npy", "w+", np.float32, (n, steps, points)
        )
        w3_store = np.lib.format.open_memmap(
            args.out / "fields/w3_default.npy", "w+", np.float32, (n, steps, points)
        )
    cols: dict[str, list[np.ndarray]] = {}
    var_cols: dict[str, dict[str, list[np.ndarray]]] = {}
    blow = {"rollout_nonfinite": 0, "w3_inf_step_fraction_sum": 0.0}
    start = time.perf_counter()

    def push(store: dict[str, list[np.ndarray]], items: dict[str, np.ndarray]) -> None:
        for k, v in items.items():
            store.setdefault(k, []).append(v)

    variants = {
        f"w3_s{s}_eta{eta:.2f}": (s, eta, 8, 1e-3, "lowmode")
        for s in GRID_S
        for eta in GRID_ETA
    }
    if args.variants:
        variants |= {
            "w3_M4": (3, 0.02, 4, 1e-3, "lowmode"),
            "w3_M16": (3, 0.02, 16, 1e-3, "lowmode"),
            "w3_eps1e-2": (3, 0.02, 8, 1e-2, "lowmode"),
            "w3_iso": (3, 0.02, 8, 1e-3, "iso"),
        }
    for first in range(0, n, args.batch):
        idx = np.arange(first, min(first + args.batch, n))
        u = np.asarray(raw[idx], dtype=np.float32)
        mu = mu_all[idx]
        pred = runner.rollout(model, u[:, 0], mu, norm, steps, args.device, args.batch)
        truth = u[:, 1 : steps + 1]
        err = pred.astype(np.float64) - truth.astype(np.float64)
        blow["rollout_nonfinite"] += int((~np.isfinite(pred).all(axis=(1, 2))).sum())
        push(cols, rd.base_statistics(truth, pred))
        push(cols, stat_columns("w0", rd.reduce_weight(err, None)))
        # sensitivity: one raw field per (M, eps, kind) setting, shared by the (s, eta) grid
        raw_cache: dict[tuple, np.ndarray] = {}

        def raw_field(m_count: int, eps: float, kind: str) -> np.ndarray:
            key = (m_count, eps, kind)
            if key not in raw_cache:
                xi = runner.perturbation_fields(
                    stream, task, idx, m_count, points, kind=kind
                )
                pert = runner.perturbed_initials(u[:, 0], xi, eps)
                flat = runner.rollout(
                    model,
                    pert.reshape(-1, points),
                    np.repeat(mu, m_count, axis=0),
                    norm,
                    steps,
                    args.device,
                    args.batch,
                )
                raw_cache[key] = rd.raw_sensitivity_nonfinite(
                    flat.reshape(len(idx), m_count, steps, points), pred, eps
                )
            return raw_cache[key]

        w3 = rd.w3_field(
            raw_field(8, 1e-3, "lowmode"), DEFAULT["s"], DEFAULT["eta"], floor
        )
        blow["w3_inf_step_fraction_sum"] += float(np.isposinf(w3[..., 0]).mean()) * len(
            idx
        )
        push(cols, stat_columns("w3", rd.reduce_weight(err, w3)))
        if err_store is not None and w3_store is not None:
            err_store[idx] = err.astype(np.float32)
            w3_store[idx] = w3.astype(np.float32)
        if args.variants:
            for name, (s, eta, m_count, eps, kind) in variants.items():
                w = rd.w3_field(raw_field(m_count, eps, kind), s, eta, floor)
                push(var_cols.setdefault(name, {}), rd.reduce_weight(err, w))
            jraw = runner.jvp_sensitivity(
                model,
                u[:, 0],
                mu,
                runner.perturbation_fields(stream, task, idx, 8, points),
                norm,
                steps,
                args.device,
            )
            push(
                var_cols.setdefault("w3_jvp", {}),
                rd.reduce_weight(err, rd.w3_field(jraw, 3, 0.02, floor)),
            )
        print(f"{idx[-1] + 1}/{n} {time.perf_counter() - start:.0f}s", flush=True)

    sel = slice(0, n)
    scalars = {
        "index": meta["index"].to_numpy()[sel],
        "stream": meta["stream"].to_numpy()[sel],
        **{
            c: meta[c].to_numpy()[sel]
            for c in ("nu", "log_nu", "c", "delta", "log_delta", "nu4")
            if c in meta.columns
        },
        **{
            c: meta[c].to_numpy()[sel]
            for c in ("ic_l2", "ic_grad_l2", "ic_linf", "ic_spec_centroid")
        },
    }
    pl_cols = {k: np.concatenate(v) for k, v in cols.items()}
    frame(pl_cols, scalars).write_parquet(args.out / "stats.parquet")
    if args.variants:
        (args.out / "variants").mkdir(exist_ok=True)
        for name, store in var_cols.items():
            frame(
                {f"{k}": np.concatenate(v) for k, v in store.items()},
                {"index": scalars["index"]},
            ).write_parquet(args.out / "variants" / f"{name}.parquet")
    if err_store is not None and w3_store is not None:
        err_store.flush()
        w3_store.flush()
    meta_out = {
        "command": " ".join(sys.argv),
        "interpreter": sys.executable,
        "code": git_commit(),
        "device": args.device,
        "task": task,
        "tier": settings.get("tier", "E1"),
        "hardware": dev.describe(torch.device(args.device)),
        "torch": torch.__version__,
        "run": str(args.run),
        "weights": args.weights,
        "run_data_sha256": settings.get("data_sha256"),
        "pool": str(args.data / f"{args.split}.npy"),
        "trajectories": n,
        "steps": steps,
        "seconds": time.perf_counter() - start,
        "rollout_nonfinite_trajectories": blow["rollout_nonfinite"],
        "w3_inf_step_fraction": blow["w3_inf_step_fraction_sum"] / n,
        "w3_default": DEFAULT
        | {"absolute_floor": floor, "floor_source": "1e-6 x normalization.state_sd"},
        "variants": sorted(variants) + (["w3_jvp"] if args.variants else []),
    }
    (args.out / "stats_meta.json").write_text(json.dumps(meta_out, indent=2))
    print(
        json.dumps(
            {
                k: meta_out[k]
                for k in ("trajectories", "seconds", "rollout_nonfinite_trajectories")
            }
        )
    )


if __name__ == "__main__":
    main()
