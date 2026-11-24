"""G1(iii) W3 cost on the production rollout runner (A09; A18).

For one trained run and its DEV pool: a batch of B trajectories (default 64) is rolled out once
(base) and the default W3 field is computed (M = 8 low-mode perturbations, eps 1e-3, finite
differences, smoothing 3, floors per A09), on the same device, model, batch and horizon. After two
warm-ups, seven paired repetitions are timed (base, then W3, alternating order); the decision value
is the median of the seven W3 / base ratios. G1(iii) passes if it is <= 10. Run with no other job on
the device. Writes <out>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from trajcp.emulators import device as dev
from trajcp.emulators import tiers
from trajcp.rollout import reduce as rd
from trajcp.rollout import runner


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dev.configure(args.device)
    model, norm, settings = runner.load_emulator(args.run, args.device)
    task = settings.get("task", "p2")
    steps = tiers.TASKS[task].steps
    raw = np.load(args.data / "dev.npy", mmap_mode="r")
    meta = pl.read_parquet(args.data / "dev_meta.parquet").sort("index")
    idx = np.arange(args.batch)
    u0 = np.asarray(raw[idx, 0], dtype=np.float32)
    if isinstance(norm, tiers.CondNormalization):
        mu = norm.cond(np.stack([meta[c].to_numpy() for c in settings["cond"]], axis=1))[idx]
    else:
        mu = ((meta["log_nu"].to_numpy() - norm.mu_mean) / norm.mu_sd)[idx]
    points, stream, floor = raw.shape[-1], meta["stream"][0], 1e-6 * norm.state_sd
    xi = runner.perturbation_fields(stream, task, idx, 8, points)

    def base() -> np.ndarray:
        out = runner.rollout(model, u0, mu, norm, steps, args.device, args.batch)
        sync(args.device)
        return out

    def w3(pred: np.ndarray) -> np.ndarray:
        pert = runner.perturbed_initials(u0, xi, 1e-3)
        flat = runner.rollout(
            model, pert.reshape(-1, points), np.repeat(mu, 8, axis=0), norm, steps,
            args.device, args.batch * 8,
        )
        sync(args.device)
        raw_s = rd.raw_sensitivity_nonfinite(flat.reshape(len(idx), 8, steps, points), pred, 1e-3)
        return rd.w3_field(raw_s, 3, 0.02, floor)

    pred = base()
    for _ in range(2):
        base()
        w3(pred)
    pairs = []
    for rep in range(7):
        order = ("base", "w3") if rep % 2 == 0 else ("w3", "base")
        times = {}
        for name in order:
            t0 = time.perf_counter()
            if name == "base":
                base()
            else:
                w3(pred)
            times[name] = time.perf_counter() - t0
        pairs.append({**times, "ratio": times["w3"] / times["base"]})
    ratio = float(np.median([p["ratio"] for p in pairs]))
    report = {
        "decision": "A09 / G1(iii)",
        "run": str(args.run),
        "task": task,
        "batch": args.batch,
        "steps": steps,
        "pairs": pairs,
        "median_ratio": ratio,
        "pass": ratio <= 10.0,
        "hardware": dev.describe(torch.device(args.device)),
        "command": " ".join(sys.argv),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"median_ratio": ratio, "pass": report["pass"]}))


if __name__ == "__main__":
    main()
