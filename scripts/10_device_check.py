"""Device cross-check for production training (A06 rule, extended to every task and tier by A18).

For each task and tier: the same model (seed 0), data and batch order are trained for 200 optimizer
steps on CPU float32 and on the target device. The relative difference of the step-1 loss must be
below 1e-5 (gate); the differences at steps 50 and 200 are reported only. Writes <out>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

from trajcp.emulators import device as dev
from trajcp.emulators import tiers

GATE, STEPS, MARKS = 1e-5, 200, (1, 50, 200)


def losses(
    task: str, tier: str, data: Path, device: torch.device, n: int = 64, steps: int = STEPS
) -> list[float]:
    spec = tiers.TASKS[task]
    raw = np.load(data / "train.npy", mmap_mode="r")
    meta = pl.read_parquet(data / "train_meta.parquet").sort("index")
    cond = np.stack([meta[c].to_numpy() for c in spec.cond], axis=1)
    norm = tiers.CondNormalization.from_training(raw[:n], cond[:n])
    u = torch.from_numpy(np.array(raw[:n], dtype=np.float32)).to(device)
    c = torch.tensor(norm.cond(cond[:n]), dtype=torch.float32, device=device)
    torch.manual_seed(0)
    model = tiers.build(tier, len(spec.cond)).to(device)
    opt = tiers.make_optimizer(model, tiers.PEAK_LR[tier])
    out: list[float] = []
    epoch = 0 if tier != "E3" else 50  # E3: an unrolled epoch (10% warm-up of 500 is 50)
    tiers.train_epoch(
        tier, model, opt, u, c, norm, epoch=epoch, total_epochs=500, seed=0,
        peak_lr=tiers.PEAK_LR[tier], on_step=lambda i, v: out.append(float(v)),
        stop=lambda: len(out) >= steps,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["p2", "p1", "p3"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=STEPS, help="smoke tests only")
    args = parser.parse_args()
    target = dev.configure(args.device)
    results = []
    for task in args.tasks:
        for tier in tiers.TIERS:
            cpu = losses(task, tier, args.data_root / task, torch.device("cpu"), steps=args.steps)
            other = losses(task, tier, args.data_root / task, target, steps=args.steps)
            rel = {
                str(m): abs(other[m - 1] - cpu[m - 1]) / abs(cpu[m - 1])
                for m in MARKS
                if m <= min(len(cpu), len(other))
            }
            results.append({"task": task, "tier": tier, "relative_loss_difference": rel,
                            "pass": rel["1"] < GATE})
            print(json.dumps(results[-1]), flush=True)
    report = {
        "rule": f"step-1 relative loss difference < {GATE} (A06/A18)",
        "hardware": dev.describe(target),
        "command": " ".join(sys.argv),
        "results": results,
        "pass": all(r["pass"] for r in results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("device cross-check failed")


if __name__ == "__main__":
    main()
