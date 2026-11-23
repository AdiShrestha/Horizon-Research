"""E1 training throughput and the A06 device cross-check on P2 (IC-02 WP-C step 1).

Mode ``epoch``: seconds per epoch (all pairs of the fit trajectories) for one (device, batch),
three timed epochs after a 20-step warm-up (the learning rate follows the real schedule).
Mode ``crosscheck``: 200 optimizer steps on the CPU and on the GPU from the same initial weights
and the same batches (batch 64); at steps 1, 50 and 200 (before the update) the relative L2
difference of the model outputs on a fixed batch, and the relative loss difference. Step 1 is
gated at 1e-5, steps 50 and 200 are reported only (A06).
Appends one JSON line per measurement to --out. Eager float32, no compile, no autocast.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.65")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

from trajcp.emulators.e1 import (  # noqa: E402
    FLOOR_LR,
    WARMUP,
    build_model,
    make_optimizer,
    scaled_peak_lr,
    train_epoch,
)
from trajcp.emulators.training import (  # noqa: E402
    Normalization,
    learning_rate,
    relative_l2,
    step_inputs,
)

STEP1_TOLERANCE = 1e-5


def load(data: Path, device: torch.device):
    raw = np.load(data / "train.npy", mmap_mode="r")
    log_nu = (
        pl.read_parquet(data / "train_meta.parquet").sort("index")["log_nu"].to_numpy()
    )
    n_fit = raw.shape[0] - round(0.1 * raw.shape[0])
    norm = Normalization.from_training(raw[:n_fit], log_nu[:n_fit])
    mu = (log_nu - norm.mu_mean) / norm.mu_sd
    u = torch.from_numpy(np.ascontiguousarray(raw[:n_fit])).to(device)
    m = torch.tensor(mu[:n_fit], dtype=torch.float32, device=device)
    return u, m, norm


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def time_epochs(
    data: Path, device_name: str, batch: int, epochs: int
) -> dict[str, object]:
    device = torch.device(device_name)
    u, mu, norm = load(data, device)
    torch.manual_seed(0)
    model = build_model().to(device)
    optimizer = make_optimizer(model)
    peak = scaled_peak_lr(batch)
    counter = {"n": 0}

    def warm_stop() -> bool:
        counter["n"] += 1
        return counter["n"] >= 20

    train_epoch(
        model,
        optimizer,
        u,
        mu,
        norm,
        epoch=0,
        total_epochs=500,
        seed=0,
        batch=batch,
        peak_lr=peak,
        stop=warm_stop,
    )
    sync(device)
    seconds = []
    for e in range(epochs):
        sync(device)
        start = time.perf_counter()
        loss, batches = train_epoch(
            model,
            optimizer,
            u,
            mu,
            norm,
            epoch=1 + e,
            total_epochs=500,
            seed=0,
            batch=batch,
            peak_lr=peak,
        )
        sync(device)
        seconds.append(time.perf_counter() - start)
    return {
        "mode": "epoch",
        "device": device_name,
        "batch": batch,
        "peak_lr": peak,
        "seconds_per_epoch": seconds,
        "median": float(np.median(seconds)),
        "batches_per_epoch": batches,
        "last_loss": loss,
        "torch": torch.__version__,
    }


def cross_check(
    data: Path, gpu: str, steps: int = 200, batch: int = 64
) -> dict[str, object]:
    devices = [torch.device("cpu"), torch.device(gpu)]
    runs = []
    for device in devices:
        u, mu, norm = load(data, device)
        torch.manual_seed(0)
        model = build_model().to(device)
        runs.append((device, u, mu, norm, model, make_optimizer(model)))
    pairs = runs[0][1].shape[0] * (runs[0][1].shape[1] - 1)
    order = torch.randperm(pairs, generator=torch.Generator().manual_seed(0))
    probe_idx = torch.randperm(pairs, generator=torch.Generator().manual_seed(1))[
        :batch
    ]
    outputs: dict[int, list[torch.Tensor]] = {}
    losses: dict[int, list[float]] = {}
    for device, u, mu, norm, model, optimizer in runs:
        n_steps = u.shape[1] - 1
        for step in range(1, steps + 1):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(
                    step / (pairs / batch),
                    peak=1e-3,
                    floor=FLOOR_LR,
                    warmup=WARMUP,
                    total=500,
                )
            idx = order[(step - 1) * batch : step * batch].to(device)
            traj, t = idx // n_steps, idx % n_steps
            cur, nxt = u[traj, t], u[traj, t + 1]
            x, c = step_inputs(cur, mu[traj], norm)
            loss = relative_l2(
                model(x, c)[:, 0], (nxt - cur) / norm.increment_sd
            ).mean()
            if step in (1, 50, 200):
                p = probe_idx.to(device)
                pt, pq = p // n_steps, p % n_steps
                px, pc = step_inputs(u[pt, pq], mu[pt], norm)
                with torch.no_grad():
                    outputs.setdefault(step, []).append(model(px, pc).cpu().double())
                losses.setdefault(step, []).append(float(loss))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    result = {}
    for step in (1, 50, 200):
        a, b = outputs[step]
        result[str(step)] = {
            "output_rel_l2": float(torch.linalg.norm(a - b) / torch.linalg.norm(a)),
            "loss_cpu": losses[step][0],
            "loss_gpu": losses[step][1],
            "loss_rel_diff": abs(losses[step][0] - losses[step][1]) / losses[step][0],
        }
    return {
        "mode": "crosscheck",
        "gpu": gpu,
        "steps": result,
        "step1_tolerance": STEP1_TOLERANCE,
        "step1_pass": result["1"]["output_rel_l2"] < STEP1_TOLERANCE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["epoch", "crosscheck"], required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if args.mode == "epoch":
        row = time_epochs(args.data, args.device, args.batch, args.epochs)
    else:
        row = cross_check(args.data, args.device)
    print(json.dumps(row), flush=True)
    with args.out.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
