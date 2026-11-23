# Ported from the parallel Horizon workspace, commit a1500f9.
"""Train the teacher-forced FNO emulator (E1) on a 1D task's D_train.

Fixed before the first run:
- Split: the last 10% of D_train by index is the validation slice; normalization statistics come
  from the first 90% only.
- Model: FNO-1D, 4 layers, 16 modes, width 64; inputs (normalized u_t, normalized log nu channel);
  FiLM on normalized log nu after every Fourier block; output = normalized one-step increment.
- Optimizer AdamW (0.9, 0.99), weight decay 1e-4, peak lr 1e-3, 5-epoch linear warm-up, cosine to
  1e-6 at the budget (500 epochs), batch 64, loss = per-sample relative L2 of normalized increments.
- Validation every 5 epochs on all validation pairs: the same loss (selection metric) and the
  one-step relative L2 of the predicted state (reported). Selection, tail rule and acceptance use
  the normalized-increment metric. Selected checkpoint = minimum validation loss.
- Convergence: the two-window tail rule (last two 10% windows differ by < 2%).
- Acceptance check after training: T-step rollout from the 100 validation initial states, count
  of non-finite values; relative L2 and sup-norm error (physical and relative to sup |u_t|).
Eager float32 on MPS; no compile, no autocast. Atomic checkpoints every 10 epochs and on SIGINT or
SIGTERM; resumes from <out>/last.safetensors + last_state.json (no pickle).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.65")
for _name in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"):
    if os.environ.get(_name):
        raise SystemExit(f"{_name} must be unset")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

from trajcp.emulators.checkpoint import load_state, save_state  # noqa: E402
from trajcp.emulators.e1 import (  # noqa: E402
    BATCH,
    BETAS,
    FLOOR_LR,
    LAYERS,
    MODES,
    PEAK_LR,
    WARMUP,
    WEIGHT_DECAY,
    WIDTH,
    build_model,
    make_optimizer,
    train_epoch,
)
from trajcp.emulators.fno import FNO  # noqa: E402
from trajcp.emulators.training import (  # noqa: E402
    Normalization,
    atomic_safetensors,
    relative_l2,
    rollout,
    rollout_error_summary,
    step_inputs,
    tail_criterion,
)

VALIDATE_EVERY, CHECKPOINT_EVERY, VALIDATION_FRACTION = 5, 10, 0.1

stop_requested = False


def request_stop(signum: int, frame: object) -> None:
    global stop_requested
    stop_requested = True


def code_version() -> dict[str, object]:
    """Commit of the code used for the run, and whether tracked files differed from it."""
    root = Path(__file__).resolve().parent.parent

    def git(*cmd: str) -> str:
        return subprocess.run(
            ["git", *cmd], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(
            git("status", "--porcelain", "--untracked-files=no", "src", "scripts")
        ),
    }


@torch.no_grad()
def validate(
    model: FNO, u: torch.Tensor, mu: torch.Tensor, norm: Normalization
) -> dict[str, float]:
    model.eval()
    losses, state_errors = [], []
    n, steps = u.shape[0], u.shape[1] - 1
    for t in range(steps):
        cur, nxt = u[:, t], u[:, t + 1]
        x, c = step_inputs(cur, mu, norm)
        pred = model(x, c)[:, 0]
        target = (nxt - cur) / norm.increment_sd
        losses.append(relative_l2(pred, target))
        state_errors.append(relative_l2(cur + norm.increment_sd * pred, nxt))
    model.train()
    assert n > 0
    return {
        "val_loss": float(torch.cat(losses).mean()),
        "val_state_rel_l2": float(torch.cat(state_errors).mean()),
    }


def rollout_check(
    model: FNO, u: torch.Tensor, mu: torch.Tensor, norm: Normalization
) -> dict[str, object]:
    return rollout_error_summary(rollout(model, u, mu, norm), u[:, 1:].cpu().numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data", type=Path, required=True, help="directory with train.npy"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--peak-lr", type=float, default=PEAK_LR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    raw = np.load(args.data / "train.npy", mmap_mode="r")
    meta = pl.read_parquet(args.data / "train_meta.parquet").sort("index")
    log_nu = meta["log_nu"].to_numpy()
    n_val = round(VALIDATION_FRACTION * raw.shape[0])
    n_fit = raw.shape[0] - n_val
    norm = Normalization.from_training(raw[:n_fit], log_nu[:n_fit])
    mu_all = (log_nu - norm.mu_mean) / norm.mu_sd
    u_fit = torch.from_numpy(np.ascontiguousarray(raw[:n_fit])).to(device)
    u_val = torch.from_numpy(np.ascontiguousarray(raw[n_fit:])).to(device)
    mu_fit = torch.tensor(mu_all[:n_fit], dtype=torch.float32, device=device)
    mu_val = torch.tensor(mu_all[n_fit:], dtype=torch.float32, device=device)
    version = code_version()
    settings = {
        "code": version,
        "data": str(args.data),
        "seed": args.seed,
        "epochs": args.epochs,
        "n_fit": n_fit,
        "n_val": n_val,
        "validation_indices": [n_fit, raw.shape[0] - 1],
        "normalization": norm.as_dict(),
        "model": {
            "width": WIDTH,
            "modes": MODES,
            "layers": LAYERS,
            "cond": "log nu, channel + FiLM",
        },
        "optimizer": {"name": "AdamW", "betas": BETAS, "weight_decay": WEIGHT_DECAY},
        "lr": {
            "peak": args.peak_lr,
            "floor": FLOOR_LR,
            "warmup": WARMUP,
            "schedule": "cosine per step",
        },
        "batch": args.batch,
        "device": str(device),
        "torch": torch.__version__,
        "data_sha256": json.loads((args.data / "train_settings.json").read_text())[
            "sha256"
        ],
    }
    (args.out / "settings.json").write_text(json.dumps(settings, indent=2))

    torch.manual_seed(args.seed)
    model = build_model().to(device)
    optimizer = make_optimizer(model)
    curves: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_state_rel_l2": [],
        "seconds": [],
    }
    clock = time.perf_counter()
    best = float("inf")
    start_epoch = 0
    elapsed_before = 0.0
    resumes: list[int] = []
    state = load_state(args.out, model, optimizer, device)
    if state is not None:
        curves, best, start_epoch = state["curves"], state["best"], state["epoch"]
        elapsed_before = state.get("elapsed", 0.0)
        resumes = [*state.get("resumes", []), start_epoch]
        print(f"resumed at epoch {start_epoch}", flush=True)

    def elapsed() -> float:
        """Training seconds, cumulative across resumes."""
        return elapsed_before + time.perf_counter() - clock

    def checkpoint(epoch: int) -> None:
        save_state(
            args.out,
            model,
            optimizer,
            {
                "curves": curves,
                "best": best,
                "epoch": epoch,
                "elapsed": elapsed(),
                "resumes": resumes,
            },
        )
        (args.out / "curves.json").write_text(json.dumps(curves))

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    for epoch in range(start_epoch, args.epochs):
        mean_loss, _ = train_epoch(
            model,
            optimizer,
            u_fit,
            mu_fit,
            norm,
            epoch=epoch,
            total_epochs=args.epochs,
            seed=args.seed,
            batch=args.batch,
            peak_lr=args.peak_lr,
            stop=lambda: stop_requested,
        )
        if stop_requested:
            checkpoint(epoch)  # the interrupted epoch is repeated on resume
            raise SystemExit(f"stopped during epoch {epoch + 1}; checkpoint saved")
        done = epoch + 1
        if done % VALIDATE_EVERY == 0 or done == args.epochs:
            metrics = validate(model, u_val, mu_val, norm)
            curves["epoch"].append(done)
            curves["train_loss"].append(mean_loss)
            curves["val_loss"].append(metrics["val_loss"])
            curves["val_state_rel_l2"].append(metrics["val_state_rel_l2"])
            curves["seconds"].append(elapsed())
            if not np.isfinite(metrics["val_loss"]):
                checkpoint(done)
                raise SystemExit(f"non-finite validation loss at epoch {done}")
            if metrics["val_loss"] < best:
                best = metrics["val_loss"]
                atomic_safetensors(
                    model,
                    args.out / "best.safetensors",
                    {
                        "epoch": str(done),
                        "val_loss": repr(best),
                        "seed": str(args.seed),
                    },
                )
            print(
                f"epoch {done} train {mean_loss:.4e} val {metrics['val_loss']:.4e} "
                f"state {metrics['val_state_rel_l2']:.3e} {elapsed():.0f}s",
                flush=True,
            )
        if done % CHECKPOINT_EVERY == 0 or done == args.epochs:
            checkpoint(done)

    best_model = build_model()
    from safetensors.torch import load_file

    best_model.load_state_dict(load_file(str(args.out / "best.safetensors")))
    best_model.to(device)
    i_best = int(np.argmin(curves["val_loss"]))
    report = {
        "best_epoch": curves["epoch"][i_best],
        "best_val_loss": curves["val_loss"][i_best],
        "best_val_state_rel_l2": curves["val_state_rel_l2"][i_best],
        "tail": tail_criterion(curves["epoch"], curves["val_loss"], args.epochs),
        "rollout": rollout_check(best_model, u_val, mu_val, norm),
        "train_seconds": curves["seconds"][-1],
        "resumed": len(resumes) > 0,
        "resume_epochs": resumes,
        "code": version,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
