"""Train one emulator (task x tier x seed x budget) on the task's D_train (A18; brief §8).

Fixed rules (shared with 02_train_e1.py, which stays as the engineering trainer):
- Split: the last 10% of D_train by index is the validation slice; every normalization statistic
  (state, increment, parameter vector) comes from the first 90% only.
- Tiers, models, optimizer and schedule: ``trajcp.emulators.tiers``. Batch 64.
- Validation every 5 epochs on all validation pairs (one-step normalized-increment relative L2, the
  selection metric for every tier); the selected checkpoint is the minimum validation loss.
- Atomic SafeTensors checkpoints every 10 epochs and on SIGINT/SIGTERM; resumes from <out>/last.*.
- Final report: rollout over the task horizon from the 100 validation initial states (non-finite
  count; relative L2 and sup-norm errors at the marks; median sup-norm error at the final step, used
  by A07), growth slope of log median sup error against log t over t in [10, T], wall time, device.
Writes <out>/settings.json, curves.json, best.safetensors, last.*, report.json, DONE.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from safetensors.torch import load_file

from trajcp.emulators import device as dev
from trajcp.emulators import tiers
from trajcp.emulators.checkpoint import load_state, save_state
from trajcp.emulators.training import atomic_safetensors, rollout_error_summary

VALIDATE_EVERY, CHECKPOINT_EVERY, VALIDATION_FRACTION = 5, 10, 0.1
stop_requested = False


def request_stop(signum: int, frame: object) -> None:
    global stop_requested
    stop_requested = True


def code_version() -> dict[str, object]:
    root = Path(__file__).resolve().parent.parent
    manifest = root / "CODE_VERSION.json"
    if manifest.exists():  # cluster bundle without .git
        return json.loads(manifest.read_text()) | {"source": "bundle manifest"}

    def git(*cmd: str) -> str:
        return subprocess.run(
            ["git", *cmd], cwd=root, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no", "src", "scripts")),
        "source": "git",
    }


def load_task(data: Path, spec: tiers.TaskSpec):
    raw = np.load(data / "train.npy", mmap_mode="r")
    meta = pl.read_parquet(data / "train_meta.parquet").sort("index")
    cond_raw = np.stack([meta[name].to_numpy() for name in spec.cond], axis=1)
    settings = json.loads((data / "train_settings.json").read_text())
    return raw, cond_raw, settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", choices=sorted(tiers.TASKS), required=True)
    parser.add_argument("--tier", choices=tiers.TIERS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None, help="budget; or use --budget-file")
    parser.add_argument("--budget-file", type=Path, default=None, help="A07 budget.json")
    parser.add_argument("--data", type=Path, required=True, help="task directory with train.npy")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=None, help="smoke tests only")
    args = parser.parse_args()
    if (args.epochs is None) == (args.budget_file is None):
        raise SystemExit("pass exactly one of --epochs and --budget-file")
    epochs = args.epochs
    if args.budget_file is not None:
        epochs = int(json.loads(args.budget_file.read_text())["epochs"])
    spec = tiers.TASKS[args.task]
    device = dev.configure(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "DONE").unlink(missing_ok=True)

    raw, cond_raw, data_settings = load_task(args.data, spec)
    if raw.shape[1] - 1 != spec.steps:
        raise SystemExit(f"{args.task}: expected T = {spec.steps}, data has {raw.shape[1] - 1}")
    n_val = round(VALIDATION_FRACTION * raw.shape[0])
    n_fit = raw.shape[0] - n_val
    norm = tiers.CondNormalization.from_training(raw[:n_fit], cond_raw[:n_fit])
    c_all = norm.cond(cond_raw)
    u_fit = torch.from_numpy(np.array(raw[:n_fit], dtype=np.float32)).to(device)
    u_val = torch.from_numpy(np.array(raw[n_fit:], dtype=np.float32)).to(device)
    c_fit = torch.tensor(c_all[:n_fit], dtype=torch.float32, device=device)
    c_val = torch.tensor(c_all[n_fit:], dtype=torch.float32, device=device)
    peak = tiers.PEAK_LR[args.tier]
    version = code_version()
    settings = {
        "decision": "A18",
        "task": args.task,
        "tier": args.tier,
        "seed": args.seed,
        "epochs": epochs,
        "budget_file": str(args.budget_file) if args.budget_file else None,
        "code": version,
        "command": " ".join(sys.argv),
        "interpreter": sys.executable,
        "data": str(args.data),
        "data_sha256": data_settings["sha256"],
        "n_fit": n_fit,
        "n_val": n_val,
        "cond": list(spec.cond),
        "normalization": norm.as_dict(),
        "model": {
            "E1": "FNO-1D width 64, modes 16, 4 layers, FiLM",
            "E3": "FNO-1D width 64, modes 16, 4 layers, FiLM; unroll m=10 after 10% one-step",
            "E2": "U-Net (32, 64, 96) x 2 blocks, GroupNorm, GELU, circular, FiLM; pushforward K=3",
        }[args.tier],
        "optimizer": {"name": "AdamW", "betas": tiers.BETAS, "weight_decay": tiers.WEIGHT_DECAY},
        "lr": {"peak": peak, "floor": tiers.FLOOR_LR, "warmup": tiers.WARMUP, "schedule": "cosine"},
        "batch": tiers.BATCH,
        "hardware": dev.describe(device),
        "max_steps": args.max_steps,
    }
    (args.out / "settings.json").write_text(json.dumps(settings, indent=2))

    torch.manual_seed(args.seed)
    model = tiers.build(args.tier, len(spec.cond)).to(device)
    optimizer = tiers.make_optimizer(model, peak)
    curves: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_state_rel_l2": [],
        "seconds": [],
        "phase": [],
    }
    best, start_epoch, elapsed_before, resumes = float("inf"), 0, 0.0, []
    state = load_state(args.out, model, optimizer, device)
    if state is not None:
        curves, best, start_epoch = state["curves"], state["best"], state["epoch"]
        elapsed_before = state.get("elapsed", 0.0)
        resumes = [*state.get("resumes", []), start_epoch]
        print(f"resumed at epoch {start_epoch}", flush=True)
    clock = time.perf_counter()

    def elapsed() -> float:
        return elapsed_before + time.perf_counter() - clock

    def checkpoint(epoch: int) -> None:
        save_state(
            args.out,
            model,
            optimizer,
            {"curves": curves, "best": best, "epoch": epoch, "elapsed": elapsed(), "resumes": resumes},
        )
        (args.out / "curves.json").write_text(json.dumps(curves))

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    counter = {"steps": 0}

    def stop() -> bool:
        counter["steps"] += 1
        return stop_requested or (args.max_steps is not None and counter["steps"] >= args.max_steps)

    for epoch in range(start_epoch, epochs):
        mean_loss, _, phase = tiers.train_epoch(
            args.tier,
            model,
            optimizer,
            u_fit,
            c_fit,
            norm,
            epoch=epoch,
            total_epochs=epochs,
            seed=args.seed,
            peak_lr=peak,
            stop=stop,
        )
        if stop_requested:
            checkpoint(epoch)
            raise SystemExit(f"stopped during epoch {epoch + 1}; checkpoint saved")
        done = epoch + 1
        smoke_end = args.max_steps is not None and counter["steps"] >= args.max_steps
        if done % VALIDATE_EVERY == 0 or done == epochs or smoke_end:
            metrics = tiers.validate(model, u_val, c_val, norm)
            curves["epoch"].append(done)
            curves["train_loss"].append(mean_loss)
            curves["val_loss"].append(metrics["val_loss"])
            curves["val_state_rel_l2"].append(metrics["val_state_rel_l2"])
            curves["seconds"].append(elapsed())
            curves["phase"].append(phase)
            if not np.isfinite(metrics["val_loss"]):
                checkpoint(done)
                raise SystemExit(f"non-finite validation loss at epoch {done}")
            if metrics["val_loss"] < best:
                best = metrics["val_loss"]
                atomic_safetensors(
                    model,
                    args.out / "best.safetensors",
                    {"epoch": str(done), "val_loss": repr(best), "seed": str(args.seed)},
                )
            print(
                f"epoch {done} [{phase}] train {mean_loss:.4e} val {metrics['val_loss']:.4e} "
                f"state {metrics['val_state_rel_l2']:.3e} {elapsed():.0f}s",
                flush=True,
            )
        if done % CHECKPOINT_EVERY == 0 or done == epochs or smoke_end:
            checkpoint(done)
        if smoke_end:
            break

    best_model = tiers.build(args.tier, len(spec.cond))
    best_model.load_state_dict(load_file(str(args.out / "best.safetensors")))
    best_model.to(device)
    pred = tiers.rollout(best_model, u_val[:, 0], c_val, norm, spec.steps)
    truth = u_val[:, 1:].cpu().numpy()
    summary = rollout_error_summary(pred, truth)
    with np.errstate(invalid="ignore", over="ignore"):
        sup = np.abs(pred.astype(np.float64) - truth.astype(np.float64)).max(axis=2)
    sup = np.where(np.isfinite(sup), sup, np.inf)
    median_sup = np.quantile(sup, 0.5, axis=0, method="higher")
    stress = None
    if args.task == "p2":  # brief §7.1: P2 T = 200, stress rollout to 500 (finite-ness only)
        stress_pred = tiers.rollout(best_model, u_val[:, 0], c_val, norm, 500)
        stress = {
            "steps": 500,
            "non_finite_trajectories": int((~np.isfinite(stress_pred).all(axis=(1, 2))).sum()),
        }
    i_best = int(np.argmin(curves["val_loss"]))
    report = {
        "task": args.task,
        "tier": args.tier,
        "seed": args.seed,
        "epochs": epochs,
        "completed_epochs": curves["epoch"][-1],
        "best_epoch": curves["epoch"][i_best],
        "best_val_loss": curves["val_loss"][i_best],
        "best_val_state_rel_l2": curves["val_state_rel_l2"][i_best],
        "final_step_median_sup": float(median_sup[-1]),
        "growth_slope_log_t_10_T": tiers.growth_slope(median_sup),
        "rollout": summary,
        "stress_rollout": stress,
        "train_seconds": curves["seconds"][-1],
        "seconds_per_epoch": curves["seconds"][-1] / max(1, curves["epoch"][-1]),
        "resume_epochs": resumes,
        "hardware": settings["hardware"],
        "code": version,
        "smoke": args.max_steps is not None,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    (args.out / "DONE").write_text(json.dumps({"best_epoch": report["best_epoch"]}))
    print(json.dumps({k: report[k] for k in ("best_epoch", "best_val_loss", "final_step_median_sup", "growth_slope_log_t_10_T", "train_seconds")}))


if __name__ == "__main__":
    main()
