"""A07 budget gate for one task, applied mechanically to E1 seed 0 trained at B and 2B.

Keep B if 2B improves both the selected validation loss and the median final-step sup-norm rollout
error by less than 15%; otherwise use 2B for every tier and seed of the task (no further doubling).
Improvement = (value_B - value_2B) / value_B. Writes <root>/<task>/budget.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

THRESHOLD = 0.15


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="emulator root (contains <task>/)")
    parser.add_argument("--task", required=True)
    parser.add_argument("--base", type=int, default=500)
    parser.add_argument("--allow-smoke", action="store_true", help="pipeline smoke tests only")
    args = parser.parse_args()
    task_dir = args.root / args.task
    runs = {b: task_dir / f"e1_seed0_b{b}" for b in (args.base, 2 * args.base)}
    reports = {}
    for b, run in runs.items():
        if not (run / "DONE").exists():
            raise SystemExit(f"{run} has not finished")
        reports[b] = json.loads((run / "report.json").read_text())
        if reports[b].get("smoke") and not args.allow_smoke:
            raise SystemExit(f"{run} is a smoke run")
    lo, hi = reports[args.base], reports[2 * args.base]
    improve = {
        key: (lo[key] - hi[key]) / lo[key] if lo[key] > 0 else 0.0 for key in ("best_val_loss", "final_step_median_sup")
    }
    keep = all(v < THRESHOLD for v in improve.values())
    epochs = args.base if keep else 2 * args.base
    out = {
        "decision": "A07",
        "task": args.task,
        "epochs": epochs,
        "selected_run": runs[epochs].name,
        "improvement_2B_over_B": improve,
        "threshold": THRESHOLD,
        "B": {k: lo[k] for k in ("best_val_loss", "final_step_median_sup", "best_epoch")},
        "2B": {k: hi[k] for k in ("best_val_loss", "final_step_median_sup", "best_epoch")},
    }
    (task_dir / "budget.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
