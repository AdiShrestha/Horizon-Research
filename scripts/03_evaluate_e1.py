"""Evaluate a trained one-step emulator on its validation slice and cross-check devices.

Reads <run>/settings.json and <run>/best.safetensors. Rolls out the selected weights from the
validation initial states on MPS float32, CPU float32 and CPU float64 (weights cast to float64,
complex weights to complex128). Writes <run>/evaluation.json with:
- the rollout error summary of the MPS float32 rollout (relative L2; sup-norm error in physical
  units and relative to sup |u_t|; median and 90th percentile at the standard step marks);
- the largest relative L2 difference between device rollouts at steps 1, 50 and 200.
Tolerance fixed before the first run: step 1, CPU float32 against MPS, below 1e-5. Steps 50 and
200 are reported, not gated.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.65")

import numpy as np
import polars as pl
import torch
from safetensors.torch import load_file

from trajcp.emulators.fno import FNO, complex_dtype_for
from trajcp.emulators.training import (
    Normalization,
    rollout,
    rollout_error_summary,
)

STEP1_TOLERANCE = 1e-5
CHECK_STEPS = (1, 50, 200)


def max_relative_difference(a: np.ndarray, b: np.ndarray, step: int) -> float:
    a64, b64 = a[:, step - 1].astype(np.float64), b[:, step - 1].astype(np.float64)
    return float((np.linalg.norm(a64 - b64, axis=1) / np.linalg.norm(b64, axis=1)).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads((args.run / "settings.json").read_text())
    data = Path(settings["data"])
    norm = Normalization(**settings["normalization"])
    first, last = settings["validation_indices"]
    raw = np.load(data / "train.npy", mmap_mode="r")
    log_nu = pl.read_parquet(data / "train_meta.parquet").sort("index")["log_nu"].to_numpy()
    u_val = np.ascontiguousarray(raw[first : last + 1])
    mu_val = (log_nu[first : last + 1] - norm.mu_mean) / norm.mu_sd
    weights = load_file(str(args.run / "best.safetensors"))
    spec = settings["model"]

    def run(device: str, dtype: torch.dtype) -> np.ndarray:
        model = FNO(
            1, 2, 1, width=spec["width"], modes=spec["modes"], layers=spec["layers"], cond_dim=1
        )
        model.load_state_dict(weights)
        model.to(device)
        complex_dtype = complex_dtype_for(dtype)
        for param in model.parameters():  # Module.to(dtype) would drop imaginary parts
            param.data = param.data.to(complex_dtype if param.is_complex() else dtype)
        u = torch.tensor(u_val, dtype=dtype, device=device)
        mu = torch.tensor(mu_val, dtype=dtype, device=device)
        return rollout(model, u, mu, norm)

    mps = run("mps", torch.float32)
    cpu32 = run("cpu", torch.float32)
    cpu64 = run("cpu", torch.float64)
    pairs = {
        "cpu32_vs_mps": (cpu32, mps),
        "cpu64_vs_cpu32": (cpu64, cpu32),
        "cpu64_vs_mps": (cpu64, mps),
    }
    cross = {
        name: {str(s): max_relative_difference(a, b, s) for s in CHECK_STEPS if s <= a.shape[1]}
        for name, (a, b) in pairs.items()
    }
    step1 = cross["cpu32_vs_mps"]["1"]
    report = {
        "weights": str(args.run / "best.safetensors"),
        "validation_indices": [first, last],
        "rollout_mps_float32": rollout_error_summary(mps, u_val[:, 1:]),
        "rollout_cpu_float64": rollout_error_summary(cpu64, u_val[:, 1:]),
        "device_cross_check_max_rel_l2": cross,
        "step1_tolerance": STEP1_TOLERANCE,
        "step1_pass": step1 < STEP1_TOLERANCE,
    }
    (args.run / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
