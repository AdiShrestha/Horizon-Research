"""Freeze P2 engineering settings using a design-only cohort and diagnose the old fit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import p2_cpu_worker as old
from p2_w34_worker import fit_folds, sensitivity
from trajcp.conformal.weights import SETTINGS, select_profile, step_maxima
from trajcp.rollout.sensitivity import intrinsic_weights

SOURCE = Path(__file__).resolve().parents[1]
BOUND = ["scripts/p2_w34_worker.py", "src/trajcp/rollout/sensitivity.py",
         "src/trajcp/conformal/weights.py", "src/trajcp/conformal/core.py", "src/trajcp/conformal/lp.py"]


def load(cohort, positions=None):
    index = slice(None) if positions is None else positions
    return [np.load(cohort / f"{name}.npy", allow_pickle=False, mmap_mode="r")[index]
            for name in ("initial", "parameters", "truth")]


def ratios(w, truth):
    return np.linalg.norm(w, axis=-1) / np.linalg.norm(truth, axis=-1)


def diagnose(model, inputs, previous, epsilon0):
    u, p, truth = inputs
    inputs32 = tuple(torch.from_numpy(x.astype(np.float32)) for x in (u, p))
    with torch.no_grad():
        t = torch.arange(u.shape[-1], dtype=torch.float32) * (2 * torch.pi / u.shape[-1])
        delta = 1e-3 * inputs32[0].square().mean(dim=-1, keepdim=True).sqrt()
        v = inputs32[0]
        init = torch.cat([v, v + delta * (2**.5 * t.sin()), v + delta * (2**.5 * (2*t).cos())])
        stack = old.rollout(model, init, inputs32[1].repeat(3, 1)).astype(float).reshape(3, *truth.shape)
    base, raw = stack[0], np.sqrt(np.mean(((stack[1:] - stack[0]) / 1e-3)**2, axis=0))
    eps, eta = 1e-4, .05
    rel_floor = eta * np.median(raw, axis=-1, keepdims=True)
    floor_only = np.maximum(raw, rel_floor)
    old_w3 = floor_only + eps
    old_profile = np.maximum(np.maximum.accumulate(np.quantile(step_maxima(truth - base, old_w3), .8, axis=0)), 1e-4)
    np.testing.assert_allclose(old_profile, previous, rtol=1e-10, atol=1e-10)
    new_base, new_raw = sensitivity(model, *inputs32)

    def summary(w, forecast):
        coeff = ratios(w, truth)
        worst = step_maxima(truth - forecast, w).max(axis=1)
        # Deliberately descriptive: six old fit rows cannot yield a finite .10 conformal rank.
        fit_q_proxy = float(np.quantile(worst, .9))
        return {"median_abs_weight_at_25_100": np.median(w, axis=(0, 2))[[24, 99]].tolist(),
                "median_fit_sup": float(np.median(worst)), "empirical_fit_score_p90": fit_q_proxy,
                "pseudo_width_p90_times_mean_ratio_at_25_100": (2*fit_q_proxy*coeff.mean(axis=0)[[24, 99]]).tolist()}

    return {
        "old_fit_rows": len(truth), "old_floor_fraction_relative": float(np.mean(raw < rel_floor)),
        "old_floor_fraction_absolute_exceeds_relative_floored_field": float(np.mean(floor_only < eps)),
        "old_profile_first_25_100": old_profile[[0, 24, 99]].tolist(),
        "old_profile_max_relative_reproduction_error": float(np.max(abs(old_profile - previous) / previous)),
        "counterfactuals": {
            "flat": summary(np.ones_like(old_w3), base),
            "old_W3": summary(old_w3, base),
            "old_W4": summary(old_w3 * old_profile[None, :, None], base),
            "training_std_floor_only_change": summary(floor_only + epsilon0, base),
            "smoothing_only_change": summary(intrinsic_weights(raw, 1, eta, eps), base),
            "eight_seeded_modes_only_change": summary(intrinsic_weights(new_raw, 0, eta, eps), new_base),
        },
    }


def freeze(old_cohort, old_run, design, output):
    torch.manual_seed(725)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    train_ids = json.loads((old_cohort / "trajectories.json").read_text())["train_ids"]
    manifest = json.loads((old_run / "manifest.json").read_text())
    fit_names = manifest["draws"][0]["draw"]["fit"]
    data = json.loads((old_cohort / "trajectories.json").read_text())
    positions = {unit: i for i, unit in enumerate(data["train_ids"] + data["pool_ids"])}
    eps = 1e-6 * float(np.float32(np.std(load(old_cohort, list(range(len(train_ids))))[-1], dtype=np.float64)))
    model = old.model_from(old_run / "models/fno/state.npy")
    previous = json.loads((old_run / "fit/0/W4/state.json").read_text())["profile"]
    old_fit = diagnose(model, load(old_cohort, [positions[u] for u in fit_names]), previous, eps)
    u, p, truth = load(design)
    forecast, raw = sensitivity(model, torch.from_numpy(u.astype(np.float32)), torch.from_numpy(p.astype(np.float32)))
    folds = fit_folds(u, p)
    grid = []
    for smoothing, eta in SETTINGS:
        w3 = intrinsic_weights(raw, smoothing, eta, eps)
        _, best, candidates = select_profile(step_maxima(truth - forecast, w3), ratios(w3, truth), folds,
                                             alpha=.1, epsilon0=eps, eta=eta)
        grid.append({"smoothing": smoothing, "eta": eta, "best": best, "candidates": candidates})
    selected = min(grid, key=lambda row: row["best"]["mean_relative_width"])
    materials = {name: hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in BOUND}
    materials["p2_cpu_worker.py"] = hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest()
    settings = {"smoothing": selected["smoothing"], "eta": selected["eta"]}
    result = {"scope": "P2 design-only frozen engineering settings", "settings": settings, "code_sha256": materials,
              "design_pool_seed": 20261124, "prospective_pool_seed": 20261125, "fold_sizes": [len(fold) for fold in folds],
              "old_fit_only": old_fit, "design_grid": grid}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-cohort", "old-run", "design", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    freeze(args.old_cohort, args.old_run, args.design, args.out)
