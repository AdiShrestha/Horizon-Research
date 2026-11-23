"""Fit-only width diagnosis and CPU W3 profiling on synthetic timing inputs."""
import argparse
from importlib.metadata import version
from pathlib import Path
from statistics import median
import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from p2_g1 import load
from p2_g1_worker import configure, labeled, rollout
from p2_tdata2 import dump, file_digest, load_state
from trajcp.conformal.core import calibrate
from trajcp.rollout.sensitivity import perturbed_initials, raw_sensitivity


def phases(model, u, p, settings, epsilon0, group, multiplier=1):
    clock = time.perf_counter
    time0 = clock()
    initial = perturbed_initials(u, p, seed=settings["perturbation_seed"], count=settings["M"],
                                 cutoff=settings["cutoff"], epsilon=settings["epsilon"]*multiplier)
    duration = {"perturbation": clock()-time0}
    time0 = clock()
    nu = torch.from_numpy(p)
    flat = [rollout(model, torch.from_numpy(x.reshape(-1, 256)), nu.repeat(len(x), 1))
            for x in (initial[i:i+group] for i in range(0, len(initial), group))]
    runs = np.concatenate(flat).reshape(len(initial), len(u), 200, 256)
    duration["rollout"] = clock()-time0
    time0 = clock()
    raw = raw_sensitivity(runs, epsilon=settings["epsilon"]*multiplier)
    duration["RMS_reduction"] = clock()-time0
    time0 = clock()
    smooth = gaussian_filter1d(raw, settings["smoothing"], axis=-1, mode="wrap")
    duration["smoothing"] = clock()-time0
    time0 = clock()
    threshold = settings["eta"] * np.median(smooth, axis=-1, keepdims=True)
    floored = np.maximum(smooth, threshold)
    w3 = floored + epsilon0
    duration["floors"] = clock()-time0
    return runs[0], raw, smooth, floored, w3, duration


def summaries(error, truth, w3):
    steps = np.max(error/w3, axis=-1)
    scores = steps.max(axis=1)
    coefficient = (np.linalg.norm(w3, axis=-1)/np.linalg.norm(truth, axis=-1)).mean(axis=1)
    q = calibrate(scores, .1)
    return {"max_score_percentiles_10_50_90": np.quantile(scores, [.1,.5,.9]).tolist(),
            "fit_only_rank_threshold": q, "mean_width_coefficient": float(np.mean(coefficient)),
            "fit_only_width_proxy": float(2*q*np.mean(coefficient)),
            "case_with_max_score": int(np.argmax(scores)),
            "step_bins_mean_ratio": [float(np.mean(a)) for a in np.array_split(steps, 4, axis=1)]}


def fit_diagnosis(model, epsilon0, settings, path):
    u, p, y = labeled(path)
    field_parts = [phases(model, u[i:i+12], p[i:i+12], settings, epsilon0, 9)[:5] for i in range(0, len(u), 12)]
    prediction, raw, smooth, floored, w3 = map(np.concatenate, zip(*field_parts, strict=True))
    error = np.abs(y-prediction)
    fields = {"B01": np.ones_like(w3), "raw_plus_absolute_floor": raw+epsilon0,
              "smoothed_plus_absolute_floor": smooth+epsilon0, "W3": w3}
    statistics = {name: summaries(error, y, value) for name, value in fields.items()}
    a,b = statistics["B01"], statistics["W3"]
    cells = np.unravel_index(np.argmax(error/w3), error.shape)
    return {"role": path.name, "rows": len(y), "statistics": statistics,
            "relative_floor_active_fraction": float(np.mean(floored > smooth)),
            "absolute_floor_exceeds_relative_floored_value_fraction": float(np.mean(epsilon0 > floored)),
            "W3_over_B01_score_multiplier": b["fit_only_rank_threshold"]/a["fit_only_rank_threshold"],
            "W3_over_B01_coefficient_multiplier": b["mean_width_coefficient"]/a["mean_width_coefficient"],
            "W3_over_B01_width_proxy_multiplier": b["fit_only_width_proxy"]/a["fit_only_width_proxy"],
            "W3_peak_normalized_error_location": {"row": int(cells[0]), "step": int(cells[1]+1), "cell": int(cells[2]),
                                                 "error": float(error[cells]), "weight": float(w3[cells])}}


def timing(model, settings, epsilon0):
    x = 2 * np.pi * np.arange(256)/256
    u = np.asarray([np.sin(x+.13*i) + .15*np.cos(3*x-.08*i) for i in range(12)], dtype=np.float32)
    p = np.linspace(.0008,.0044,12,dtype=np.float32)[:, None]
    def one():
        start = time.perf_counter()
        rollout(model, torch.from_numpy(u), torch.from_numpy(p))
        return time.perf_counter()-start
    for group in (9, 3, 1):
        phases(model, u, p, settings, epsilon0, group)
    one()
    repeats = []
    base = phases(model, u, p, settings, epsilon0, 9)
    for i in range(5):
        row = {"one": one()}
        for group in ((9,3,1) if i%2 == 0 else (1,3,9)):
            row[str(group)] = phases(model, u, p, settings, epsilon0, group)[-1]
        repeats.append(row)
    candidates = {}
    for group in (9,3,1):
        stats = [row[str(group)] for row in repeats]
        timed = [sum(r.values()) for r in stats]
        fresh = phases(model, u, p, settings, epsilon0, group)
        candidates[str(group)] = {"phase_median_seconds": {step: median(row[step] for row in stats) for step in stats[0]},
                                  "median_ratio_to_one": median(t/r["one"] for t,r in zip(timed,repeats,strict=True)),
                                  "max_unperturbed_forecast_delta": float(np.max(np.abs(base[0]-fresh[0]))),
                                  "max_absolute_weight_delta": float(np.max(np.abs(base[4]-fresh[4]))),
                                  "relative_weight_delta_p99": float(np.quantile(np.abs(base[4]-fresh[4])/base[4], .99))}
    return {"data": "deterministic synthetic trig inputs and fixed viscosities", "batch": 12,
            "median_one_seconds": median(row["one"] for row in repeats), "groups": candidates, "samples": repeats}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("freeze", "state", "fit0", "fit1", "out"):
        p.add_argument("--"+arg, required=True, type=Path)
    args = p.parse_args()
    plan = load(args.freeze)
    configure(plan["training_seed"])
    model, floor = load_state(args.state)
    fit = [fit_diagnosis(model, floor, plan["weights"], path) for path in (args.fit0,args.fit1)]
    timed = timing(model, plan["weights"], floor)
    dump(args.out, {"scope": "frozen prior D_fit labels and synthetic timing inputs only",
                    "freeze_sha256": file_digest(args.freeze), "checkpoint_sha256": file_digest(args.state),
                    "runtime": {n:version(n) for n in ("torch","numpy","scipy")}, "fit": fit, "timing": timed})
