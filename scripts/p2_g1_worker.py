"""Data-only CPU stages for the T=200 P2 engineering fixture."""
import argparse
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from torch import nn

from p2_cpu_worker import FNO1d, probe
from p2_tdata2 import dump
from p2_w34_worker import load_state, fit_folds
from trajcp.conformal.weights import fit_profile, select_profile, step_maxima
from trajcp.rollout.sensitivity import intrinsic_weights, perturbed_initials, raw_sensitivity

T = 200


def configure(seed):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


def inputs(path):
    with np.load(path, allow_pickle=False) as batch:
        return np.asarray(batch["initial"], dtype=np.float32), np.asarray(batch["nu"], dtype=np.float32)


def paths(source):
    return sorted(source.glob("input-*.npz"))


def labeled(source):
    batches = paths(source)
    u, p = map(np.concatenate, zip(*(inputs(path) for path in batches), strict=True))
    truth = np.concatenate([np.load(path.with_name(path.name.replace("input-", "truth-")).with_suffix(".npy"),
                                    allow_pickle=False) for path in batches])
    assert truth.shape == (len(u), T, 256)
    return u, p, truth


@torch.no_grad()
def rollout(model, u, nu):
    history = []
    for _ in range(T):
        u = model(u, nu)
        history.append(u)
    return torch.stack(history, dim=1).numpy()


def ordinary(model, u, p):
    return rollout(model, torch.from_numpy(u), torch.from_numpy(p))


def joint(model, u, p, plan, epsilon0):
    settings = plan["weights"]
    # Fixed batch shape for calibration, test, benchmark and audit, including final chunks.
    batch = plan["chunk"]
    ix = np.arange(batch) % len(u)
    padded_u, padded_p = u[ix], p[ix]
    initial = perturbed_initials(padded_u, padded_p, seed=settings["perturbation_seed"],
                                 count=settings["M"], cutoff=settings["cutoff"], epsilon=settings["epsilon"])
    flat = torch.from_numpy(initial.reshape(-1, 256))
    out = rollout(model, flat, torch.from_numpy(padded_p).repeat(len(initial), 1))
    runs = out.reshape(len(initial), batch, T, 256)[:, :len(u)]
    raw = raw_sensitivity(runs, epsilon=settings["epsilon"])
    w3 = intrinsic_weights(raw, settings["smoothing"], settings["eta"], epsilon0)
    return runs[0].astype(np.float64), w3


def train(plan, scoped, out):
    initial, viscosity, truth = labeled(scoped)
    frames = torch.from_numpy(np.concatenate((initial[:, None], truth.astype(np.float32)), axis=1))
    nu = torch.from_numpy(viscosity)
    model = FNO1d()
    opt = torch.optim.Adam(model.parameters(), lr=.004)
    rng = torch.Generator().manual_seed(plan["training_seed"] + 101)
    losses = []
    for _ in range(plan["train_steps"]):
        i = torch.randint(len(frames), (8,), generator=rng)
        t = torch.randint(T, (8,), generator=rng)
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(frames[i, t], nu[i]), frames[i, t+1])
        loss.backward()
        opt.step()
        losses.append(loss.item())
    flat = nn.utils.parameters_to_vector(model.parameters()).detach().numpy()
    state = np.append(flat, np.std(truth, dtype=np.float64)).astype("<f4")
    np.save(out / "state.npy", state, allow_pickle=False)
    np.save(out / "probe.npy", probe(model.eval()), allow_pickle=False)
    return {"first_loss": losses[0], "last_loss": losses[-1], "steps": len(losses)}


def fit(plan, scoped, out, checkpoint):
    model, epsilon0 = load_state(checkpoint)
    initial, viscosity, truth = labeled(scoped)
    pieces = [joint(model, initial[i:i+plan["chunk"]], viscosity[i:i+plan["chunk"]], plan, epsilon0)
              for i in range(0, len(initial), plan["chunk"])]
    forecast, w3 = map(np.concatenate, zip(*pieces, strict=True))
    error = truth - forecast
    eta = plan["weights"]["eta"]
    w1 = fit_profile(np.abs(error).max(axis=-1), "q", beta=plan["weights"]["w1_beta"], epsilon0=epsilon0, eta=eta)
    coeff = np.linalg.norm(w3, axis=-1) / np.linalg.norm(truth, axis=-1)
    w4, choice, candidates = select_profile(step_maxima(error, w3), coeff, fit_folds(initial, viscosity),
                                             alpha=plan["alpha"], epsilon0=epsilon0, eta=eta)
    dump(out / "profiles.json", {"W1": w1.tolist(), "W4": w4.tolist(),
                                  "W4_choice": choice, "W4_candidates": candidates})
    return {"fit_rows": len(truth), "training_floor": epsilon0}


def forecast(plan, scoped, out, checkpoint):
    model, epsilon0 = load_state(checkpoint)
    assert all(path.name.startswith("input-") for path in scoped.iterdir())
    count = 0
    for path in paths(scoped):
        u, p = inputs(path)
        prediction, w3 = joint(model, u, p, plan, epsilon0)
        np.savez(out / path.name.replace("input-", "fields-"), forecast=prediction, W3=w3)
        count += len(u)
    return {"forecast_rows": count}


def benchmark(plan, scoped, out, checkpoint):
    model, epsilon0 = load_state(checkpoint)
    u, p = inputs(paths(scoped)[0])
    assert len(u) == plan["chunk"]
    for _ in range(plan["timing"]["warmup"]):
        ordinary(model, u, p)
        joint(model, u, p, plan, epsilon0)
    samples = []
    for repeat in range(plan["timing"]["repeats"]):
        values = {}
        operations = [("one", lambda: ordinary(model, u, p)), ("W3", lambda: joint(model, u, p, plan, epsilon0))]
        for name, operation in operations[::1 if repeat % 2 == 0 else -1]:
            start = time.perf_counter()
            operation()
            values[name] = time.perf_counter() - start
        samples.append(values)
    dump(out / "timing.json", samples)
    return {"batch": len(u), "median_paired_ratio": float(np.median([r["W3"]/r["one"] for r in samples]))}


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "fit", "forecast", "benchmark", "probe"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("scope", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir()
    configure(plan["training_seed"])
    started = time.perf_counter()
    if args.stage == "train":
        summary = train(plan, args.scope, args.out)
    elif args.stage == "probe":
        model, _ = load_state(args.checkpoint)
        np.save(args.out / "probe.npy", probe(model), allow_pickle=False)
        summary = {"probe": True}
    else:
        summary = {"fit": fit, "forecast": forecast, "benchmark": benchmark}[args.stage](
            plan, args.scope, args.out, args.checkpoint)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    dump(args.out / "worker.json", {**summary, "seconds": time.perf_counter() - started,
                                   "rss_bytes": int(rss if sys.platform == "darwin" else rss*1024)})
