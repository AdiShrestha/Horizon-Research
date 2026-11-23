"""Study-side adapter for Factory's data-only P2 CPU trajectory worker."""
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from torch import nn

from p2_cpu_worker import FNO1d, STATE_VALUES, arrays, probe, rollout, train
from trajcp.conformal.core import calibrate, field_score, sup_score_from_steps
from trajcp.conformal.weights import select_profile, step_maxima
from trajcp.rollout.sensitivity import intrinsic_weights, perturbed_initials, raw_sensitivity, trajectory_key

PERTURBATION_SEED = 812075
FOLD_SEED = 813097
ALPHA = .10


def load_state(path):
    data = np.load(path, allow_pickle=False)
    if data.shape != (STATE_VALUES + 1,) or data.dtype != np.dtype("<f4") or not np.isfinite(data).all() or data[-1] <= 0:
        raise ValueError("invalid P2 FNO state including training field standard deviation")
    model = FNO1d()
    nn.utils.vector_to_parameters(torch.from_numpy(data[:-1].copy()), model.parameters())
    return model.eval(), 1e-6 * float(data[-1])


def sensitivity(model, initial, parameters):
    inputs = perturbed_initials(initial.numpy(), parameters.numpy(), seed=PERTURBATION_SEED)
    batched = torch.from_numpy(inputs.reshape(-1, inputs.shape[-1]))
    out = rollout(model, batched, parameters.repeat(len(inputs), 1))
    runs = out.reshape(len(inputs), len(initial), *out.shape[1:])
    return runs[0].astype(np.float64), raw_sensitivity(runs)


def fit_folds(initial, parameters):
    keys = [trajectory_key(u, p) for u, p in zip(initial, parameters, strict=True)]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate trajectory fingerprints in D_fit")
    order = np.array(sorted(range(len(keys)), key=keys.__getitem__))
    split = np.random.default_rng(FOLD_SEED).permutation(order)
    return np.array_split(split, 2)


def fitting(truth, forecast, w3, folds, epsilon0, eta):
    error = truth - forecast
    m = step_maxima(error, w3)
    coeff = np.linalg.norm(w3, axis=-1) / np.linalg.norm(truth, axis=-1)
    profile, choice, candidates = select_profile(m, coeff, folds, alpha=ALPHA, epsilon0=epsilon0, eta=eta)
    cached = sup_score_from_steps(m, profile)
    direct = field_score(error, w3 * profile[None, :, None])
    score_error = float(np.max(np.abs(cached - direct)))
    np.testing.assert_allclose(cached, direct, rtol=1e-13, atol=1e-13)
    # Test invariance of calibrated widths to global profile normalization on a D_fit fold.
    pseudo = folds[0]
    scale = 16.
    q = calibrate(cached[pseudo], ALPHA)
    scaled = calibrate(sup_score_from_steps(m[pseudo], profile * scale), ALPHA)
    np.testing.assert_allclose(2 * q * profile, 2 * scaled * profile * scale, rtol=1e-13, atol=1e-13)
    return {"profile": profile.tolist(), "choice": choice, "candidates": candidates,
            "fit_only_max_direct_cached_score_delta": score_error}


def main(request_file):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    request = json.loads(Path(request_file).read_text())
    torch.manual_seed(request["seed"])
    out, stage, view, cfg = Path(request["out_dir"]), request["stage"], request["data"], request["config"]
    if ("truth" in view) != (stage in ("train", "fit_weights")):
        raise ValueError("P2 scope violates label boundary")
    started = time.perf_counter()
    details = {"stage": stage, "torch": torch.__version__, "device": "cpu"}
    if stage == "train":
        model, losses = train(view, request["seed"], cfg["steps"])
        flat = nn.utils.parameters_to_vector(model.parameters()).detach().numpy()
        sigma = np.std(np.load(view["truth"], allow_pickle=False), dtype=np.float64)
        np.save(out / "state.npy", np.append(flat, sigma).astype("<f4"), allow_pickle=False)
        np.save(out / "probe.npy", probe(model), allow_pickle=False)
        details.update(first_loss=losses[0], last_loss=losses[-1], training_field_std=float(sigma))
    else:
        model, epsilon0 = load_state(request["model_state"])
        if stage == "load_check":
            np.save(out / "probe.npy", probe(model), allow_pickle=False)
        else:
            initial, params = arrays(view)
            if stage == "forecast":
                prediction = rollout(model, initial, params)
                np.save(out / "forecast.npy", prediction.astype("<f8"), allow_pickle=False)
            elif stage in ("fit_weights", "weights"):
                forecast, raw = sensitivity(model, initial, params)
                weights = intrinsic_weights(raw, cfg["smoothing"], cfg["eta"], epsilon0)
                if stage == "fit_weights":
                    if cfg["arm"] != "W4":
                        raise ValueError("only W4 can read D_fit labels")
                    state = fitting(np.load(view["truth"], allow_pickle=False), forecast, weights,
                                    fit_folds(initial.numpy(), params.numpy()), epsilon0, cfg["eta"])
                    (out / "state.json").write_text(json.dumps(state, allow_nan=False))
                else:
                    expected = np.load(request["forecast"], allow_pickle=False)
                    details["max_batched_forecast_delta"] = float(np.max(np.abs(expected - forecast)))
                    if cfg["arm"] == "W4":
                        state = json.loads(Path(request["fit_state"]).read_text())
                        weights *= np.array(state["profile"])[None, :, None]
                    elif cfg["arm"] != "W3" or "fit_state" in request:
                        raise ValueError("W3 cannot depend on a fitted envelope")
                    np.save(out / "weights.npy", weights.astype("<f8", copy=False), allow_pickle=False)
            else:
                raise ValueError("unknown P2 worker stage")
            (out / "ids.json").write_bytes(Path(view["ids"]).read_bytes())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    details.update(peak_rss_bytes=int(peak if sys.platform == "darwin" else 1024 * peak),
                   work_seconds=time.perf_counter() - started)
    (out / "telemetry.json").write_text(json.dumps(details, allow_nan=False))


if __name__ == "__main__":
    main(sys.argv[1])
