"""Locked W3 joint and serial schedules through the Factory worker contract."""
import base64
from itertools import repeat
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

from p2_g1_factory_worker import read_inputs
from p2_g1_worker import ordinary, rollout
from p2_tdata2 import dump, file_digest, load_state, probe
from trajcp.rollout.sensitivity import intrinsic_weights, perturbed_initials, raw_sensitivity


def scheduled(model, u, p, frozen, floor, schedule):
    """Only grouping changes. Each schedule uses its own unperturbed member."""
    settings = frozen["weights"]
    n = frozen["timing"]["batch"]
    ix = np.arange(n) % len(u)
    initial = perturbed_initials(u[ix], p[ix], seed=settings["perturbation_seed"],
                                 count=settings["M"], cutoff=settings["cutoff"], epsilon=settings["epsilon"])
    nu = torch.from_numpy(p[ix])
    if schedule == "joint":
        result = rollout(model, torch.from_numpy(initial.reshape(-1, 256)), nu.repeat(len(initial), 1))
        runs = result.reshape(len(initial), n, 200, 256)
    elif schedule == "serial":
        runs = np.stack([rollout(model, torch.from_numpy(member), nu) for member in initial])
    else:
        raise ValueError("unregistered W3 schedule")
    runs = runs[:, :len(u)]
    raw = raw_sensitivity(runs, epsilon=settings["epsilon"])
    weights = intrinsic_weights(raw, settings["smoothing"], settings["eta"], floor)
    return runs[0].astype(np.float64), weights


def bench(model, u, p, frozen, floor, schedule):
    timing = frozen["timing"]
    assert len(u) == timing["batch"]
    operations = {"one": lambda: ordinary(model, u, p),
                  "joint": lambda: scheduled(model, u, p, frozen, floor, "joint"),
                  "serial": lambda: scheduled(model, u, p, frozen, floor, "serial")}
    base_order = ("one", "joint", "serial") if schedule == "joint" else ("serial", "joint", "one")
    for _ in repeat(None, timing["warmups"]):
        for name in base_order:
            operations[name]()
    measurements = []
    for i in range(timing["replicates_per_session"]):
        order = base_order if i % 2 == 0 else base_order[::-1]
        row = {"order": list(order), "started_ns": time.time_ns()}
        for name in order:
            begin = time.perf_counter()
            operations[name]()
            row[name] = time.perf_counter()-begin
        row["ended_ns"] = time.time_ns()
        measurements.append(row)
    return {"batch": len(u), "warmups": timing["warmups"], "threads": torch.get_num_threads(),
            "schedule": schedule, "triplets": measurements,
            "pairs": [{"one": r["one"], "W3": r[schedule]} for r in measurements]}


def main(file):
    request = json.loads(Path(file).read_text())
    config, view = request["config"], request["data"]
    frozen, schedule = config["frozen"], config["schedule"]
    stage, out = request["stage"], Path(request["out_dir"])
    torch.set_num_threads(frozen["timing"]["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(request["seed"])
    start = time.perf_counter()
    if stage == "train":
        # Materialize the previously committed FNO; do not train on, or read, current labels.
        state = out / "state.npy"
        state.write_bytes(base64.b64decode(frozen["state_npy_base64"], validate=True))
        assert file_digest(state) == frozen["state_sha256"]
        model, _ = load_state(state)
        np.save(out / "probe.npy", probe(model), allow_pickle=False)
    elif stage == "load_check":
        model, _ = load_state(request["model_state"])
        np.save(out / "probe.npy", probe(model), allow_pickle=False)
    elif stage == "fit_weights":
        # Factory issues the usual fit scope. This adapter only returns a locked prior profile.
        arm = config["arm"]
        dump(out / "state.json", {"profile": frozen["profiles"][arm]["profile"]})
    elif stage == "benchmark":
        assert "truth" not in view
        model, floor = load_state(request["model_state"])
        u, p = read_inputs(view)
        dump(out / "timing.json", bench(model, u, p, frozen, floor, schedule))
    else:
        assert "truth" not in view and stage in ("forecast", "weights")
        model, floor = load_state(request["model_state"])
        u, p = read_inputs(view)
        if stage == "forecast":
            prediction, _ = scheduled(model, u, p, frozen, floor, schedule)
            np.save(out / "forecast.npy", prediction.astype("<f8", copy=False), allow_pickle=False)
        else:
            state = request.get("fit_state")
            expected = np.load(request["forecast"], allow_pickle=False)
            if config["arm"] == "W1":
                profile = json.loads(Path(state).read_text())["profile"]
                weights = np.broadcast_to(np.asarray(profile)[None, :, None], expected.shape)
            else:
                forecast, weights = scheduled(model, u, p, frozen, floor, schedule)
                np.testing.assert_array_equal(forecast, expected)
                if config["arm"] == "W4":
                    weights *= np.asarray(json.loads(Path(state).read_text())["profile"])[None, :, None]
                else:
                    assert config["arm"] == "W3" and state is None
            np.save(out / "weights.npy", np.ascontiguousarray(weights, dtype="<f8"), allow_pickle=False)
        (out / "ids.json").write_bytes(Path(view["ids"]).read_bytes())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    dump(out / "telemetry.json", {"stage": stage, "device": "cpu", "threads": torch.get_num_threads(),
                                  "peak_rss_bytes": int(peak if sys.platform == "darwin" else 1024*peak),
                                  "seconds": time.perf_counter() - start})


if __name__ == "__main__":
    main(sys.argv[1])
