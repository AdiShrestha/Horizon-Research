"""Adapter from confined Factory worker views to the frozen T=200 rehearsal."""
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

import p2_g1_worker as worker
from p2_tdata2 import dump, load_state, probe


def read_inputs(view):
    return tuple(np.load(view[key], allow_pickle=False).astype(np.float32) for key in ("initial", "parameters"))


def scoped_copy(view, out, labels):
    scope = out / "scoped"
    scope.mkdir()
    u, p = read_inputs(view)
    np.savez(scope / "input-000.npz", initial=u, nu=p)
    if labels:
        truth = np.load(view["truth"], allow_pickle=False)
        np.save(scope / "truth-000.npy", truth, allow_pickle=False)
    return scope


def main(path):
    request = json.loads(Path(path).read_text())
    out, stage, view = Path(request["out_dir"]), request["stage"], request["data"]
    role = request["config"].get("arm")
    plan = request["config"]["plan"]
    if ("truth" in view) != (stage in ("train", "fit_weights")):
        raise ValueError("G1 worker label scope changed")
    worker.configure(request["seed"])
    start = time.perf_counter()
    details = {}
    if stage == "train":
        details = worker.train(plan, scoped_copy(view, out, True), out)
    elif stage == "load_check":
        model, _ = load_state(request["model_state"])
        np.save(out / "probe.npy", probe(model), allow_pickle=False)
    elif stage == "fit_weights":
        scope = scoped_copy(view, out, True)
        details = worker.fit(plan, scope, out, request["model_state"])
        profiles = json.loads((out / "profiles.json").read_text())
        dump(out / "state.json", {"profile": profiles[role],
                                  **({"choice": profiles["W4_choice"]} if role == "W4" else {})})
    elif stage == "benchmark":
        scope = scoped_copy(view, out, False)
        details = worker.benchmark(plan, scope, out, request["model_state"])
        p = out / "timing.json"
        pairs = json.loads(p.read_text())
        p.write_text(json.dumps({"batch": plan["chunk"], "warmups": plan["timing"]["warmup"], "pairs": pairs}))
    else:
        u, p = read_inputs(view)
        model, floor = load_state(request["model_state"])
        if stage == "forecast":
            forecast, _ = worker.joint(model, u, p, plan, floor)
            np.save(out / "forecast.npy", forecast.astype("<f8", copy=False), allow_pickle=False)
        elif stage == "weights":
            expected = np.load(request["forecast"], allow_pickle=False)
            if role == "W1":
                state = json.loads(Path(request["fit_state"]).read_text())
                weights = np.broadcast_to(np.asarray(state["profile"])[None, :, None], expected.shape)
            else:
                forecast, weights = worker.joint(model, u, p, plan, floor)
                np.testing.assert_array_equal(forecast, expected)
                if role == "W4":
                    state = json.loads(Path(request["fit_state"]).read_text())
                    weights *= np.asarray(state["profile"])[None, :, None]
                elif role != "W3" or "fit_state" in request:
                    raise ValueError("intrinsic W3 cannot read a fitted envelope")
            np.save(out / "weights.npy", np.ascontiguousarray(weights, dtype="<f8"), allow_pickle=False)
        else:
            raise ValueError("unknown G1 worker stage")
        (out / "ids.json").write_bytes(Path(view["ids"]).read_bytes())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    dump(out / "telemetry.json", {**details, "stage": stage, "device": "cpu", "seconds": time.perf_counter()-start,
                                  "peak_rss_bytes": int(peak if sys.platform == "darwin" else 1024*peak)})


if __name__ == "__main__":
    main(sys.argv[1])
