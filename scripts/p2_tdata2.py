"""Freeze and run the separately seeded P2 engineering analogue of T-data-2."""
import argparse
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import math
import os
from pathlib import Path
import resource
import struct
import subprocess
import sys
import tempfile
import time
from zipfile import ZipFile

import numpy as np
import torch

import p2_cpu_worker
from p2_w34_worker import load_state, probe, sensitivity
from trajcp.data.weighted_audit import WeightedAudit
from trajcp.rollout.sensitivity import intrinsic_weights, trajectory_key

SOURCE = Path(__file__).resolve().parents[1]
SEED = 20261126
CHUNK = 12
ARMS = ("B01", "W3", "W4")
CASES = [(512, 10), (512, 20), (1024, 20), (2048, 20), (1024, 80), (2048, 80)]
COMPARISONS = {
    "approved_spatial_1024_2048_at_20": ("g1024_s20", "g2048_s20"),
    "brief_spatial_512_1024_at_20": ("g512_s20", "g1024_s20"),
    "old_coupled_512_10_to_1024_20": ("g512_s10", "g1024_s20"),
    "temporal_512_10_to_20": ("g512_s10", "g512_s20"),
    "temporal_1024_20_to_80": ("g1024_s20", "g1024_s80"),
    "temporal_2048_20_to_80": ("g2048_s20", "g2048_s80"),
    "spatial_1024_2048_at_80": ("g1024_s80", "g2048_s80"),
}
AUDIT_CODE = ("scripts/p2_tdata2.py", "scripts/p2_tdata2_solver.py",
              "src/trajcp/data/exponax_gen.py", "src/trajcp/data/weighted_audit.py")


def digest(raw):
    return sha256(raw).hexdigest()


def file_digest(path):
    with path.open("rb") as f:
        return __import__("hashlib").file_digest(f, "sha256").hexdigest()


def dump(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def array(raw):
    return np.load(BytesIO(raw), allow_pickle=False)


def read(z, member):
    return json.loads(z.read(member))


def worker_hashes(binding):
    special = {"worker.py": SOURCE / "scripts/p2_w34_worker.py",
               "p2_cpu_worker.py": Path(p2_cpu_worker.__file__)}
    return {name: file_digest(special.get(name, SOURCE / "src" / name)) for name in binding}


def exclude(initial, parameters):
    return {trajectory_key(u, p).hex() for u, p in zip(initial, parameters, strict=True)}


def verify_factory(bundle, kernel, maintainer, custodian):
    env = dict(os.environ, PYTHONPATH=str(kernel), PYTHONDONTWRITEBYTECODE="1")
    def cli(*args):
        return json.loads(subprocess.check_output([sys.executable, "-m", "sf7.cli", *args], env=env))
    k2 = cli("kernel", "verify")
    assert k2["version"] == "7.2.3" and k2["level"] == "K2"
    replay = cli("verify-bundle", str(bundle), "--maintainer-key", maintainer.read_text().strip(),
                 "--custodian-key", custodian.read_text().strip())
    assert replay["status"] == "VERIFIED" and replay["rescored_targets"] and replay["paired_draws"] == 2
    return k2["merkle_root"], replay


def prior_material(bundle, design, design_provenance):
    scope = json.loads(design_provenance.read_text())
    assert scope["seed"] == 20261124
    assert all(file_digest(design / name) == expected for name, expected in scope["artifacts"].items())
    design_ids = exclude(np.load(design / "initial.npy", allow_pickle=False),
                         np.load(design / "parameters.npy", allow_pickle=False))
    with ZipFile(bundle) as z:
        registered = read(z, "registration.json")
        spec = registered["spec"]
        assert worker_hashes(spec["code"]["worker.py"]["files"]) == spec["code"]["worker.py"]["files"]
        plan = spec["plan"]
        assert plan["evaluation"]["primary_alpha"] == .1 and plan["evaluation"]["primary_horizon"] == 100
        run = "runs/725/"
        manifest = read(z, run + "manifest.json")
        assert manifest["seed"] == 725 and manifest["family"] == registered["family"]
        inputs = read(z, "raw/trajectories.json")
        assert inputs["provenance"]["seed"] == 20261125
        ids = inputs["train_ids"] + inputs["pool_ids"]
        initial, viscosity = (array(z.read(f"raw/{name}.npy")) for name in ("initial", "parameters"))
        keys = exclude(initial, viscosity)
        assert len(design_ids) == 24 and len(keys) == len(ids) == 85 and not keys & design_ids
        settings = {k: v for k, v in plan["arms"]["W3"]["worker"]["config"].items() if k != "arm"}
        assert settings == {k: v for k, v in plan["arms"]["W4"]["worker"]["config"].items() if k != "arm"}
        draws = []
        for item in manifest["draws"]:
            draw = item["draw"]
            index = draw["fit_index"]
            rel = f"fit/{index}/W4/state.json"
            raw = z.read(run + rel)
            assert digest(raw) == manifest["files"][rel]
            fit = json.loads(raw)
            profiles = np.asarray(fit["profile"])
            assert profiles.shape == (100,) and np.isfinite(profiles).all() and np.all(profiles > 0)
            scores = read(z, run + item["reductions"])["arms"]
            n = len(draw["calibration"])
            rank = math.ceil(9 * (n + 1) / 10)
            assert 0 < rank <= n == 19
            thresholds = {}
            for arm in ARMS:
                select = [row["score"] for row in scores[arm] if row["unit"] in draw["calibration"]]
                assert len(select) == n
                thresholds[arm] = sorted(select)[rank - 1]
                assert np.isfinite(thresholds[arm]) and thresholds[arm] > 0
            draws.append({"fit": index, "calibration": draw["calibration_index"], "rank": rank,
                          "profile_sha256": digest(raw), "profile": fit["profile"], "choice": fit["choice"],
                          "thresholds": thresholds})
        state = z.read(run + "models/fno/state.npy")
        assert digest(state) == manifest["files"]["models/fno/state.npy"]
        tmp_path = Path(tempfile.gettempdir())
        with tempfile.NamedTemporaryFile(dir=tmp_path, suffix=".npy") as tf:
            tf.write(state)
            tf.flush()
            configure_torch()
            model, floor = load_state(Path(tf.name))
            original = array(z.read(run + "models/fno/probe.npy"))
            reloaded = array(z.read(run + "models/fno/reload/probe.npy"))
            np.testing.assert_array_equal(probe(model), original)
            np.testing.assert_array_equal(reloaded, original)
            prior_chunks = []
            positions = {name: i for i, name in enumerate(ids)}
            for item, draw in zip(manifest["draws"], draws, strict=True):
                chunk = item["field_chunks"][0]
                selected = [positions[i] for i in chunk["order"]]
                w3 = w3_weights(model, initial[selected], viscosity[selected], settings, floor)
                w4 = w3 * np.asarray(draw["profile"])[None, :, None]
                for arm, estimate in (("W3", w3), ("W4", w4)):
                    expected = array(z.read(run + chunk["weights"][arm]))
                    np.testing.assert_array_equal(estimate, expected)
                prior_chunks.append({"fit": draw["fit"], "trajectories": len(selected),
                                     "max_weight_difference": 0.0})
        return {"worker_code": spec["code"]["worker.py"]["files"], "draws": draws,
                "settings": settings, "state_sha256": digest(state), "training_floor": floor,
                "prior_chunk_checks": prior_chunks, "probe_max_absolute_difference": 0.0,
                "excluded_prior_fingerprints": sorted(keys | design_ids)}


def configure_torch():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(725)


def w3_weights(model, initial, parameters, settings, floor):
    _, raw = sensitivity(model, torch.from_numpy(initial.astype(np.float32)),
                         torch.from_numpy(parameters.astype(np.float32)))
    return intrinsic_weights(raw, settings["smoothing"], settings["eta"], floor)


def freeze(args):
    root, replay = verify_factory(args.bundle, args.kernel, args.maintainer, args.custodian)
    prior = prior_material(args.bundle, args.design, args.design_provenance)
    config = json.loads((SOURCE / "configs/task/p2_burgers1d_pilot.json").read_text())
    config.update(master_seed=SEED, steps=100)
    freeze = {"scope": "P2 engineering T-data-2 audit, frozen before independently seeded solver run",
              "bundle_sha256": file_digest(args.bundle), "kernel_merkle_root": root,
              "verified_family": replay["family_id"], "verified_ledger_head": replay["ledger_head"],
              "design_provenance_sha256": file_digest(args.design_provenance),
              "audit_code": {name: file_digest(SOURCE / name) for name in AUDIT_CODE},
              "solver_config": config, "solver_cases": CASES, "comparisons": COMPARISONS,
              "weight_input_case": "g1024_s20", "chunk": CHUNK, "audit_ids": list(range(50)),
              "decision_limit": 1e-3, "prior": prior,
              "runtime": {name: version(name) for name in ("torch", "numpy", "scipy")},
              "solver_versions": {"exponax": "0.2.1", "jax": "0.12.0.1", "jaxlib": "0.12.0.1",
                                  "numpy": "2.5.0", "scipy": "1.18.1"}}
    dump(args.out, freeze)
    print(f"Frozen plan {file_digest(args.out)}")


def frames(stream):
    while header := stream.read(8):
        (size,) = struct.unpack("<Q", header)
        if not 0 < size <= 24 * 1024**2:
            raise ValueError("unexpected solver frame size")
        raw = stream.read(size)
        if len(raw) != size:
            raise EOFError("solver frame ended early")
        with np.load(BytesIO(raw), allow_pickle=False) as batch:
            yield batch


def collect(model, floor, frozen, solver):
    prior = frozen["prior"]
    comparisons = frozen["comparisons"]
    reduced = {name: {arm: WeightedAudit() for arm in ("B01", "W3", "W4/0", "W4/1")}
               for name in comparisons}
    seen, audit = set(prior["excluded_prior_fingerprints"]), []
    profiles = {f"W4/{draw['fit']}": np.array(draw["profile"])[None, :, None] for draw in prior["draws"]}
    times = {f"g{g}_s{s}": {"compile_seconds": 0., "solve_seconds": 0., "max_initial_delta": 0.}
             for g, s in frozen["solver_cases"]}
    weight_seconds = 0.
    for batch in frames(solver.stdout):
        ids = batch["ids"]
        expected = frozen["audit_ids"][len(audit):len(audit) + len(ids)]
        np.testing.assert_array_equal(ids, expected)
        assert 0 < len(ids) <= frozen["chunk"]
        initial, nu = batch["initial"], batch["nu"]
        inputs = exclude(initial, nu[:, None])
        assert not inputs & seen and len(inputs) == len(ids)
        seen.update(inputs)
        for i, nu_i, slope, amp, key in zip(ids, nu, batch["slope"], batch["amplitude"],
                                           batch["keys"], strict=True):
            audit.append({"id": int(i), "viscosity": float(nu_i), "slope": float(slope),
                          "amplitude": float(amp), "prng_key": key.tolist()})
        for summary, row in zip(times.values(), batch["timings"], strict=True):
            summary["compile_seconds"] += float(row[0])
            summary["solve_seconds"] += float(row[1])
            summary["max_initial_delta"] = max(summary["max_initial_delta"], float(row[2]))
        started = time.perf_counter()
        w3 = w3_weights(model, initial, nu[:, None], prior["settings"], floor)
        weights = {"B01": 1., "W3": w3, **{key: w3 * profile for key, profile in profiles.items()}}
        weight_seconds += time.perf_counter() - started
        for name, (left, right) in comparisons.items():
            discrepancy = np.abs(batch[right] - batch[left])
            for arm, reducer in reduced[name].items():
                reducer.update(ids, discrepancy, weights[arm])
    return times, weight_seconds, audit, reduced


def run(args):
    frozen = json.loads(args.freeze.read_text())
    assert file_digest(args.bundle) == frozen["bundle_sha256"]
    assert worker_hashes(frozen["prior"]["worker_code"]) == frozen["prior"]["worker_code"]
    assert {name: file_digest(SOURCE / name) for name in AUDIT_CODE} == frozen["audit_code"]
    assert {name: version(name) for name in frozen["runtime"]} == frozen["runtime"]
    names = list(frozen["solver_versions"])
    actual = subprocess.check_output([str(args.solver_python), "-c",
                                     "import json,sys;from importlib.metadata import version;"
                                     "print(json.dumps({x: version(x) for x in sys.argv[1:]}))", *names])
    assert json.loads(actual) == frozen["solver_versions"]
    configure_torch()
    started = time.perf_counter()
    with ZipFile(args.bundle) as z, tempfile.NamedTemporaryFile(suffix=".npy") as tf:
        raw = z.read("runs/725/models/fno/state.npy")
        assert digest(raw) == frozen["prior"]["state_sha256"]
        tf.write(raw)
        tf.flush()
        model, floor = load_state(Path(tf.name))
        assert floor == frozen["prior"]["training_floor"]
        env = dict(os.environ, PYTHONPATH=str(SOURCE / "src"), JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true")
        with subprocess.Popen([str(args.solver_python), str(SOURCE / "scripts/p2_tdata2_solver.py"),
                               str(args.freeze)], stdout=subprocess.PIPE, env=env) as solver:
            try:
                times, weight_seconds, audit, reduced = collect(model, floor, frozen, solver)
            except BaseException:
                solver.terminate()
                raise
            if solver.wait() != 0:
                raise RuntimeError("refinement solver exited unsuccessfully")
    assert [row["id"] for row in audit] == frozen["audit_ids"]
    assert len({tuple(row["prng_key"]) for row in audit}) == len(audit) == 50
    results = {}
    for name, reductions in reduced.items():
        results[name] = []
        for draw in frozen["prior"]["draws"]:
            index = draw["fit"]
            arms = {arm: reductions[f"W4/{index}" if arm == "W4" else arm].report(q)
                    for arm, q in draw["thresholds"].items()}
            results[name].append({"fit": index, "calibration": draw["calibration"], "arms": arms})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {"scope": "50 matched independently seeded P2 engineering trajectories",
              "freeze_sha256": file_digest(args.freeze), "solver_timings": times,
              "weight_seconds": weight_seconds, "wall_seconds": time.perf_counter() - started,
              "reducer_process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss*1024),
              "peak_batch_trajectories": frozen["chunk"], "audit_trajectories": audit,
              "report_cell_convention": "zero-based cell; step 1 is t=.005; x = cell / 256",
              "retained_solver_fields_bytes": 0, "results": results}
    dump(args.out, report)
    for name, draws in results.items():
        print(name, [[(arm, round(v["conservative_maximum"]["epsilon_over_q"], 8))
                      for arm, v in draw["arms"].items()] for draw in draws])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    f, r = commands.add_parser("freeze"), commands.add_parser("run")
    for cmd, options in ((f, ("bundle", "kernel", "maintainer", "custodian", "design", "design-provenance", "out")),
                         (r, ("bundle", "freeze", "solver-python", "out"))):
        for name in options:
            cmd.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.action](args)
