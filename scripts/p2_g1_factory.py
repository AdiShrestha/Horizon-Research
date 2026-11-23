"""Run and audit the native Factory engineering G1 route on the frozen study-side inputs."""
import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from p2_tdata2 import SOURCE, dump, file_digest
from p2_g1 import ARMS, code_hashes, load

STUDY = "p2-g1-native"


def cli(home, kernel, *args, cwd=None, code=0):
    env = dict(os.environ, SF7_HOME=str(home), PYTHONPATH=str(kernel), PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-B", "-m", "sf7.cli", *map(str, args)], env=env,
                       text=True, capture_output=True, cwd=cwd, timeout=650)
    if p.returncode != code:
        raise RuntimeError((args, p.stdout, p.stderr))
    return json.loads(p.stdout)


def stage_inputs(prior, inbox):
    inbox.mkdir()
    initial, parameters, truth, ids = [], [], [], []
    # The current native split contract partitions one exact 83-unit pool.
    for role in ("train", "fit0", "cal0", "test0"):
        for source in sorted((prior / "data" / role).glob("input-*.npz")):
            with np.load(source, allow_pickle=False) as files:
                ids.extend(f"{role}-{int(i):04d}" for i in files["ids"])
                initial.append(files["initial"])
                parameters.append(files["nu"])
            truth.append(np.load(source.with_name(source.name.replace("input-", "truth-")).with_suffix(".npy"),
                                 allow_pickle=False))
    for label, pieces in (("initial", initial), ("parameters", parameters), ("truth", truth)):
        np.save(inbox / (label + ".npy"), np.ascontiguousarray(np.concatenate(pieces), dtype="<f8"), allow_pickle=False)
    dump(inbox / "trajectories.json", {"schema": "sf7-trajectory-data-1", "task": "P2", "train_ids": ids[:32],
                                     "pool_ids": ids[32:], "files": {n: n+".npy" for n in ("initial", "parameters", "truth")},
                                     "provenance": {"solver": "JAX CPU float64 Exponax ETDRK4 2048/20",
                                                    "projection": "256 cells periodic Fourier",
                                                    "initial_condition_rule": "existing frozen synthetic G1 role inputs",
                                                    "seed": 20261128}})
    return {path.name: file_digest(path) for path in inbox.iterdir()}


def native_plan(frozen):
    worker = {"entrypoint": "source/worker.py", "config": {"plan": frozen}}
    arms = {"B01": {"kind": "flat", "model": "FNO"}}
    for name in ("W1", "W3", "W4"):
        arms[name] = {"kind": "intrinsic" if name == "W3" else "fitted", "model": "FNO",
                      "worker": {**worker, "config": {"plan": frozen, "arm": name}}}
    return {"kind": "trajectory-band", "schema_version": 7, "intent": "engineering", "study_id": STUDY,
            "dataset": {"id": "P2", "manifest": "trajectories.json", "field_shape": [256], "max_horizon": 200,
                        "parameter_count": 1, "quadrature": [1/256]*256, "time_step": .005,
                        "physical_units": "dimensionless Burgers field", "channels": "scalar"},
            "evaluation": {"horizons": [200], "alphas": [.1], "primary_horizon": 200, "primary_alpha": .1,
                           "score": "sup", "metrics": ["simultaneous_coverage", "relative_width",
                                                       "relative_width_quarter", "relative_width_end"]},
            "training_seeds": [frozen["training_seed"]], "arms": arms,
            "models": {"FNO": {**worker, "state": {"format": "float32-npy-v1", "values": 10886}}},
            "draws": {"train": 32, "fit": 24, "calibration": 29, "test": 30, "fit_draws": 2, "calibration_draws": 1},
            "comparisons": [{"candidate": arm, "baseline": "B01", "role": "primary" if arm == "W4" else "secondary"}
                            for arm in ("W1", "W3", "W4")],
            "inference": {"method": "paired-by-draw", "seed": 345, "bootstrap_replicates": 63, "permutation_flips": 255},
            "stream": {"chunk_trajectories": frozen["chunk"]},
            "field_audit": {"mode": "p2-g1-retain-fields", "max_bytes": 192*1024**2}}


def stage_code(root, kernel_source):
    scripts = ("p2_g1_factory_worker.py", "p2_tdata2.py", "p2_w34_worker.py", "p2_g1_worker.py")
    for name in scripts:
        shutil.copyfile(SOURCE / "scripts" / name, root / ("worker.py" if name == scripts[0] else name))
    shutil.copyfile(kernel_source / "engineering/p2_cpu_worker.py", root / "p2_cpu_worker.py")
    shutil.copytree(SOURCE / "src/trajcp", root / "trajcp", ignore=shutil.ignore_patterns("__pycache__"))


def independent(source, run, raw):
    data = load(source / "trajectories.json")
    ids = {name: i for i,name in enumerate(data["train_ids"] + data["pool_ids"])}
    truth = np.load(source / "truth.npy", mmap_mode="r", allow_pickle=False)
    commit = load(run / "manifest.json")
    max_abs, checks = 0., 0
    summaries = []
    for native, recorded in zip(commit["draws"], raw, strict=True):
        draw = native["draw"]
        values = {arm: {} for arm in ARMS}
        for chunk in native["field_chunks"]:
            y = truth[[ids[i] for i in chunk["order"]]]
            pred = np.load(run / chunk["forecasts"]["FNO"], allow_pickle=False)
            hits = np.abs(y - pred)
            assert y.shape[1] == 200
            for arm in ARMS:
                weights = np.ones_like(y) if arm == "B01" else np.load(run / chunk["weights"][arm], allow_pickle=False)
                scores = (hits / weights).max(axis=(1, 2))
                ratios = np.sqrt(np.mean(weights**2, axis=-1) / np.mean(y*y, axis=-1))
                for unit, score, ratio, error, weight in zip(chunk["order"], scores, ratios, hits, weights, strict=True):
                    values[arm][unit] = (float(score), float(ratio.mean()), float(ratio[49]), float(ratio[-1]),
                                         error, weight)
        row = {"fit": draw["fit_index"], "arms": {}}
        for arm, mapping in values.items():
            cal = [mapping[i][0] for i in draw["calibration"]]
            k = math.ceil(Fraction(9, 10)*(len(cal)+1))
            q = sorted(cal)[k-1]
            outcome = recorded["arms"][arm][0]
            assert q == outcome["threshold"] and k == outcome["rank"] == 27 and outcome["horizon"] == 200
            np.testing.assert_array_equal(cal, outcome["calibration_scores"])
            count = 0
            for unit, native_row in zip(draw["test"], outcome["rows"], strict=True):
                _, avg, quarter, end, error, weight = mapping[unit]
                covered = bool((error <= weight*q).all())
                assert native_row["unit"] == unit and covered == native_row["simultaneous_coverage"]
                count += covered
                for metric, coefficient in (("relative_width", avg), ("relative_width_quarter", quarter),
                                             ("relative_width_end", end)):
                    expected = 2*q*coefficient
                    max_abs = max(max_abs, abs(native_row[metric]-expected))
                    np.testing.assert_allclose(native_row[metric], expected, atol=5e-13, rtol=5e-13)
            checks += len(cal)+len(draw["test"])
            row["arms"][arm] = {"q": q, "covered": count}
        summaries.append(row)
    roles = [item["draw"] for item in commit["draws"]]
    overlap = {key: len(set(roles[0][key]) & set(roles[1][key])) for key in ("fit", "calibration", "test")}
    return {"compared_trajectory_arms": checks, "max_absolute_width_difference": max_abs,
            "draws": summaries, "same_role_cross_draw_overlap": overlap}


def receipts(run):
    result = {}
    for path in sorted(run.rglob("evidence.json")):
        receipt = load(path)
        telemetry = load(path.with_name("telemetry.json"))
        stage = telemetry["stage"]
        conf = receipt["confinement"]
        assert receipt["exit_code"] == 0 and conf["policy"] == "deny_default" and not conf["gpu"]
        assert any(p.endswith("/raw/truth.npy") for p in conf["probed_denied_reads"])
        assert any(p.endswith("/trajectory/draw.key") for p in conf["probed_denied_reads"])
        truths = [p for p in conf["read"] if p.endswith("/truth.npy")]
        assert len(truths) == int(stage in ("train", "fit_weights"))
        if truths:
            assert truths == [str(path.parent / "input/truth.npy")]
        if stage == "weights":
            arm = path.parent.name
            fits = [p for p in conf["read"] if "/fit/" in p]
            assert len(fits) == int(arm != "W3") and all(p.endswith("/"+arm+"/state.json") for p in fits)
            stage += "/" + arm
        result.setdefault(stage, []).append(dict(telemetry, kernel_seconds=receipt["seconds"]))
    assert {k: len(v) for k,v in result.items()} == {
        "train": 1, "load_check": 1, "fit_weights": 4, "benchmark": 1, "forecast": 10,
        "weights/W1": 10, "weights/W3": 10, "weights/W4": 10}
    return result


def run(args):
    prior = args.prior
    shell = args.out
    shell.mkdir()
    frozen = load(prior / "freeze.json")
    previous_code = {name: sha for name, sha in frozen["code"].items()
                     if name != "factory/trajectory_field_audit.py"}
    current_code = code_hashes()
    assert all(current_code[name] == sha for name, sha in previous_code.items())
    assert file_digest(prior / "manifest.json") == load(SOURCE / "artifacts/p2_g1_report.json")["evidence_manifest_sha256"]
    for name, sha in load(prior / "manifest.json").items():
        assert file_digest(prior / name) == sha
    home, ws, release = shell / "owner", shell / "study", shell / "release"
    factory = args.factory / "factory"
    key = shell / "maintainer.key"
    maintainer = cli(home, factory, "release", "keygen", "--key", key)["public_key"]
    cli(home, factory, "release", "build", "--key", key, "--out", release)
    cli(home, factory, "trust", "add", maintainer)
    installed = Path(cli(home, factory, "kernel", "install", release)["path"])
    kernel = cli(home, installed, "kernel", "verify")
    cli(home, installed, "owner", "init")
    study = cli(home, installed, "open", ws, "--study-id", STUDY, "--kind", "trajectory-band")
    vault = home / "vaults" / STUDY
    source = vault / "inbox"
    hashes = stage_inputs(prior, source)
    dump_path = ws / "project/sources.json"
    dump_path.write_text(json.dumps({"schema_version": 7, "artifacts": [
        {"name": name, "sha256": digest, "bytes": (source / name).stat().st_size, "license": "CC0-1.0",
         "digest_source": "frozen synthetic study-side G1 rehearsal"} for name,digest in hashes.items()]}, indent=2))
    (ws / "project/plan.json").write_text(json.dumps(native_plan(frozen), indent=2))
    stage_code(ws / "source", factory)
    assert cli(home, installed, "status", ws)["state"] == "TRAJECTORY_SEAL_SOURCES"
    assert cli(home, installed, "advance", ws)["state"] == "TRAJECTORY_SOURCE_ATTESTATION"
    cli(home, installed, "owner", "attest-sources", ws)
    assert cli(home, installed, "advance", ws)["state"] == "TRAJECTORY_OWNER_APPROVAL"
    cli(home, installed, "owner", "countersign", ws)
    assert not (ws / "evaluation.json").exists()
    assert cli(home, installed, "status", ws)["state"] == "TRAJECTORY_CONFIRM"
    started = time.perf_counter()
    assert cli(home, installed, "advance", ws)["state"] == "TRAJECTORY_COMPLETE"
    elapsed = time.perf_counter()-started
    assert not cli(home, installed, "advance", ws)["executed"]
    result = load(ws / "evaluation.json")
    store = vault / "trajectory"
    job = store / "runs" / str(frozen["training_seed"])
    raw = load(store / "scored.json")
    oracle = independent(vault / "raw", job, raw)
    stages = receipts(job)
    state_equal = file_digest(job / "models/FNO/state.npy") == file_digest(prior / "train/state.npy")
    archive = Path(cli(home, installed, "bundle", ws, "--include-targets")["path"])
    verifier = shell / "verifier"
    verifier.mkdir()
    keys = ["--maintainer-key", maintainer, "--custodian-key", study["custodian_key"]]
    passed = cli(verifier / "owner", installed, "verify-bundle", archive, *keys, cwd=verifier)
    assert passed["status"] == "VERIFIED" and passed["rescored_targets"] and passed["paired_draws"] == 2
    dump(shell / "qualification.json", {"kernel": kernel, "evidence": result, "independent": oracle,
                                          "scope_receipts": stages, "clean_replay": passed,
                                          "same_state_as_study_side": state_equal, "source_hashes": hashes,
                                          "archive_sha256": file_digest(archive), "archive_bytes": archive.stat().st_size,
                                          "confirm_seconds": elapsed})
    shutil.copyfile(archive, shell / "native-g1-evidence.zip")
    (shell / "maintainer.pub").write_text(maintainer+"\n")
    (shell / "custodian.pub").write_text(study["custodian_key"]+"\n")
    print(result["g1"]["criteria"], elapsed, oracle["max_absolute_width_difference"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("factory", "prior", "out"):
        p.add_argument("--"+flag, type=Path, required=True)
    run(p.parse_args())
