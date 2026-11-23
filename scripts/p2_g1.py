"""Freeze, execute and independently replay the bounded P2 G1 rehearsal."""
import argparse
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import resource
import shutil
from statistics import fmean, median
import subprocess
import sys
import time

import numpy as np
from scipy.stats import betabinom

import p2_cpu_worker
import sf7.trajectory_field_audit as oracle
from p2_tdata2 import SOURCE, dump, file_digest
from p2_g1_worker import paths
from trajcp.conformal.core import calibrate, conformal_rank, field_score
from trajcp.data.weighted_audit import WeightedAudit
from trajcp.rollout.sensitivity import trajectory_key

ARMS = ("B01", "W1", "W3", "W4")
HORIZON = 200
Q = np.full(256, 1 / 256)
FROZEN_SOURCE = ("scripts/p2_g1.py", "scripts/p2_g1_solver.py", "scripts/p2_g1_worker.py",
                 "scripts/p2_tdata2.py", "scripts/p2_tdata2_solver.py", "scripts/p2_w34_worker.py",
                 "src/trajcp/data/exponax_gen.py", "src/trajcp/data/weighted_audit.py",
                 "src/trajcp/rollout/sensitivity.py", "src/trajcp/conformal/core.py",
                 "src/trajcp/conformal/lp.py", "src/trajcp/conformal/weights.py")


def load(path):
    return json.loads(path.read_text())


def code_hashes():
    return {**{name: file_digest(SOURCE / name) for name in FROZEN_SOURCE},
            "factory/p2_cpu_worker.py": file_digest(Path(p2_cpu_worker.__file__)),
            "factory/trajectory_field_audit.py": file_digest(Path(oracle.__file__))}


def freeze(out):
    cfg = load(SOURCE / "configs/task/p2_burgers1d_pilot.json")
    cfg.update(steps=HORIZON, master_seed=20261128)
    roles = {"train": {"seed": 20261128, "count": 32}}
    for i in range(2):
        for j, (role, n) in enumerate((("fit", 24), ("cal", 29), ("test", 30))):
            roles[role + str(i)] = {"seed": 20261129 + 3*i + j, "count": n}
    roles["audit"] = {"seed": 20261135, "count": 50}
    report = {
        "scope": "synthetic P2 section 14 G1 rehearsal only",
        "solver": cfg,
        "numerics": {"candidate": [2048, 20], "audit_comparator": [4096, 20], "projection": 256,
                     "step_ranges": [[1, 200], [101, 200]], "criterion": .001},
        "roles": roles, "arms": list(ARMS), "horizon": HORIZON, "chunk": 12, "alpha": .10,
        "training_seed": 813181, "train_steps": 120,
        "weights": {"smoothing": 3, "eta": .02, "M": 8, "cutoff": 8, "epsilon": .001,
                    "perturbation_seed": 813223, "w1_beta": .50,
                    "W1": "empirical 1-beta step maximum quantile, isotonic and floored on D_fit",
                    "W4": "q/LP/parametric, twofold width selection and D_fit refit"},
        "analysis": {"width": "mean of trajectory-time relative quadrature L2 full widths",
                     "width_gate": .20, "width_combination": "unweighted mean of the two draw arm widths",
                     "coverage_level": .95, "coverage_gate": "central 95% marginal Beta-Binomial tolerance for every arm and draw",
                     "timing_gate": 10., "timing_summary": "median paired W3/one ratio"},
        "timing": {"warmup": 2, "repeats": 7, "role": "audit", "input_chunk": 0},
        "worker_versions": {name: version(name) for name in ("torch", "numpy", "scipy")},
        "solver_versions": {"exponax": "0.2.1", "jax": "0.12.0.1", "jaxlib": "0.12.0.1", "numpy": "2.5.0", "scipy": "1.18.1"},
        "code": code_hashes(),
        "previous_evidence": {name: file_digest(SOURCE / "artifacts" / name) for name in (
            "p2_tdata2_report.json", "p2_refine_2048_report.json")},
    }
    dump(out, report)
    print("Frozen design", file_digest(out), flush=True)


def check_frozen(frozen):
    assert frozen["code"] == code_hashes()
    assert frozen["worker_versions"] == {name: version(name) for name in frozen["worker_versions"]}
    assert all(file_digest(SOURCE / "artifacts" / n) == h for n, h in frozen["previous_evidence"].items())


def worker(stage, plan, scope, out, checkpoint=None):
    args = [sys.executable, str(SOURCE / "scripts/p2_g1_worker.py"), stage, str(plan), str(scope), str(out)]
    if checkpoint is not None:
        args.extend(("--checkpoint", str(checkpoint)))
    subprocess.run(args, check=True)
    return load(out / "worker.json")


def content(path):
    return {p.relative_to(path).as_posix(): file_digest(p) for p in sorted(path.rglob("*")) if p.is_file()}


def scopes(frozen, data):
    keys, fingerprints = set(), set()
    result = {}
    for role, settings in frozen["roles"].items():
        local = []
        for path in paths(data / role):
            with np.load(path, allow_pickle=False) as parts:
                for i, initial, nu, words in zip(parts["ids"], parts["initial"], parts["nu"], parts["keys"], strict=True):
                    key, identity = tuple(words.tolist()), trajectory_key(initial, nu)
                    assert key not in keys and identity not in fingerprints
                    keys.add(key)
                    fingerprints.add(identity)
                    local.append(int(i))
        assert local == list(range(settings["count"]))
        result[role] = {"count": len(local), "seed": settings["seed"]}
    prior = set()
    for name, fn in (("audit_1024", "p2_tdata2_report.json"), ("audit_2048", "p2_refine_2048_report.json")):
        report = load(SOURCE / "artifacts" / fn)
        rows = report["audit_trajectories"] if name == "audit_1024" else [
            r for cohort in report["cohorts"] for r in cohort["audit_trajectories"]]
        prior.update(tuple(r["prng_key"]) for r in rows)
    assert len(prior) == 100 and not keys & prior
    return result


def fields(data, blind, role):
    for path in paths(data / role):
        with np.load(path, allow_pickle=False) as batch:
            ids = batch["ids"].tolist()
        n = path.stem.split("-")[-1]
        truth = np.load(data / role / f"truth-{n}.npy", allow_pickle=False)
        with np.load(blind / role / f"fields-{n}.npz", allow_pickle=False) as payload:
            forecast, w3 = payload["forecast"], payload["W3"]
        assert all(x.shape == (len(ids), HORIZON, 256) for x in (truth, forecast, w3))
        assert all(x.dtype == np.float64 and np.isfinite(x).all() for x in (truth, forecast, w3))
        yield n, ids, truth, forecast, w3


def arm_weights(w3, profile):
    return {"B01": np.ones_like(w3), "W1": np.broadcast_to(np.asarray(profile["W1"])[None, :, None], w3.shape),
            "W3": w3, "W4": w3 * np.asarray(profile["W4"])[None, :, None]}


def calibration(data, blind, draw, fit, alpha):
    scores = {arm: [] for arm in ARMS}
    for _, _, truth, forecast, w3 in fields(data, blind, f"cal{draw}"):
        for arm, weight in arm_weights(w3, fit).items():
            scores[arm].extend(field_score(truth - forecast, weight).tolist())
    n = len(scores["B01"])
    return {"n": n, "rank": conformal_rank(n, alpha), "scores": scores,
            "thresholds": {arm: calibrate(np.array(values), alpha) for arm, values in scores.items()}}


def compute_widths(truth, prediction, weight, q):
    discrepancy = np.abs(truth - prediction)
    covered = (discrepancy <= q * weight).all(axis=(1, 2))
    ratio = np.linalg.norm(weight, axis=-1) / np.linalg.norm(truth, axis=-1)
    return covered, 2*q*ratio.mean(axis=1), 2*q*ratio[:, -1]


def interval(m, k, n, tail):
    return [int(betabinom.ppf(prob, m, k, n+1-k)) for prob in (tail, 1-tail)]


def evaluate(frozen, evidence):
    data, blind = evidence / "data", evidence / "blind"
    alpha, analysis = frozen["alpha"], frozen["analysis"]
    draws = []
    for draw in range(2):
        state = load(evidence / f"fit{draw}/profiles.json")
        record = load(evidence / f"cal{draw}.json")
        rows = {arm: [] for arm in ARMS}
        for _, ids, truth, forecast, w3 in fields(data, blind, f"test{draw}"):
            for arm, weight in arm_weights(w3, state).items():
                q = record["thresholds"][arm]
                covered, widths, end = compute_widths(truth, forecast, weight, q)
                rows[arm].extend({"id": int(i), "covered": bool(c), "relative_width": float(w),
                                  "relative_width_end": float(t)}
                                 for i,c,w,t in zip(ids, covered, widths, end, strict=True))
        m, k, n = len(rows["B01"]), record["rank"], record["n"]
        tail = (1 - analysis["coverage_level"])/2
        bounds = interval(m, k, n, tail)
        arms = {}
        for arm, members in rows.items():
            covered = sum(row["covered"] for row in members)
            arms[arm] = {"threshold": record["thresholds"][arm], "covered": covered,
                         "coverage": covered/m, "beta_binomial_counts": bounds,
                         "bonus_family_reference_counts": interval(m, k, n, tail/8),
                         "passes_coverage": bounds[0] <= covered <= bounds[1],
                         "mean_relative_width": fmean(row["relative_width"] for row in members),
                         "mean_relative_width_end": fmean(row["relative_width_end"] for row in members),
                         "calibration_score_ties": n-len(set(record["scores"][arm])), "rows": members}
        draws.append({"draw": draw, "n_cal": n, "rank": k, "test": m, "W4_fit": state["W4_choice"], "arms": arms})
    widths = {arm: fmean(draw["arms"][arm]["mean_relative_width"] for draw in draws) for arm in ARMS}
    reductions = {arm: 1 - widths[arm]/widths["B01"] for arm in ARMS[1:]}
    speed = load(evidence / "benchmark/timing.json")
    ratio = median(row["W3"]/row["one"] for row in speed)
    return {"draws": draws, "mean_widths": widths, "width_reduction_vs_B01": reductions,
            "W3_median_ratio": ratio, "timing": speed, "audit": audit_numeric(frozen, evidence),
            "engineering_criteria": {"coverage": all(a["passes_coverage"] for d in draws for a in d["arms"].values()),
                                     "width": max(reductions.values()) >= analysis["width_gate"],
                                     "W3_cost": ratio <= analysis["timing_gate"]}, "alpha": alpha}


def audit_numeric(frozen, evidence):
    results = []
    data, blind = evidence / "data", evidence / "blind"
    for draw in range(2):
        state = load(evidence / f"fit{draw}/profiles.json")
        thresholds = load(evidence / f"cal{draw}.json")["thresholds"]
        windows = {start: {arm: WeightedAudit() for arm in ARMS} for start in (0, 100)}
        for suffix, ids, truth, _, w3 in fields(data, blind, "audit"):
            comparison = np.load(data / "audit" / f"compare-{suffix}.npy", allow_pickle=False)
            discrepancy = np.abs(comparison - truth)
            for arm, weight in arm_weights(w3, state).items():
                for start, arms in windows.items():
                    arms[arm].update(ids, discrepancy[:, start:], weight[:, start:])
        for start, arms in windows.items():
            results.append({"draw": draw, "step_offset": start,
                            "arms": {arm: result.report(thresholds[arm]) for arm,result in arms.items()}})
    return results


def independent_check(frozen, evidence, result):
    """Replay exact rank and direct bands with the separate scalar field reducer."""
    data, blind = evidence / "data", evidence / "blind"
    differences = []
    total = 0
    for draw in range(2):
        profiles = load(evidence / f"fit{draw}/profiles.json")
        calibrated = load(evidence / f"cal{draw}.json")
        expected = result["draws"][draw]["arms"]
        for role in ("cal", "test"):
            recalculated = {arm: [] for arm in ARMS}
            for _, ids, truth, pred, w3 in fields(data, blind, f"{role}{draw}"):
                for arm, weights in arm_weights(w3, profiles).items():
                    for i, unit in enumerate(ids):
                        row = oracle.field_row(unit, truth[i], pred[i], weights[i], Q)
                        recalculated[arm].append(row)
            for arm in ARMS:
                orig = calibrated["scores"][arm]
                checked = recalculated[arm]
                q = calibrated["thresholds"][arm]
                total += len(checked)
                if role == "cal":
                    scores = [row["score"] for row in checked]
                    np.testing.assert_array_equal(scores, orig)
                    k = math.ceil((len(scores)+1)*9/10)
                    assert k == calibrated["rank"] and sorted(scores)[k-1] == q
                else:
                    for i, (row, original) in enumerate(zip(checked, expected[arm]["rows"], strict=True)):
                        assert row["unit"] == original["id"] == i
                        assert (row["score"] <= q) == original["covered"]
                        for key, ref in (("relative_width", "relative"), ("relative_width_end", "end")):
                            computed = 2*q*row[ref]
                            differences.append(abs(original[key] - computed))
                            np.testing.assert_allclose(original[key], computed, rtol=5e-13, atol=5e-13)
    return {"status": "VERIFIED", "calibration_and_test_trajectory_arms": total,
            "max_absolute_width_delta": max(differences)}


def execute(args):
    frozen = load(args.freeze)
    check_frozen(frozen)
    evidence = args.evidence.resolve()
    assert not args.report.resolve().is_relative_to(evidence)
    evidence.mkdir()
    plan = evidence / "freeze.json"
    shutil.copyfile(args.freeze, plan)
    data = evidence / "data"
    env = dict(os.environ, PYTHONPATH=str(SOURCE / "src"), JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true")
    versions = subprocess.check_output([str(args.solver_python), "-c",
                                       "from importlib.metadata import version;import sys,json;"
                                       "print(json.dumps({n:version(n) for n in sys.argv[1:]}))",
                                       *frozen["solver_versions"]])
    assert json.loads(versions) == frozen["solver_versions"]
    started = time.perf_counter()
    subprocess.run([str(args.solver_python), str(SOURCE / "scripts/p2_g1_solver.py"), str(plan), str(data)],
                   env=env, check=True)
    dump(evidence / "scopes.json", scopes(frozen, data))
    print("Solver and identity separation complete", flush=True)
    checkpoint = evidence / "train/state.npy"
    work = {"train": worker("train", plan, data / "train", evidence / "train")}
    work["probe"] = worker("probe", plan, evidence, evidence / "probe", checkpoint)
    np.testing.assert_array_equal(np.load(evidence / "train/probe.npy", allow_pickle=False),
                                  np.load(evidence / "probe/probe.npy", allow_pickle=False))
    for draw in range(2):
        role = f"fit{draw}"
        work[role] = worker("fit", plan, data / role, evidence / role, checkpoint)
    blind, features = evidence / "blind", evidence / "blind_inputs"
    blind.mkdir()
    features.mkdir()
    for role in ("cal0", "cal1", "test0", "test1", "audit"):
        scoped = features / role
        scoped.mkdir()
        for path in paths(data / role):
            shutil.copyfile(path, scoped / path.name)
        work[role] = worker("forecast", plan, scoped, blind / role, checkpoint)
    work["benchmark"] = worker("benchmark", plan, features / "audit", evidence / "benchmark", checkpoint)
    dump(evidence / "worker_reports.json", work)
    for draw in range(2):
        profile = load(evidence / f"fit{draw}/profiles.json")
        dump(evidence / f"cal{draw}.json", calibration(data, blind, draw, profile, frozen["alpha"]))
    inventory = content(evidence)
    seal = evidence / "manifest.json"
    dump(seal, inventory)
    print("Evidence sealed, independently replaying results", flush=True)
    report = evaluate(frozen, evidence)
    report["independent_replay"] = independent_check(frozen, evidence, report)
    assert inventory == {k: v for k,v in content(evidence).items() if k != "manifest.json"}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report.update(freeze_sha256=file_digest(plan), evidence_manifest_sha256=file_digest(seal),
                  evidence_bytes=sum((evidence / p).stat().st_size for p in inventory),
                  wall_seconds=time.perf_counter() - started,
                  evaluator_peak_rss_bytes=int(rss if sys.platform == "darwin" else 1024*rss),
                  solver_time=load(data / "solver-timings.json"), workers=work,
                  roles=frozen["roles"])
    dump(args.report, report)
    print(report["engineering_criteria"], report["width_reduction_vs_B01"], report["W3_median_ratio"])


def verify(args):
    evidence = args.evidence.resolve()
    seal = evidence / "manifest.json"
    manifest = load(seal)
    assert {k: v for k,v in content(evidence).items() if k != "manifest.json"} == manifest
    old = load(args.report)
    frozen = load(evidence / "freeze.json")
    check_frozen(frozen)
    assert file_digest(evidence / "freeze.json") == old["freeze_sha256"]
    assert file_digest(seal) == old["evidence_manifest_sha256"]
    report = evaluate(frozen, evidence)
    assert all(report[k] == old[k] for k in report)
    independent = independent_check(frozen, evidence, report)
    assert independent == old["independent_replay"]
    dump(args.out, {**independent, "manifest_sha256": file_digest(seal),
                     "audit_and_g1_results_identical": True, "freeze_sha256": old["freeze_sha256"]})
    print(independent)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="action", required=True)
    opts = {"freeze": ("out",), "run": ("freeze", "solver-python", "evidence", "report"),
            "verify": ("evidence", "report", "out")}
    for mode, flags in opts.items():
        s = subs.add_parser(mode)
        for flag in flags:
            s.add_argument("--"+flag, required=True, type=Path)
    args = p.parse_args()
    {"freeze": lambda opts: freeze(opts.out), "run": execute, "verify": verify}[args.action](args)
