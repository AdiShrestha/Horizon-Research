"""Run the one frozen 2048/4096 P2 engineering qualification."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from zipfile import ZipFile

from p2_tdata2 import (
    AUDIT_CODE, SOURCE, collect, configure_torch, digest, dump, file_digest, load_state,
    prior_material, verify_factory, worker_hashes,
)

SCRIPT_NAMES = ["scripts/p2_refine_2048.py", "scripts/p2_refine_solver.py", *AUDIT_CODE]
CASES = [[1024, 20], [2048, 20], [4096, 20], [2048, 80], [4096, 80]]
COMPARISONS = {
    "spatial_2048_4096_at_20": ["g2048_s20", "g4096_s20"],
    "time_2048_20_to_80": ["g2048_s20", "g2048_s80"],
    "time_4096_20_to_80": ["g4096_s20", "g4096_s80"],
    "spatial_2048_4096_at_80": ["g2048_s80", "g4096_s80"],
}
COHORTS = {"paired": 20261126, "fresh": 20261127}
PAST = SOURCE / "artifacts"


def json_at(path):
    return json.loads(path.read_text())


def parent_record():
    freeze, report, verification = (PAST / f"p2_tdata2_{name}.json"
                                     for name in ("freeze", "report", "verification"))
    old, old_report, check = map(json_at, (freeze, report, verification))
    assert file_digest(freeze) == old_report["freeze_sha256"] == check["freeze_sha256"]
    assert file_digest(report) == check["report_sha256"]
    assert old["solver_config"]["master_seed"] == COHORTS["paired"]
    return old, old_report, {path.name: file_digest(path) for path in (freeze, report, verification)}


def freeze(args):
    old, previous, hashes = parent_record()
    kernel, replay = verify_factory(args.bundle, args.kernel, args.maintainer, args.custodian)
    assert kernel == old["kernel_merkle_root"]
    assert replay["family_id"] == old["verified_family"] and replay["ledger_head"] == old["verified_ledger_head"]
    assert prior_material(args.bundle, args.design, args.design_provenance) == old["prior"]
    assert file_digest(args.bundle) == old["bundle_sha256"]
    plan = {
        "scope": "bounded P2 2048/4096 refinement qualification at T=100",
        "cohorts": COHORTS, "parent_artifacts": hashes,
        "old_audit_keys": [row["prng_key"] for row in previous["audit_trajectories"]],
        "code": {name: file_digest(SOURCE / name) for name in SCRIPT_NAMES},
        "solver_cases": CASES, "comparisons": COMPARISONS,
        "weight_input_case": "g1024_s20", "decision_limit": 1e-3,
    }
    for key in ("audit_ids", "bundle_sha256", "chunk", "prior", "runtime", "solver_config", "solver_versions"):
        plan[key] = old[key]
    dump(args.out, plan)
    print(f"Frozen both cohorts and solver settings: {file_digest(args.out)}", flush=True)


def solver_command(python, mode, plan_file, out):
    return [str(python), str(SOURCE / "scripts/p2_refine_solver.py"), mode, str(plan_file), str(out)]


def verify_ids(plan, identities):
    old = set(plan["prior"]["excluded_prior_fingerprints"])
    cohort_checks, seen = {}, set()
    prior_keys = {tuple(key) for keys in identities["prior_keys"].values() for key in keys}
    assert len(prior_keys) == 65 + 24 + 85
    for name, cohort in identities["cohorts"].items():
        keys = {tuple(row) for row in cohort["keys"]}
        hashes = set(cohort["fingerprints"])
        assert len(keys) == len(hashes) == 50 and not keys & prior_keys and not hashes & (old | seen)
        if name == "paired":
            assert cohort["keys"] == plan["old_audit_keys"]
        else:
            assert not keys & {tuple(row) for row in plan["old_audit_keys"]}
        seen.update(hashes)
        cohort_checks[name] = {"unique_keys": len(keys), "unique_fingerprints": len(hashes),
                               "prior_non_audit_key_count": len(prior_keys), "prior_input_fingerprint_count": len(old),
                               "same_keys_as_previous_audit": name == "paired",
                               "key_and_fingerprint_overlap_across_cohorts": 0,
                               "fingerprint_commitment": digest(json.dumps(cohort["fingerprints"]).encode())}
    return cohort_checks


def evaluate(name, model, floor, plan, identities, args, folder, env):
    solver_config = dict(plan["solver_config"], master_seed=plan["cohorts"][name])
    cohort = dict(plan, solver_config=solver_config)
    plan_path, measure = folder / f"{name}-plan.json", folder / f"{name}-solver-rss.json"
    dump(plan_path, cohort)
    started = time.perf_counter()
    with subprocess.Popen(solver_command(args.solver_python, "stream", plan_path, measure),
                          stdout=subprocess.PIPE, env=env) as solver:
        try:
            times, weight_seconds, audit, reductions = collect(model, floor, cohort, solver)
        except BaseException:
            solver.terminate()
            raise
        if solver.wait() != 0:
            raise RuntimeError("P2 refinement solver exited unsuccessfully")
    assert [row["id"] for row in audit] == plan["audit_ids"]
    assert [row["prng_key"] for row in audit] == identities[name]["keys"]
    results = {}
    for label, family in reductions.items():
        results[label] = []
        for draw in plan["prior"]["draws"]:
            index = draw["fit"]
            arms = {arm: family[f"W4/{index}" if arm == "W4" else arm].report(q)
                    for arm, q in draw["thresholds"].items()}
            results[label].append({"fit": index, "calibration": draw["calibration"], "arms": arms})
    baseline = times["g1024_s20"]["solve_seconds"]
    return {
        "cohort": name, "seed": plan["cohorts"][name], "audit_trajectories": audit,
        "results": results, "solver_timings": times, "weight_seconds": weight_seconds,
        **json_at(measure), "solve_cost_vs_1024_20": {label: x["solve_seconds"] / baseline for label, x in times.items()},
        "wall_seconds": time.perf_counter() - started,
    }


def run(args):
    plan = json_at(args.freeze)
    assert plan["cohorts"] == COHORTS and plan["solver_cases"] == CASES and plan["comparisons"] == COMPARISONS
    assert plan["decision_limit"] == .001 and plan["audit_ids"] == list(range(50))
    assert parent_record()[2] == plan["parent_artifacts"]
    assert {name: file_digest(SOURCE / name) for name in SCRIPT_NAMES} == plan["code"]
    assert {name: version(name) for name in plan["runtime"]} == plan["runtime"]
    actual = subprocess.check_output([str(args.solver_python), "-c",
                                     "import json,sys;from importlib.metadata import version;"
                                     "print(json.dumps({x:version(x) for x in sys.argv[1:]}))",
                                     *plan["solver_versions"]])
    assert json.loads(actual) == plan["solver_versions"]
    assert file_digest(args.bundle) == plan["bundle_sha256"]
    assert worker_hashes(plan["prior"]["worker_code"]) == plan["prior"]["worker_code"]
    configure_torch()
    env = dict(os.environ, PYTHONPATH=str(SOURCE / "src"), JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="p2-refine-") as tmp, ZipFile(args.bundle) as z:
        folder = Path(tmp)
        identity_file = folder / "identities.json"
        subprocess.run(solver_command(args.solver_python, "identities", args.freeze, identity_file), env=env, check=True)
        original_ids = json_at(identity_file)
        checks = verify_ids(plan, original_ids)
        state = z.read("runs/725/models/fno/state.npy")
        assert digest(state) == plan["prior"]["state_sha256"]
        checkpoint = folder / "state.npy"
        checkpoint.write_bytes(state)
        model, floor = load_state(checkpoint)
        assert floor == plan["prior"]["training_floor"]
        chosen = tuple(plan["cohorts"]) if args.cohort == "both" else (args.cohort,)
        reports = []
        for name in chosen:
            reports.append(evaluate(name, model, floor, plan, original_ids["cohorts"], args, folder, env))
            print(f"{name} complete, {reports[-1]['wall_seconds']:.2f}s", flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    dump(args.out, {"freeze_sha256": file_digest(args.freeze), "identity_checks": checks,
                    "cohorts": reports, "retained_solver_fields_bytes": 0, "peak_batch_trajectories": plan["chunk"],
                    "consumer_peak_rss_bytes": int(rss if sys.platform == "darwin" else 1024 * rss),
                    "total_wall_seconds": time.perf_counter() - started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="action", required=True)
    f, r = cmds.add_parser("freeze"), cmds.add_parser("run")
    for cmd, options in ((f, ("bundle", "kernel", "maintainer", "custodian", "design", "design-provenance", "out")),
                         (r, ("bundle", "freeze", "solver-python", "out"))):
        for option in options:
            cmd.add_argument("--" + option, type=Path, required=True)
    r.add_argument("--cohort", choices=("both", *COHORTS), default="both")
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.action](args)
