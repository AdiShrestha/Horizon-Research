"""Freeze and qualify one W3 schedule change using the installed Factory 7.2.4 route."""
import argparse
import base64
from contextlib import contextmanager
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import secrets
import shutil
from statistics import median
import subprocess
import sys
import threading
import time
from zipfile import ZipFile

import numpy as np

from p2_g1 import FROZEN_SOURCE, load
from p2_g1_factory import cli, native_plan, stage_code
from p2_tdata2 import SOURCE, dump, file_digest
from trajcp.rollout.sensitivity import trajectory_key

STUDY = "p2-w3-schedule"
SCHEDULES = ("joint", "serial")
ARMS = ("B01", "W1", "W3", "W4")


def code_hashes(factory):
    names = (*FROZEN_SOURCE, "scripts/p2_g1_factory.py", "scripts/p2_g1_factory_worker.py",
             "scripts/p2_w3_schedule.py", "scripts/p2_w3_schedule_worker.py")
    return {**{n: file_digest(SOURCE/n) for n in names},
            "factory-p2_cpu_worker": file_digest(factory / "factory/engineering/p2_cpu_worker.py")}


def freeze(args):
    out, factory = args.out, args.factory
    out.mkdir()
    previous = factory / "dist/p2-g1-7.2.4-evidence.zip"
    home = out / "freeze-verifier"
    verified = cli(home, args.kernel, "verify-bundle", previous,
                   "--maintainer-key", (factory/"dist/p2-g1-7.2.4-maintainer.pub").read_text().strip(),
                   "--custodian-key", (factory/"dist/p2-g1-7.2.4-custodian.pub").read_text().strip())
    assert verified["status"] == "VERIFIED" and verified["rescored_targets"]
    paths = SOURCE / "artifacts"
    original = load(paths / "p2_g1_freeze.json")
    prev_report = load(factory / "dist/p2-g1-7.2.4-qualification.json")
    seeds = {"benchmark": {"seed": 20261136, "count": 12}, "pool": {"seed": 20261137, "count": 83}}
    with ZipFile(previous) as z:
        signed = json.loads(z.read("registration.json"))["spec"]["plan"]
        assert signed["models"]["FNO"]["config"]["plan"] == original
        root = f"runs/{original['training_seed']}/"
        state = z.read(root+"models/FNO/state.npy")
        profiles = {arm: json.loads(z.read(root+f"fit/0/{arm}/state.json")) for arm in ("W1", "W4")}
        commitments = json.loads(z.read(root+"manifest.json"))["files"]
        assert sha256(state).hexdigest() == commitments["models/FNO/state.npy"]
        for arm in profiles:
            path = f"fit/0/{arm}/state.json"
            assert sha256(z.read(root+path)).hexdigest() == commitments[path]
        draw = json.loads(z.read(root+"manifest.json"))["draws"][0]
        reductions = json.loads(z.read(root+draw["reductions"]))["arms"]
        cal = set(draw["draw"]["calibration"])
        for arm in ARMS:
            scores = sorted(r["score"] for r in reductions[arm] if r["unit"] in cal)
            assert scores[26] == prev_report["independent"]["draws"][0]["arms"][arm]["q"]
    key = secrets.token_bytes(32)
    fd = os.open(out/"draw.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key)
    cap = {
        "forecast": {"atol": 2e-5, "rtol": 0.}, "envelope": {"atol": 5e-4, "rtol": .002},
        "score_and_q": {"atol": .005, "rtol": .002}, "width": {"atol": .01, "rtol": .002},
        "test_inclusion_flips": 0,
    }
    plan = {"scope": "single prospective CPU scheduling comparison, no weight ablation", "version": "7.2.4",
            "schedules": list(SCHEDULES), "profiles": profiles, "state_sha256": sha256(state).hexdigest(),
            "state_npy_base64": base64.b64encode(state).decode("ascii"), "locked_earlier_fit": 0,
            "previous_q": prev_report["independent"]["draws"][0]["arms"],
            "prior_archive_sha256": file_digest(previous), "prior_verifier": verified,
            "draw_key_commitment": sha256(key).hexdigest(), "roles": seeds, "chunk": 12,
            "solver": {**original["solver"], "master_seed": seeds["benchmark"]["seed"]},
            "solver_versions": original["solver_versions"], "weights": original["weights"],
            "samples": {"benchmark": 12, "fit": 24, "calibration": 29, "test": 30},
            "timing": {"threads": 2, "batch": 12, "warmups": 2, "replicates_per_session": 7,
                       "order": "alternate one/joint/serial and reverse; opposite starting order in second session",
                       "qualification": "serial median W3/standalone <=10 in both confined sessions; all parity gates pass",
                       "gate": 10., "host_sample_interval_seconds": 2.,
                       "load_policy": "report all replicates and observed load and power, no exclusions"},
            "tolerances": cap, "code": code_hashes(factory),
            "previous_reports": {n: file_digest(paths/n) for n in ("p2_g1_report.json", "p2_tdata2_report.json",
                                                                   "p2_refine_2048_report.json")},
            "worker_versions": {n: version(n) for n in ("torch", "numpy", "scipy")}}
    dump(out / "freeze.json", plan)
    print("Both schedules prospectively frozen:", file_digest(out / "freeze.json"), flush=True)


def frozen_config(frozen, factory):
    assert frozen["code"] == code_hashes(factory)
    assert frozen["worker_versions"] == {n: version(n) for n in frozen["worker_versions"]}
    assert all(file_digest(SOURCE/"artifacts"/n) == h for n,h in frozen["previous_reports"].items())


def build_inventory(data, out):
    out.mkdir()
    chunks = [[], [], []]
    names = {"benchmark": [], "pool": []}
    keys, inputs = set(), set()
    for role in names:
        scope = data/role
        for batch in sorted(scope.glob("input-*.npz")):
            with np.load(batch, allow_pickle=False) as raw:
                ids, u, p, words = (raw[n] for n in ("ids","initial","nu","keys"))
                for i, field, param, w in zip(ids, u, p, words, strict=True):
                    key, fp = tuple(w.tolist()), trajectory_key(field,param)
                    assert key not in keys and fp not in inputs
                    keys.add(key)
                    inputs.add(fp)
                    names[role].append(f"{role}-{int(i):04d}")
                chunks[0].append(u)
                chunks[1].append(p)
            combined = batch.with_name(batch.name.replace("input-", "truth-")).with_suffix(".npy")
            chunks[2].append(np.load(combined, allow_pickle=False))
    for path, values in zip(("initial","parameters","truth"),chunks,strict=True):
        np.save(out/(path+".npy"), np.ascontiguousarray(np.concatenate(values), dtype="<f8"), allow_pickle=False)
    dump(out/"trajectories.json", {"schema":"sf7-trajectory-data-1", "task":"P2",
                                  "train_ids":names["benchmark"], "pool_ids":names["pool"],
                                  "files":{n:n+".npy" for n in ("initial","parameters","truth")},
                                  "provenance":{"solver":"CPU float64 Exponax at 2048/20", "projection":"256-cell Fourier",
                                                "initial_condition_rule":"fresh frozen engineering keys 20261136/20261137",
                                                "seed":20261136}})
    return keys, inputs


def separations(keys, fingerprints):
    earlier = SOURCE/"data/p2_g1_rehearsal/data"
    old_keys, old_fingerprints = set(), set()
    for path in earlier.glob("*/input-*.npz"):
        with np.load(path, allow_pickle=False) as batch:
            old_keys.update(tuple(row) for row in batch["keys"].tolist())
            old_fingerprints.update(trajectory_key(u,p) for u,p in zip(batch["initial"],batch["nu"],strict=True))
    earlier_audit = load(SOURCE/"artifacts/p2_refine_2048_report.json")["cohorts"]
    old_keys.update(tuple(row["prng_key"]) for cohort in earlier_audit for row in cohort["audit_trajectories"])
    assert len(keys) == len(fingerprints) == 95
    assert len(old_keys) == 348 and len(old_fingerprints) == 248 and not keys & old_keys and not fingerprints & old_fingerprints
    return {"fresh_keys": len(keys), "prior_key_count":len(old_keys), "prior_input_fingerprints":len(old_fingerprints),
            "key_overlap":0, "input_overlap":0}


def phase_plan(frozen, schedule):
    original = load(SOURCE/"artifacts/p2_g1_freeze.json")
    plan = native_plan(original)
    plan["study_id"] = STUDY
    plan["draws"].update(train=12, fit_draws=1)
    plan["field_audit"]["max_bytes"] = 112*1024**2
    for spec in [*plan["models"].values(), *(a["worker"] for a in plan["arms"].values() if "worker" in a)]:
        arm = spec["config"].get("arm")
        spec["config"] = {"frozen":frozen, "schedule":schedule, **({"arm":arm} if arm else {})}
    return plan


def register(factory, kernel, out, frozen, schedule, source, key):
    setup = out / schedule
    setup.mkdir()
    home, ws = setup / "owner", setup / "study"
    pkg = load(factory/"dist/package_validation_7.2.4-development.json")
    cli(home, kernel, "trust", "add", pkg["signed_kernel_release"]["signer"])
    installed = Path(cli(home,kernel,"kernel","install",kernel)["path"])
    assert cli(home, installed, "kernel", "verify")["level"] == "K2"
    cli(home, installed, "owner", "init")
    study = cli(home,installed,"open",ws,"--study-id",STUDY,"--kind","trajectory-band")
    vault = home/"vaults"/STUDY
    shutil.copytree(source,vault/"inbox")
    artifacts = [{"name":p.name,"sha256":file_digest(p),"bytes":p.stat().st_size,"license":"CC0-1.0",
                  "digest_source":"prospective frozen scheduling qualification"}
                 for p in sorted(source.iterdir())]
    (ws/"project/sources.json").write_text(json.dumps({"schema_version":7,"artifacts":artifacts}, indent=2))
    (ws/"project/plan.json").write_text(json.dumps(phase_plan(frozen,schedule),indent=2))
    stage_code(ws/"source",factory/"factory")
    shutil.copyfile(SOURCE/"scripts/p2_g1_factory_worker.py",ws/"source/p2_g1_factory_worker.py")
    shutil.copyfile(SOURCE/"scripts/p2_w3_schedule_worker.py",ws/"source/worker.py")
    assert cli(home,installed,"status",ws)["state"] == "TRAJECTORY_SEAL_SOURCES"
    assert cli(home,installed,"advance",ws)["state"] == "TRAJECTORY_SOURCE_ATTESTATION"
    cli(home,installed,"owner","attest-sources",ws)
    (vault/"trajectory").mkdir(exist_ok=True)
    p = vault/"trajectory/draw.key"
    assert not p.exists()
    fd = os.open(p,os.O_WRONLY | os.O_CREAT | os.O_EXCL,0o600)
    with os.fdopen(fd,"wb") as f:
        f.write(key)
    assert cli(home,installed,"advance",ws)["state"] == "TRAJECTORY_OWNER_APPROVAL"
    cli(home,installed,"owner","countersign",ws)
    assert cli(home,installed,"status",ws)["state"] == "TRAJECTORY_CONFIRM"
    assert not (ws/"evaluation.json").exists()
    print(schedule,"sealed and countersigned",flush=True)
    return {"home":home,"workspace":ws,"vault":vault,"kernel":installed,"custodian":study["custodian_key"]}


def host_sample():
    adapter = subprocess.run(["/usr/bin/pmset","-g","ps"], capture_output=True, text=True, timeout=4)
    return {"timestamp_ns":time.time_ns(), "load_1_5_15":list(os.getloadavg()),
            "power_source":adapter.stdout.strip(), "power_status":adapter.returncode}


@contextmanager
def host_notes(interval):
    stop = threading.Event()
    samples = [host_sample()]
    def worker():
        while not stop.wait(interval):
            samples.append(host_sample())
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        yield samples
    finally:
        stop.set()
        thread.join()
        samples.append(host_sample())


def verify_receipts(folder):
    counts = {}
    for path in folder.rglob("evidence.json"):
        receipt = load(path)
        conf = receipt["confinement"]
        telemetry = load(path.with_name("telemetry.json"))
        stage = telemetry["stage"]
        assert receipt["exit_code"] == 0 and conf["policy"] == "deny_default" and not conf["gpu"]
        assert telemetry["threads"] == 2
        assert any(x.endswith("/raw/truth.npy") for x in conf["probed_denied_reads"])
        assert any(x.endswith("/trajectory/draw.key") for x in conf["probed_denied_reads"])
        assert len([x for x in conf["read"] if x.endswith("/truth.npy")]) == int(stage in ("train","fit_weights"))
        if stage == "weights":
            arm = path.parent.name
            fits = [r for r in conf["read"] if "/fit/" in r]
            assert len(fits) == int(arm != "W3") and all(r.endswith("/"+arm+"/state.json") for r in fits)
            stage += "/"+arm
        counts.setdefault(stage, []).append({**telemetry, "wall_seconds":receipt["seconds"]})
    assert {k:len(v) for k,v in counts.items()} == {
        "train":1,"load_check":1,"benchmark":1,"fit_weights":2,"forecast":5,"weights/W1":5,"weights/W3":5,"weights/W4":5}
    return counts


def confirm(factory, params, frozen, out, schedule):
    home, ws, vault, kernel = (params[k] for k in ("home","workspace","vault","kernel"))
    started = time.perf_counter()
    with host_notes(frozen["timing"]["host_sample_interval_seconds"]) as host:
        assert cli(home,kernel,"advance",ws)["state"] == "TRAJECTORY_COMPLETE"
    elapsed = time.perf_counter()-started
    assert cli(home,kernel,"status",ws)["state"] == "TRAJECTORY_COMPLETE"
    assert not cli(home,kernel,"advance",ws)["executed"]
    benchmark_folder = vault/"trajectory/runs"/str(load(SOURCE/"artifacts/p2_g1_freeze.json")["training_seed"])
    checkpoint = benchmark_folder/"models/FNO/state.npy"
    assert file_digest(checkpoint) == frozen["state_sha256"]
    for arm in ("W1","W4"):
        assert load(benchmark_folder/f"fit/0/{arm}/state.json")["profile"] == frozen["profiles"][arm]["profile"]
    scope = verify_receipts(benchmark_folder)
    bundle = Path(cli(home,kernel,"bundle",ws,"--include-targets")["path"])
    # Bind the signer to the development distribution published before this experiment.
    public_key = load(factory/"dist/package_validation_7.2.4-development.json")["signed_kernel_release"]["signer"]
    verify_dir = out/schedule/"clean"
    verify_dir.mkdir()
    proof = cli(verify_dir/"owner",kernel,"verify-bundle",bundle,
                "--maintainer-key",public_key,"--custodian-key",params["custodian"],cwd=verify_dir)
    assert proof["status"] == "VERIFIED" and proof["rescored_targets"] and proof["paired_draws"] == 1
    target = out/(schedule+"-evidence.zip")
    shutil.copyfile(bundle,target)
    result = {"elapsed_seconds":elapsed, "evidence_sha256":file_digest(target), "evidence_bytes":target.stat().st_size,
              "maintainer":public_key,"custodian":params["custodian"], "verifier":proof,
              "telemetry":scope,"host":host,"evaluation":load(ws/"evaluation.json")}
    print(schedule,"clean replay verified",flush=True)
    return result


def archive_arrays(bundle):
    with ZipFile(bundle) as z:
        decode = lambda name: json.loads(z.read(name))
        registration = decode("registration.json")
        seed = registration["spec"]["plan"]["training_seeds"][0]
        root = f"runs/{seed}/"
        man = decode(root+"manifest.json")
        record, = man["draws"]
        split_digest = registration["spec"]["split"]
        timing = decode(root+"models/FNO/benchmark/timing.json")
        raw = decode("raw/trajectories.json")
        return split_digest, record, timing, raw


def field_payload(z, root, chunk, arm):
    from io import BytesIO
    prediction = np.load(BytesIO(z.read(root+chunk["forecasts"]["FNO"])), allow_pickle=False)
    weights = np.ones_like(prediction) if arm == "B01" else np.load(
        BytesIO(z.read(root+chunk["weights"][arm])),allow_pickle=False)
    return prediction, weights


def metric(a, b, tol):
    raw = np.abs(b-a)
    scaled = raw/(tol["atol"]+tol["rtol"]*np.abs(a))
    worst = np.unravel_index(np.argmax(scaled), scaled.shape)
    abs_peak = np.unravel_index(np.argmax(raw),raw.shape)
    return {"max_absolute":float(raw[abs_peak]), "max_tolerance_fraction":float(scaled[worst]),
            "peak_index":[int(i) for i in worst], "absolute_peak_index":[int(i) for i in abs_peak],
            "passes":bool(scaled[worst] <= 1)}


def combine(first, current, details):
    if first is None:
        return {**current,"max_absolute_location":details(current["absolute_peak_index"]),"max_tolerance_location":details(current["peak_index"])}
    if current["max_absolute"] > first["max_absolute"]:
        first.update(max_absolute=current["max_absolute"], max_absolute_location=details(current["absolute_peak_index"]))
    if current["max_tolerance_fraction"] > first["max_tolerance_fraction"]:
        first.update(max_tolerance_fraction=current["max_tolerance_fraction"],
                     max_tolerance_location=details(current["peak_index"]), peak_index=current["peak_index"])
    first["passes"] &= current["passes"]
    return first


def paired_evidence(out, frozen, executions):
    limits = frozen["tolerances"]
    left,right = [out/(name+"-evidence.zip") for name in SCHEDULES]
    a,b = archive_arrays(left),archive_arrays(right)
    assert a[0] == b[0] and a[1]["draw"] == b[1]["draw"] and a[3] == b[3]
    draw = a[1]["draw"]
    seed = load(SOURCE/"artifacts/p2_g1_freeze.json")["training_seed"]
    root = f"runs/{seed}/"
    from io import BytesIO
    ind = {name:i for i,name in enumerate(a[3]["train_ids"]+a[3]["pool_ids"])}
    fields, envelope = {}, {}
    rows = {arm:{} for arm in ARMS}
    with ZipFile(left) as za, ZipFile(right) as zb:
        assert za.read("raw/truth.npy") == zb.read("raw/truth.npy")
        truth = np.load(BytesIO(za.read("raw/truth.npy")),allow_pickle=False)
        for ca,cb in zip(a[1]["field_chunks"],b[1]["field_chunks"],strict=True):
            ids = ca["order"]
            assert ids == cb["order"]
            y = truth[[ind[x] for x in ids]]
            def location(index):
                i,t,x = index
                return {"id":ids[i],"macro_step":t+1,"cell":x}
            for arm in ARMS:
                ja,wa = field_payload(za,root,ca,arm)
                se,we = field_payload(zb,root,cb,arm)
                if arm == "B01":
                    fields["forecast"] = combine(fields.get("forecast"),metric(ja,se,limits["forecast"]),location)
                envelope[arm] = combine(envelope.get(arm),metric(wa,we,limits["envelope"]),location)
                for index, unit in enumerate(ids):
                    items = []
                    for f,w in ((ja,wa),(se,we)):
                        err = np.abs(y[index]-f[index])
                        score = float((err/w[index]).max())
                        coeff = np.linalg.norm(w[index],axis=-1)/np.linalg.norm(y[index],axis=-1)
                        items.append((score,coeff,err,w[index]))
                    rows[arm][unit] = items
    analyzed = {}
    for arm, units in rows.items():
        ids = draw["calibration"]+draw["test"]
        scores = np.array([[units[i][j][0] for i in ids] for j in (0,1)])
        n = len(draw["calibration"])
        k = math.ceil((n+1)*.9)
        q = np.sort(scores[:,:n], axis=1)[:,k-1]
        signed_ranks = [run["evaluation"]["metrics"][0]["arms"][arm][0]["rank"] for run in executions]
        assert signed_ranks == [k,k]
        scored_counts = [run["evaluation"]["g1"]["coverage"][0]["covered"][arm] for run in executions]
        widths, end, included, locked = [],[],[],[]
        prior = frozen["previous_q"][arm]["q"]
        for j,v in enumerate(q):
            members = [units[i][j] for i in draw["test"]]
            widths.append(np.array([2*v*x[1].mean() for x in members]))
            end.append(np.array([2*v*x[1][-1] for x in members]))
            included.append([bool((x[2]<=v*x[3]).all()) for x in members])
            locked.append([bool((x[2]<=prior*x[3]).all()) for x in members])
            assert sum(included[j]) == scored_counts[j]
            np.testing.assert_allclose(widths[j].mean(), executions[j]["evaluation"]["g1"]["mean_relative_width"][arm],
                                       atol=5e-13, rtol=5e-13)
        delta = metric(*scores,limits["score_and_q"])
        flip_ids = [i for i,x,y in zip(draw["test"],*included,strict=True) if x != y]
        frozen_flips = [i for i,x,y in zip(draw["test"],*locked,strict=True) if x != y]
        width = metric(*widths,limits["width"])
        last = metric(*end,limits["width"])
        analyzed[arm] = {"score":delta, "worst_score_id":ids[delta["peak_index"][0]],
                         "q":q.tolist(), "q_parity":metric(q[:1],q[1:],limits["score_and_q"]),
                         "mean_width": [float(w.mean()) for w in widths],
                         "width":width, "worst_width_id":draw["test"][width["peak_index"][0]],
                         "width_at_T":last, "worst_width_at_T_id":draw["test"][last["peak_index"][0]],
                         "covered":scored_counts, "inclusion_differences":flip_ids,
                         "inclusion_differences_with_locked_prior_q":frozen_flips}
    parity = fields["forecast"]["passes"] and all(x["passes"] for x in envelope.values())
    parity &= all(all(arm[key]["passes"] for key in ("score","q_parity","width","width_at_T")) and
                  len(arm["inclusion_differences"]) <= limits["test_inclusion_flips"] and
                  len(arm["inclusion_differences_with_locked_prior_q"]) <= limits["test_inclusion_flips"]
                  for arm in analyzed.values())
    return {"shared_split":a[0], "forecast":fields["forecast"], "envelopes":envelope, "arms":analyzed,
            "passes":bool(parity)}, (a[2],b[2])


def check_timings(records, runs, frozen):
    sessions = []
    for name, signed, ran in zip(SCHEDULES, records, runs, strict=True):
        spec = frozen["timing"]
        assert signed["schedule"] == name and signed["batch"] == spec["batch"]
        assert signed["threads"] == spec["threads"] and signed["warmups"] == spec["warmups"]
        trips = signed["triplets"]
        assert len(trips) == spec["replicates_per_session"]
        active = [s for s in ran["host"] if trips[0]["started_ns"] <= s["timestamp_ns"] <= trips[-1]["ended_ns"]]
        assert active, "no observed host metadata during the measured timing block"
        planned = ["one","joint","serial"] if name == "joint" else ["serial","joint","one"]
        for i,r in enumerate(trips):
            assert r["order"] == planned[::1 if i%2 == 0 else -1]
            assert r["started_ns"] < r["ended_ns"] and all(math.isfinite(r[k]) and r[k]>0 for k in ("one",*SCHEDULES))
        assert signed["pairs"] == [{"one":r["one"],"W3":r[name]} for r in trips]
        ratios = {key:[r[key]/r["one"] for r in trips] for key in SCHEDULES}
        stats = {key:{"min":min(rows),"median":median(rows),"max":max(rows),"all":rows} for key,rows in ratios.items()}
        sessions.append({"session":name,"triplets":trips,"ratios":stats,"observed_during_benchmark":active,
                         "passes_serial_cost":stats["serial"]["median"] <= spec["gate"]})
    return sessions


def run(args):
    out = args.out
    frozen = load(out/"freeze.json")
    frozen_config(frozen,args.factory)
    key = (out/"draw.key").read_bytes()
    assert sha256(key).hexdigest() == frozen["draw_key_commitment"]
    env = dict(os.environ,PYTHONPATH=str(SOURCE/"src"),JAX_PLATFORMS="cpu",JAX_ENABLE_X64="true")
    command = [str(args.solver_python),"-c","import json,sys;from importlib.metadata import version;"
               "print(json.dumps({n:version(n) for n in sys.argv[1:]}))",*frozen["solver_versions"]]
    assert json.loads(subprocess.check_output(command,env=env)) == frozen["solver_versions"]
    data = out/"fresh"
    subprocess.run([str(args.solver_python),str(SOURCE/"scripts/p2_g1_solver.py"),str(out/"freeze.json"),str(data)],
                   env=env,check=True)
    source = out/"native-input"
    identities = separations(*build_inventory(data,source))
    prepared = [register(args.factory,args.kernel,out,frozen,s,source,key) for s in SCHEDULES]
    # Both plans and their common private split were sealed before either schedule ran.
    executions = [confirm(args.factory,p,frozen,out,s) for s,p in zip(SCHEDULES,prepared,strict=True)]
    parity, records = paired_evidence(out,frozen,executions)
    timing = check_timings(records,executions,frozen)
    cost = all(x["passes_serial_cost"] for x in timing)
    summary = {"scope":frozen["scope"],"freeze_sha256":file_digest(out/"freeze.json"),
               "identity_checks":identities,"host":{"system":platform.platform(),"logical_cpus":os.cpu_count()},
               "solver_timings":load(data/"solver-timings.json"),
               "executions":dict(zip(SCHEDULES,executions,strict=True)),
               "parity":parity,"timing":timing,"decision":{"parity":parity["passes"],"cost":cost,
                                                          "advance_serial":bool(cost and parity["passes"])}}
    dump(out/"qualification.json",summary)
    print(summary["decision"],flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="action",required=True)
    for mode in ("freeze","run"):
        s = subs.add_parser(mode)
        for opt in ("factory","kernel","out"):
            s.add_argument("--"+opt,type=Path,required=True)
        if mode == "run":
            s.add_argument("--solver-python",type=Path,required=True)
    args = p.parse_args()
    {"freeze":freeze,"run":run}[args.action](args)
