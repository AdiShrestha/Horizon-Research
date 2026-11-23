"""Run the frozen §5 P2 engineering candidate with a disposable installed Factory."""
import argparse
import json
from pathlib import Path
import shutil
import time

import run_p2_cpu_pilot as pilot
from sf7.canon import file_digest
from sf7.fsio import read_json, write_json

SOURCE = Path(__file__).resolve().parents[1]
WORKER = SOURCE / "scripts/p2_w34_worker.py"


def plan(settings, commitment):
    spec = pilot.pilot_plan()
    spec["study_id"] = "p2-w34-engineering"
    model = spec["models"]["fno"]
    model["state"]["values"] += 1
    model["config"]["settings_sha256"] = commitment
    spec["draws"].update(fit=24, test=10, fit_draws=2, calibration_draws=1)
    spec["arms"].update({
        "W3": {"model": "fno", "kind": "intrinsic",
               "worker": {"entrypoint": model["entrypoint"], "config": dict(settings, arm="W3")}},
        "W4": {"model": "fno", "kind": "fitted",
               "worker": {"entrypoint": model["entrypoint"], "config": dict(settings, arm="W4")}},
    })
    spec["comparisons"].extend([
        {"candidate": "W3", "baseline": "B01", "role": "ablation"},
        {"candidate": "W4", "baseline": "W3", "role": "ablation"},
    ])
    spec["stream"]["chunk_trajectories"] = 12
    spec["field_audit"]["max_bytes"] = 48 * 1024**2
    return spec


def setup_workers(ws, frozen):
    root = ws / "source"
    for name, expected in frozen["code_sha256"].items():
        if name == "p2_cpu_worker.py":
            origin, relative = pilot.WORKER, name
        elif name.startswith("scripts/"):
            origin, relative = SOURCE / name, "worker.py"
        else:
            origin, relative = SOURCE / name, name.removeprefix("src/")
        assert file_digest(origin) == expected, origin
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, dest)


def receipts(vault, spec):
    run = vault / "trajectory/runs/725"
    telemetry = {}
    for path in sorted(run.rglob("telemetry.json")):
        details = read_json(path)
        evidence = read_json(path.parent / "evidence.json")
        stage = details["stage"]
        reads = evidence["confinement"]["read"]
        denied = evidence["confinement"]["probed_denied_reads"]
        assert any(p.endswith("/raw/truth.npy") for p in denied)
        assert len([p for p in reads if p.endswith("/truth.npy")]) == int(stage in ("train", "fit_weights"))
        if stage == "weights":
            arm = path.parent.name
            assert arm in ("W3", "W4")
            fit_reads = [p for p in reads if "/fit/" in p]
            assert len(fit_reads) == int(arm == "W4")
            if fit_reads:
                index = path.relative_to(run).parts[1]
                assert fit_reads == [str((run / "fit" / index / "W4/state.json").resolve())]
            stage += "_" + arm
        assert evidence["confinement"]["policy"] == "deny_default" and evidence["exit_code"] == 0
        assert "trajcp/rollout/sensitivity.py" in evidence["code"]
        telemetry.setdefault(stage, []).append(dict(details, kernel_wall_seconds=evidence["seconds"]))
    assert {k: len(v) for k, v in telemetry.items()} == dict(train=1, load_check=1, fit_weights=2,
                                                            forecast=6, weights_W3=6, weights_W4=6)
    profiles = [read_json(run / "fit" / str(i) / "W4/state.json") for i in range(spec["draws"]["fit_draws"])]
    return telemetry, profiles


def qualify(cohort, freeze_file, out):
    frozen = read_json(freeze_file)
    assert read_json(cohort / "trajectories.json")["provenance"]["seed"] == frozen["prospective_pool_seed"]
    assert freeze_file.stat().st_mtime_ns < (cohort / "truth.npy").stat().st_mtime_ns
    commitment = file_digest(freeze_file)
    spec = plan(frozen["settings"], commitment)
    out.mkdir(parents=True, exist_ok=False)
    home, ws = out / "owner", out / "study"
    keyfile, release = out / "maintainer.key", out / "release"
    cli = pilot.cli
    maintain = cli(home, pilot.ROOT, "release", "keygen", "--key", keyfile)["public_key"]
    cli(home, pilot.ROOT, "release", "build", "--key", keyfile, "--out", release)
    cli(home, pilot.ROOT, "trust", "add", maintain)
    installed = Path(cli(home, pilot.ROOT, "kernel", "install", release)["path"])
    kernel = cli(home, installed, "kernel", "verify")
    assert kernel["level"] == "K2"
    cli(home, installed, "owner", "init")
    study = cli(home, installed, "open", ws, "--study-id", spec["study_id"], "--kind", "trajectory-band")
    vault = home / "vaults" / spec["study_id"]
    shutil.copytree(cohort, vault / "inbox")
    write_json(ws / "project/plan.json", spec)
    write_json(ws / "project/sources.json", {"schema_version": 7, "artifacts": [
        {"name": path.name, "sha256": file_digest(path), "bytes": path.stat().st_size, "license": "CC0-1.0",
         "digest_source": "fresh deterministic engineering cohort from frozen configuration"}
        for path in sorted(cohort.iterdir())]})
    setup_workers(ws, frozen)
    assert cli(home, installed, "advance", ws)["state"] == "TRAJECTORY_SOURCE_ATTESTATION"
    cli(home, installed, "owner", "attest-sources", ws)
    assert cli(home, installed, "advance", ws)["state"] == "TRAJECTORY_OWNER_APPROVAL"
    cli(home, installed, "owner", "countersign", ws)
    assert not (ws / "evaluation.json").exists()
    started = time.monotonic()
    completed = cli(home, installed, "advance", ws)
    elapsed = time.monotonic() - started
    assert completed["state"] == "TRAJECTORY_COMPLETE", completed
    metrics = read_json(ws / "evaluation.json")["metrics"]
    raw = read_json(vault / "trajectory/scored.json")
    telemetry, profiles = receipts(vault, spec)
    archive = Path(cli(home, installed, "bundle", ws, "--include-targets")["path"])
    clean = out / "independent"
    clean.mkdir()
    custodian = study["custodian_key"]
    verified = cli(clean / "owner", installed, "verify-bundle", archive,
                   "--maintainer-key", maintain, "--custodian-key", custodian, cwd=clean)
    assert verified["status"] == "VERIFIED" and verified["rescored_targets"] and verified["paired_draws"] == 2
    report = {"scope": "fresh separately seeded P2 §5 engineering pilot", "kernel": kernel, "settings": frozen["settings"],
              "settings_sha256": commitment, "counts": spec["draws"], "wall_seconds": elapsed,
              "metrics": metrics, "fit_only_profiles": profiles, "telemetry": telemetry, "verifier": verified,
              "thresholds": [{name: rows[0]["threshold"] for name, rows in d["arms"].items()} for d in raw],
              "retained_field_bytes": sum(d["field_audit"]["retained_field_bytes"] for d in raw),
              "archive_bytes": archive.stat().st_size, "archive_sha256": file_digest(archive)}
    write_json(out / "p2-w34-qualification.json", report)
    write_json(out / "p2-w34-clean-replay.json", verified)
    shutil.copyfile(archive, out / "p2-w34-field-audit.zip")
    (out / "maintainer.pub").write_text(maintain + "\n")
    (out / "custodian.pub").write_text(custodian + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "freeze", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.cohort.resolve(), args.freeze.resolve(), args.out.resolve())
    print(json.dumps({k: v for k, v in result.items() if k not in ("fit_only_profiles", "telemetry")}, indent=2))
