"""Job graph and runner for the A18 production package (1D tasks P2, P1, P3).

    python cluster/jobs.py plan   --data-root DATA --work WORK [--smoke]
    python cluster/jobs.py status --work WORK
    python cluster/jobs.py run    JOB --work WORK          # one job (what each SLURM job executes)
    python cluster/jobs.py local  --work WORK [--match S]  # every pending job, in dependency order
    python cluster/jobs.py submit --work WORK --config cluster/cluster.json [--dry-run]

The graph (per task, in the order p2, p1, p3):
  verify -> devcheck -> train E1 seed 0 at B = 500 and 2B = 1,000 epochs (in parallel)
  -> gate (A07 writes <work>/emulators/<task>/budget.json)
  -> train E1 seeds 1-4, E2 seeds 0-4, E3 seeds 0-4 at the gated budget (in parallel)
  -> Stage 1 on the DEV pool for every trained (tier, seed); seed 0 of each tier stores fields (A08),
     P2 E1 seed 0 also writes the W3 ablation variants (A09)
  -> on P2 only: the G1(iii) W3 cost on the selected E1 seed 0 (exclusive node).
Every job is idempotent: a finished job (status/<job>.done) is skipped; training resumes from its
last checkpoint. ``--smoke`` builds the same graph with tiny budgets in a separate work directory to
exercise the whole pipeline in minutes. Nothing here touches the CONFIRM stream.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ("p2", "p1", "p3")
TIERS = ("E1", "E2", "E3")
SEEDS = (0, 1, 2, 3, 4)
BASE_EPOCHS = 500
GPU_KINDS = {"devcheck", "train", "stage1", "w3cost"}


@dataclass
class Job:
    id: str
    kind: str
    deps: list[str]
    params: dict[str, object] = field(default_factory=dict)


def graph(smoke: bool) -> dict[str, Job]:
    jobs: dict[str, Job] = {}

    def add(job: Job) -> None:
        jobs[job.id] = job

    base = 1 if smoke else BASE_EPOCHS
    add(Job("verify", "verify", []))
    add(Job("devcheck", "devcheck", ["verify"]))
    for task in TASKS:
        gate_runs = []
        for epochs in (base, 2 * base):
            jid = f"train-{task}-e1-s0-b{epochs}"
            add(Job(jid, "train", ["devcheck"], {"task": task, "tier": "E1", "seed": 0,
                                                "epochs": epochs, "name": f"e1_seed0_b{epochs}"}))
            gate_runs.append(jid)
        add(Job(f"gate-{task}", "gate", gate_runs, {"task": task, "base": base}))
        for tier in TIERS:
            for seed in SEEDS:
                if tier == "E1" and seed == 0:
                    trained = f"gate-{task}"
                else:
                    trained = f"train-{task}-{tier.lower()}-s{seed}"
                    add(Job(trained, "train", [f"gate-{task}"], {"task": task, "tier": tier,
                                                                 "seed": seed, "name": f"{tier.lower()}_seed{seed}"}))
                add(Job(f"stage1-{task}-{tier.lower()}-s{seed}", "stage1", [trained],
                        {"task": task, "tier": tier, "seed": seed, "fields": seed == 0,
                         "variants": task == "p2" and tier == "E1" and seed == 0}))
    add(Job("w3cost-p2", "w3cost", ["stage1-p2-e1-s0"], {"task": "p2"}))
    return jobs


class Work:
    def __init__(self, work: Path) -> None:
        self.root = work.resolve()
        self.state = json.loads((self.root / "graph.json").read_text())
        self.jobs = {j["id"]: Job(**j) for j in self.state["jobs"]}
        self.data = Path(self.state["data_root"])
        self.smoke = bool(self.state["smoke"])

    def status(self, jid: str) -> str:
        if (self.root / "status" / f"{jid}.done").exists():
            return "done"
        if (self.root / "status" / f"{jid}.failed").exists():
            return "failed"
        return "pending"

    def emulator_dir(self, task: str) -> Path:
        return self.root / "emulators" / task

    def run_dir(self, task: str, tier: str, seed: int) -> Path:
        if tier == "E1" and seed == 0:
            budget = json.loads((self.emulator_dir(task) / "budget.json").read_text())
            return self.emulator_dir(task) / budget["selected_run"]
        return self.emulator_dir(task) / f"{tier.lower()}_seed{seed}"


def python() -> list[str]:
    return [sys.executable]


def command(work: Work, job: Job, device: str) -> list[str]:
    p = job.params
    if job.kind == "verify":
        return python() + [str(ROOT / "cluster" / "verify_bundle.py"), "--data-root", str(work.data),
                           "--out", str(work.root / "checks" / "verify.json")]
    if job.kind == "devcheck":
        return python() + [str(ROOT / "scripts" / "10_device_check.py"), "--device", device,
                           "--data-root", str(work.data), "--out", str(work.root / "checks" / "device_check.json")]
    if job.kind == "train":
        task = str(p["task"])
        cmd = python() + [str(ROOT / "scripts" / "02_train.py"), "--task", task, "--tier", str(p["tier"]),
                          "--seed", str(p["seed"]), "--data", str(work.data / task),
                          "--out", str(work.emulator_dir(task) / str(p["name"])), "--device", device]
        if "epochs" in p:
            cmd += ["--epochs", str(p["epochs"])]
        else:
            cmd += ["--budget-file", str(work.emulator_dir(task) / "budget.json")]
        if work.smoke:
            cmd += ["--max-steps", "3"]
        return cmd
    if job.kind == "gate":
        cmd = python() + [str(ROOT / "scripts" / "11_a07_gate.py"), "--root", str(work.root / "emulators"),
                          "--task", str(p["task"]), "--base", str(p["base"])]
        return cmd + (["--allow-smoke"] if work.smoke else [])
    if job.kind == "stage1":
        task, tier, seed = str(p["task"]), str(p["tier"]), int(p["seed"])
        cmd = python() + [str(ROOT / "scripts" / "03_rollout_reduce.py"),
                          "--run", str(work.run_dir(task, tier, seed)), "--data", str(work.data / task),
                          "--split", "dev", "--out", str(work.root / "stats" / task / f"{tier.lower()}_s{seed}"),
                          "--device", device, "--batch", "64"]
        if p.get("fields"):
            cmd.append("--fields")
        if p.get("variants"):
            cmd.append("--variants")
        if work.smoke:
            cmd += ["--limit", "8"]
        return cmd
    if job.kind == "w3cost":
        task = str(p["task"])
        return python() + [str(ROOT / "scripts" / "14_w3_cost.py"), "--run", str(work.run_dir(task, "E1", 0)),
                           "--data", str(work.data / task), "--device", device,
                           "--out", str(work.root / "checks" / f"w3_cost_{task}.json")]
    raise ValueError(job.kind)


def run_one(work: Work, jid: str, device: str) -> int:
    job = work.jobs[jid]
    if work.status(jid) == "done":
        print(f"{jid}: already done")
        return 0
    for dep in job.deps:
        if work.status(dep) != "done":
            print(f"{jid}: dependency {dep} is {work.status(dep)}")
            return 3
    status_dir, log_dir = work.root / "status", work.root / "logs"
    status_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / f"{jid}.failed").unlink(missing_ok=True)
    cmd = command(work, job, device)
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(ROOT / "src"), "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1",
                "PYTHONUNBUFFERED": "1"})
    start = time.time()
    with open(log_dir / f"{jid}.log", "a") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {shlex.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)

        def forward(signum: int, frame: object) -> None:
            # SLURM's pre-timeout signal reaches this process; the trainer checkpoints on SIGTERM
            proc.send_signal(signal.SIGTERM)

        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        code = proc.wait()
    record = {"job": asdict(job), "command": cmd, "exit": code, "seconds": time.time() - start,
              "host": os.uname().nodename if hasattr(os, "uname") else None,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    name = "done" if code == 0 else "failed"
    (status_dir / f"{jid}.{name}").write_text(json.dumps(record, indent=2))
    print(f"{jid}: {name} (exit {code}, {record['seconds']:.0f}s)")
    return code


def topo(jobs: dict[str, Job]) -> list[str]:
    order, seen = [], set()

    def visit(j: str) -> None:
        if j in seen:
            return
        seen.add(j)
        for d in jobs[j].deps:
            visit(d)
        order.append(j)

    for j in jobs:
        visit(j)
    return order


def cmd_plan(args: argparse.Namespace) -> None:
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if (work / "graph.json").exists() and not args.force:
        raise SystemExit(f"{work}/graph.json exists (use --force to rewrite the plan)")
    jobs = graph(args.smoke)
    state = {"data_root": str(args.data_root.resolve()), "smoke": args.smoke,
             "created": time.strftime("%Y-%m-%d %H:%M:%S"), "jobs": [asdict(j) for j in jobs.values()]}
    (work / "graph.json").write_text(json.dumps(state, indent=2))
    kinds: dict[str, int] = {}
    for j in jobs.values():
        kinds[j.kind] = kinds.get(j.kind, 0) + 1
    print(json.dumps({"work": str(work), "jobs": len(jobs), "by_kind": kinds}, indent=2))


def cmd_status(args: argparse.Namespace) -> None:
    work = Work(args.work)
    counts: dict[str, list[str]] = {"done": [], "failed": [], "pending": []}
    for jid in topo(work.jobs):
        counts[work.status(jid)].append(jid)
    for k, v in counts.items():
        print(f"{k}: {len(v)}")
    for jid in counts["failed"]:
        print(f"  failed: {jid} (log: {work.root / 'logs' / (jid + '.log')})")


def cmd_local(args: argparse.Namespace) -> None:
    work = Work(args.work)
    failed: set[str] = set()
    for jid in topo(work.jobs):
        if args.match and args.match not in jid:
            continue
        if any(d in failed or work.status(d) != "done" for d in work.jobs[jid].deps):
            failed.add(jid)
            print(f"{jid}: skipped (dependency not done)")
            continue
        if run_one(work, jid, args.device) != 0:
            failed.add(jid)
    print(f"local run finished; {len(failed)} job(s) failed or skipped")


SBATCH = """#!/bin/bash
{directives}
set -uo pipefail
{modules}
cd {root}
{python} {root}/cluster/jobs.py run {jid} --work {work} --device cuda &
pid=$!
on_timeout() {{
  echo "time limit near: checkpointing {jid}"
  kill -TERM "$pid"; wait "$pid"
  {requeue}
}}
trap on_timeout USR1
wait "$pid"; code=$?
if [ "$code" -gt 128 ]; then wait "$pid"; code=$?; fi
exit "$code"
"""


def cmd_submit(args: argparse.Namespace) -> None:
    work = Work(args.work)
    cfg = json.loads(args.config.read_text())
    ids_path = work.root / "slurm_ids.json"
    ids = json.loads(ids_path.read_text()) if ids_path.exists() else {}
    script_dir = work.root / "sbatch"
    script_dir.mkdir(parents=True, exist_ok=True)
    (work.root / "logs").mkdir(parents=True, exist_ok=True)
    py = str((ROOT / cfg["python"]).resolve()) if not os.path.isabs(cfg["python"]) else cfg["python"]
    for jid in topo(work.jobs):
        job = work.jobs[jid]
        if work.status(jid) == "done":
            continue
        if jid in ids and not args.resubmit:
            continue
        d = [f"#SBATCH --job-name=hz-{jid}", f"#SBATCH --output={work.root}/logs/{jid}.slurm.out",
             "#SBATCH --open-mode=append", f"#SBATCH --time={cfg['time'][job.kind]}",
             f"#SBATCH --cpus-per-task={cfg['cpus_per_task']}", f"#SBATCH --mem={cfg['mem']}",
             "#SBATCH --signal=B:USR1@300"]
        gpu = job.kind in GPU_KINDS
        partition = cfg.get("partition") if gpu or not cfg.get("cpu_partition") else cfg["cpu_partition"]
        if partition:
            d.append(f"#SBATCH --partition={partition}")
        for key in ("account", "qos"):
            if cfg.get(key):
                d.append(f"#SBATCH --{key}={cfg[key]}")
        if gpu and cfg.get("constraint"):
            d.append(f"#SBATCH --constraint={cfg['constraint']}")
        if gpu and cfg.get("gres"):
            d.append(f"#SBATCH --gres={cfg['gres']}")
        if job.kind in cfg.get("exclusive_for", []):
            d.append("#SBATCH --exclusive")
        if cfg.get("requeue_on_timeout"):
            d.append("#SBATCH --requeue")
        deps = [str(ids[x]) for x in job.deps if x in ids and work.status(x) != "done"]
        script = SBATCH.format(
            directives="\n".join(d),
            modules="\n".join(f"module load {m}" for m in cfg.get("modules", [])),
            root=shlex.quote(str(ROOT)), python=shlex.quote(py), jid=jid, work=shlex.quote(str(work.root)),
            requeue='scontrol requeue "$SLURM_JOB_ID"; exit 0' if cfg.get("requeue_on_timeout") else "exit 1",
        )
        path = script_dir / f"{jid}.sh"
        path.write_text(script)
        sb = ["sbatch", "--parsable"]
        if deps:
            sb.append("--dependency=afterok:" + ":".join(deps))
        sb.append(str(path))
        if args.dry_run:
            print(shlex.join(sb))
            ids[jid] = f"DRY-{jid}"
            continue
        out = subprocess.run(sb, capture_output=True, text=True, check=True).stdout.strip()
        ids[jid] = out.split(";")[0]
        print(f"{jid}: {ids[jid]}")
        ids_path.write_text(json.dumps(ids, indent=2))
    if args.dry_run:
        print(f"dry run: {len(ids)} jobs would be submitted; scripts in {script_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("status")
    p.add_argument("--work", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("job")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("local")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--match", default=None)
    p = sub.add_parser("submit")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resubmit", action="store_true")
    args = parser.parse_args()
    if args.cmd == "plan":
        cmd_plan(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "run":
        sys.exit(run_one(Work(args.work), args.job, args.device))
    elif args.cmd == "local":
        cmd_local(args)
    else:
        cmd_submit(args)


if __name__ == "__main__":
    main()
