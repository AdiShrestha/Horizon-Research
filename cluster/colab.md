# Running the A18 package on Google Colab (A100 GPU)

This guide runs cluster package 1 (`horizon-a18-e778095.tar`) on Google Colab with an NVIDIA A100, under decision A20:

- **P2 runs on Colab A100.** It is the G1 task, and the cheapest of the three.
- **P1 and P3 wait** for a larger GPU allocation, such as KU GPU access once it is granted.

The package was built for SLURM, but its local runner (`cluster/jobs.py local`) needs no scheduler. Every job is resumable, so a Colab disconnect costs time, not results.

The binding limit is **compute units (CU)**: 200 a month on the owner's plan, about 150 left in November. Read sections 1, 2 and 5 before spending any. Section 5 measures the real speed and converts the work into CU.

---

## 1. Rules that still apply on Colab

- **One GPU model per task (A20).** Every P2 model, P2 Stage 1 pass and the P2 W3 cost must run on an A100. Check at the start of every session (section 4). A100 40 GB and 80 GB are the same chip, so either is fine, and each run records which one it used. If Colab gives you anything else (L4, T4, V100), do not run P2 jobs in that session.
- **P2 only.** Every run command in this guide uses `--match p2` or names a P2 job. Do not start P1 or P3 jobs on Colab (A20).
- **Do not edit code or data.** The verify job checks the hash of every code file and data array.
- **Never let two processes run the same job.** Concurrency (section 7) is allowed only on disjoint job sets.
- **`w3cost-p2` runs alone on the GPU.** It is the G1(iii) timing.
- **Keep everything the jobs write.** Never delete a failed run directory, because the trainer resumes in place.
- **Stop paying when idle.** A connected Colab runtime spends CU even when nothing runs. When a session's work is done, use **Runtime → Disconnect and delete runtime**.

## 2. What you need

| Item | Requirement |
|---|---|
| Colab | A plan with A100 access, such as Colab Pro (included with Google AI Pro). Note the **CU per hour** Colab shows for the A100 runtime (Resources panel), because section 5 needs it. |
| Google Drive | At least **10 GB free** for P2: 2.9 GB for the bundle and about 4 GB of P2 work (fields about 1.8 GB, plus checkpoints, statistics and logs). If P1 and P3 later run from Drive too, plan for about 20 GB. |
| On your laptop | `horizon-a18-e778095.tar` and `horizon-a18-e778095.tar.sha256` from `Horizon Research/cluster_bundle/`. |

## 3. One-time setup

1. In Google Drive, create the folder `MyDrive/horizon/`.
2. Upload both bundle files into it, and wait until the 2.9 GB upload has finished.
3. Create a Colab notebook, then choose **Runtime → Change runtime type → A100 GPU**. Tick High-RAM if it is offered.

Lines starting with `!` below are shell commands inside notebook cells.

## 4. Session setup (start of every session, about 5 minutes)

`/content` is wiped when a session ends, so the bundle and the Python environment are rebuilt each time. The work directory (checkpoints, results, job status) lives on Drive and persists.

**Cell 4.1: check the GPU and mount Drive.**

```python
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
from google.colab import drive
drive.mount('/content/drive')
```

The name must contain `A100`. If it doesn't, disconnect, change the runtime type and try again. Do not run jobs on another GPU.

**Cell 4.2: check and extract the bundle to fast local disk.**

```python
%%bash
set -e
cd /content/drive/MyDrive/horizon
sha256sum -c horizon-a18-e778095.tar.sha256      # must print: horizon-a18-e778095.tar: OK
mkdir -p /content/hz
tar -xf horizon-a18-e778095.tar -C /content/hz
ls /content/hz/horizon-a18-e778095/source
```

The bundle must always be extracted to exactly `/content/hz/horizon-a18-e778095`, because the job plan stores that absolute data path.

**Cell 4.3: build the pinned Python 3.12 environment.** The environment is separate from Colab's own packages.

```python
%%bash
set -e
pip -q install uv
uv venv /content/venv --python 3.12
uv pip install --python /content/venv/bin/python \
    -r /content/hz/horizon-a18-e778095/source/cluster/requirements-cluster.txt
/content/venv/bin/python - <<'PY'
import torch, jax, numpy, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "CUDA build", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))
print("jax", jax.__version__, "numpy", numpy.__version__)
PY
```

The output must show `torch 2.15.0`, a CUDA build and an A100.

In the first session only, record the installed versions:

```python
!uv pip freeze --python /content/venv/bin/python > /content/drive/MyDrive/horizon/pip-freeze-colab.txt
```

**Cell 4.4: shortcuts used below.**

```python
import os
SRC  = "/content/hz/horizon-a18-e778095/source"
WORK = "/content/drive/MyDrive/horizon/work"
PY   = "/content/venv/bin/python"
os.chdir(SRC)
```

## 5. First session: checks, timing and the CU budget (about 45 minutes)

**5.1 Plan the job graph (first session only).**

```python
!{PY} cluster/jobs.py plan --data-root data --work {WORK}
```

This writes all 104 jobs to `WORK/graph.json`. Only P2 jobs are run on Colab. Never re-plan with `--force`.

**5.2 Verify the bundle and run the device check.**

```python
!{PY} cluster/jobs.py run verify   --work {WORK} --device cuda
!{PY} cluster/jobs.py run devcheck --work {WORK} --device cuda
```

- `verify` checks the data, the 106 code hashes and 117 tests.
- `devcheck` compares CPU with the A100 for every task × tier. The step-1 loss must agree within 10⁻⁵. It also covers P1 and P3, which only reads their data and costs a few minutes.

Both must end with `done (exit 0 …)`. If either fails, stop and report (section 10).

**5.3 Timing: one job, then four at once.** These are 10-epoch P2 runs in a scratch folder, not campaign runs. Ten epochs reproduce E3's real one-step and unrolled mix. The second part measures how much running jobs concurrently gains on the A100.

```python
%%bash -s "$PY"
cd /content/hz/horizon-a18-e778095/source
export PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
T() { $1 scripts/02_train.py --task p2 --tier $2 --seed 0 --epochs 10 --data data/p2 --out /content/timing/$3 --device cuda | tail -1; }
for TIER in E1 E2 E3; do T $1 $TIER solo-$TIER; done          # one at a time
for K in 1 2 3 4; do T $1 E1 par-E1-$K & done; wait            # four E1 runs at once
```

**5.4 Convert to hours and compute units.** Set `CU_PER_HOUR` to the A100 rate Colab shows you.

```python
import json
CU_PER_HOUR = 0.0   # <- enter the A100 rate from Colab's Resources panel
r = lambda name: json.load(open(f"/content/timing/{name}/report.json"))["seconds_per_epoch"]
s = {t: r(f"solo-{t}") for t in ("E1", "E2", "E3")}
par = sum(r(f"par-E1-{k}") for k in range(1, 5)) / 4
gain = 4 * s["E1"] / par            # throughput of 4 concurrent jobs relative to 1
print({t: round(v, 1) for t, v in s.items()}, "s/epoch alone;", round(par, 1), "s/epoch with 4 at once;",
      f"concurrency gain x{gain:.2f}")

def hours(epochs_by_tier):
    return sum(s[t] * e for t, e in epochs_by_tier.items()) / 3600

g1 = hours({"E1": 1500})                                  # E1 seed 0 at 500 and 1,000 (gate)
for B in (500, 1000):
    rest = hours({"E1": 6 * B, "E2": 5 * B, "E3": 5 * B})  # E1 s1-4, E1-dropout, E1-Wang, E2 s0-4, E3 s0-4
    for label, h in (("G1 runs", g1), (f"rest of P2 at B={B}", rest)):
        print(f"{label}: {h:.1f} GPU-h alone, {h / gain:.1f} h wall with 4 at once, "
              f"~{CU_PER_HOUR * h / gain:.0f} CU")
```

How to read it:
- **G1 runs** is what P2 needs before 13 December: the two budget runs. Stage 1 and the W3 cost add a little; section 8 shows how to measure them.
- **Rest of P2** is the tier comparison and the ensemble. The gate decides between the two B lines.
- Spend this month's units on the G1 runs first. If the rest of P2 does not fit in the remaining units, it continues next month (A20 order).

**5.5 Before closing the session:** Runtime → Disconnect and delete runtime.

## 6. Running P2 in A20 order

Run section 4 at the start of each session. Then:

**Step A: the two budget runs, concurrently** (two processes, two different jobs):

```python
import subprocess
bg = subprocess.Popen(f"{PY} cluster/jobs.py run train-p2-e1-s0-b500 --work {WORK} --device cuda "
                      f"> {WORK}/bg-b500.out 2>&1", shell=True)
```

```python
!{PY} cluster/jobs.py run train-p2-e1-s0-b1000 --work {WORK} --device cuda
```

The 500-epoch run usually finishes first. Check it with `print(bg.poll())`, where `None` means still running and `0` means done.

**Step B: the gate, then Stage 1 for E1 seed 0, then the W3 cost alone:**

```python
!{PY} cluster/jobs.py run gate-p2         --work {WORK} --device cuda
!{PY} cluster/jobs.py run stage1-p2-e1-s0 --work {WORK} --device cuda
!{PY} cluster/jobs.py run w3cost-p2       --work {WORK} --device cuda
```

That completes the G1 material. Collect it (section 9) and bring it to the laptop. Stage 2 and the G1 evidence pack run on the laptop CPU.

**Step C: the rest of P2, as units allow.** E3 and E2 seed 0 come first. Then run four disjoint job sets at once:

```python
procs = {m: subprocess.Popen(f"{PY} cluster/jobs.py local --work {WORK} --device cuda --match {m} "
                             f"> {WORK}/bg-{m}.out 2>&1", shell=True)
         for m in ("p2-e3-", "p2-e2-", "p2-e1-", "p2-e1d")}
```

```python
{m: p.poll() for m, p in procs.items()}      # None = still running
```

```python
!{PY} cluster/jobs.py local --work {WORK} --device cuda --match p2-e1w     # E1-Wang, after the others finish
```

The trailing hyphen in `p2-e1-` keeps it from matching `p2-e1d` and `p2-e1w`. Each set holds one tier's trainings and Stage 1 passes, so no job is in two sets.

**Resuming:** after a disconnect, rerun section 4 and the same cell. Finished jobs are skipped, and interrupted trainings resume from their last checkpoint, losing at most 10 epochs. A job interrupted mid-run may show as `failed`; the next run retries it. To stop cleanly, use Runtime → Interrupt execution: the trainer saves a checkpoint first.

## 7. Concurrency notes

- At batch 64 a 0.3–0.7 M-parameter model leaves most of an A100 idle, so several processes usually raise total throughput. Section 5.3 measured the gain. If it is close to ×1, run one job at a time instead.
- GPU memory is not a constraint: each P2 job needs well under 4 GB.
- Never run two processes on the same job, and never run `w3cost-p2` alongside anything else.

## 8. Monitoring

```python
!{PY} cluster/jobs.py status --work {WORK}                 # counts all 104; only the P2 jobs matter here
!tail -n 3 {WORK}/logs/train-p2-e1-s0-b1000.log            # live log of any job
!nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

Where to look:
- **Training:** `WORK/emulators/p2/<run>/report.json` gives the best epoch, losses, final-step error, growth slope and `seconds_per_epoch`.
- **Budget decision:** `WORK/emulators/p2/budget.json` gives the A07 choice (500 or 1,000 epochs).
- **Stage 1 time:** `WORK/stats/p2/<tier>_s<seed>/stats_meta.json` (`seconds`).
- **W3 cost:** `WORK/checks/w3_cost_p2.json`, with `median_ratio` and `pass`.

Check that every P2 run used an A100:

```python
import glob, json
print({json.load(open(f))["hardware"].get("gpu") for f in glob.glob(f"{WORK}/emulators/p2/*/settings.json")})
```

## 9. Collecting results and bringing them back

```python
!bash cluster/collect.sh {WORK} /content/drive/MyDrive/horizon
!ls -la /content/drive/MyDrive/horizon/results-*
```

This writes `results-core-<stamp>.tar`, `results-fields-<stamp>.tar` and `results-<stamp>.sha256` to `MyDrive/horizon/`. You can collect at any time: after Step B for G1, and again later.

Download them, then on the laptop:

```bash
cd "/Users/adi/Desktop/Projects/Horizon Research/source"
.venv-run/bin/python scripts/12_ingest_cluster.py ~/Downloads/results-core-<stamp>.tar --dest artifacts/cluster/a18-colab-<stamp>
.venv-run/bin/python scripts/12_ingest_cluster.py ~/Downloads/results-fields-<stamp>.tar --dest artifacts/cluster/a18-colab-<stamp>
```

Ingest refuses to overwrite and checks every hash. Give each collection its own destination.

## 10. Stop and report to the Architect

Stop and report in any of these cases:
- `verify` or `devcheck` fails: send `WORK/logs/<job>.log` and `WORK/checks/*.json`;
- a training stops with `non-finite validation loss`: send the run's `curves.json` and log;
- a job fails twice with the same error: send its log;
- the GPU check shows a non-A100 GPU in a run's `settings.json`;
- compute units run out before the G1 material (Step B) is complete.

Never fix a failure by editing code, deleting a run directory or re-planning.

## 11. Quick reference

| Task | Command |
|---|---|
| Session start | Cells 4.1–4.4 |
| One job | `{PY} cluster/jobs.py run <job-id> --work {WORK} --device cuda` |
| A job set | `{PY} cluster/jobs.py local --work {WORK} --device cuda --match <substring>` |
| Status | `{PY} cluster/jobs.py status --work {WORK}` |
| Collect | `bash cluster/collect.sh {WORK} /content/drive/MyDrive/horizon` |
| End of session | Runtime → Disconnect and delete runtime |

P2 job IDs: `train-p2-e1-s0-b500`, `train-p2-e1-s0-b1000`, `gate-p2`, `stage1-p2-e1-s0`, `w3cost-p2`, `train-p2-e<1|2|3>-s<0-4>`, `stage1-p2-e<1|2|3>-s<0-4>`, `train-p2-e1d-s0`, `train-p2-e1w-s0`.
