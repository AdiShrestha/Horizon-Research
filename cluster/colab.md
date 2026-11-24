# Running the A18 package on Google Colab (L4 GPU)

This guide runs cluster package 1 (`horizon-a18-e778095.tar`: the 104 jobs that train and evaluate every 1D emulator) on Google Colab with an NVIDIA L4. The package was built for a SLURM cluster, but its local runner (`cluster/jobs.py local`) needs no scheduler. Every job is resumable, so Colab disconnects cost time, not results.

Read sections 1 and 2 before you spend compute units. Section 5 measures the real speed, and you should decide whether Colab is worth it only after seeing that number.

---

## 1. Rules that still apply on Colab

- **One GPU type for the whole campaign.** Every production model must come from the same GPU model (A18). If you start on L4, finish on L4. Do not mix in A100 or T4 sessions or cluster GPUs. Check with `nvidia-smi` at the start of every session (section 4).
- **Do not edit code or data.** The verify job checks the hash of every code file and every array against the manifests in the bundle.
- **Never run two processes on the same job.** Section 7 shows the only safe way to run jobs in parallel.
- **The W3 cost job (`w3cost-p2`) must run with nothing else on the GPU.** It measures timing for G1(iii).
- **Keep everything the jobs write:** logs, failed runs and reports. Do not delete a failed run directory. `jobs.py` retries in place, and the trainer resumes from its checkpoint.

## 2. What you need

| Item | Requirement |
|---|---|
| Colab plan | A plan that offers the **L4** runtime (paid compute units). Check the units per hour Colab shows for L4 before starting. |
| Google Drive space | **At least 20 GB free.** The bundle is 2.9 GB. The work directory grows to about 12–15 GB: seed-0 error and W3 fields are about 11 GB, plus checkpoints, statistics and logs. The free 15 GB Drive tier is not enough. |
| Time | Measured in section 5. Expect very roughly 200–350 L4-hours for everything if run one job at a time. This is unmeasured; section 5 replaces it with real numbers. |
| On your laptop | The bundle files `horizon-a18-e778095.tar` and `horizon-a18-e778095.tar.sha256` from `Horizon Research/cluster_bundle/`. |

## 3. One-time setup: put the bundle on Drive

1. In Google Drive, create the folder `MyDrive/horizon/`.
2. Upload both bundle files into it. The upload is 2.9 GB, so let it finish completely:
   - `horizon-a18-e778095.tar`
   - `horizon-a18-e778095.tar.sha256`
3. Create a new Colab notebook, then choose **Runtime → Change runtime type → L4 GPU**.

Everything below is notebook cells. Lines starting with `!` are shell commands.

## 4. Session setup (run at the start of every session)

A Colab VM loses `/content` when the session ends. The bundle and the Python environment therefore live in `/content` and are rebuilt each session, which takes about 5 minutes. The **work directory** (checkpoints, results, job status) lives on Drive and persists.

**Cell 4.1: check the GPU and mount Drive.**

```python
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
from google.colab import drive
drive.mount('/content/drive')
```

The first line must say `NVIDIA L4`. If it doesn't, stop and change the runtime type. Do not run jobs on another GPU (rule 1).

**Cell 4.2: verify and extract the bundle to fast local disk.**

```python
%%bash
set -e
cd /content/drive/MyDrive/horizon
sha256sum -c horizon-a18-e778095.tar.sha256      # must print: horizon-a18-e778095.tar: OK
mkdir -p /content/hz
tar -xf horizon-a18-e778095.tar -C /content/hz
ls /content/hz/horizon-a18-e778095/source
```

The bundle must always be extracted to exactly `/content/hz/horizon-a18-e778095`. The job plan records the absolute data path, so the same path is needed in every session.

**Cell 4.3: build the Python 3.12 environment.** The bundle was built and tested on CPython 3.12 with pinned versions (torch 2.15.0, jax 0.12.0.1, numpy 2.5.0, …). It is installed into its own virtual environment with `uv`, so Colab's preinstalled packages are neither used nor disturbed.

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

The last lines must show `torch 2.15.0`, a CUDA build and `NVIDIA L4`.

The first session only: save the exact installed versions next to the work directory.

```python
!mkdir -p /content/drive/MyDrive/horizon && uv pip freeze --python /content/venv/bin/python > /content/drive/MyDrive/horizon/pip-freeze-colab.txt
```

**Cell 4.4: shortcuts used below.**

```python
import os
SRC  = "/content/hz/horizon-a18-e778095/source"
WORK = "/content/drive/MyDrive/horizon/work"
PY   = "/content/venv/bin/python"
os.chdir(SRC)
print(SRC, WORK, PY, sep="\n")
```

## 5. First session: checks and a timing run (about 30–45 minutes)

**5.1 Plan the job graph (first session only, never again).**

```python
!{PY} cluster/jobs.py plan --data-root data --work {WORK}
```

This writes `WORK/graph.json` with 104 jobs. Running `plan` again stops with "graph.json exists". That is deliberate: never use `--force` on a work directory that has results.

**5.2 Verify the bundle, then run the device check.**

```python
!{PY} cluster/jobs.py run verify   --work {WORK} --device cuda
!{PY} cluster/jobs.py run devcheck --work {WORK} --device cuda
```

- `verify` (about 1 minute) checks the six data arrays (SHA-256, shape, dtype, finiteness), 106 code-file hashes and 117 tests.
- `devcheck` (about 5–15 minutes) trains each task × tier for 200 steps on CPU and on the L4. The step-1 loss must agree within 10⁻⁵.

Both must print `done (exit 0 …)`. If either fails, **stop**: send `WORK/logs/verify.log` or `WORK/logs/devcheck.log` and `WORK/checks/*.json` to the Architect. All training depends on these two jobs.

**5.3 Timing run.** These are 10-epoch runs of E1, E2 and E3 on P2, written to a scratch folder that is not part of the campaign. Ten epochs keep E3's real mix: 10% one-step epochs, then unrolled epochs.

```python
%%bash -s "$PY"
cd /content/hz/horizon-a18-e778095/source
export PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
for TIER in E1 E2 E3; do
  $1 scripts/02_train.py --task p2 --tier $TIER --seed 0 --epochs 10 \
     --data data/p2 --out /content/timing/$TIER --device cuda | tail -1
done
```

**5.4 Project the campaign from the measured speed.**

```python
import json
s = {t: json.load(open(f"/content/timing/{t}/report.json"))["seconds_per_epoch"] for t in ("E1", "E2", "E3")}
print({t: round(v, 1) for t, v in s.items()}, "seconds per P2 epoch")

def task_hours(B, scale, wang):
    # E1 seed 0 at 500 and 1,000 (gate); E1 seeds 1-4 and E1-dropout (+ E1-Wang) at B; E2 and E3 seeds 0-4 at B
    e1_like = 1500 + (4 + 1 + (1 if wang else 0)) * B
    return scale * (s["E1"] * e1_like + s["E2"] * 5 * B + s["E3"] * 5 * B) / 3600

for B in (500, 1000):
    p2 = task_hours(B, 1.0, True)    # P2: 180,000 training pairs per epoch
    p1 = task_hours(B, 2.5, False)   # P1: 450,000 pairs per epoch, no E1-Wang
    p3 = task_hours(B, 2.5, True)    # P3: 450,000 pairs per epoch
    print(f"budget {B} on every task: P2 {p2:.0f} h, P1 {p1:.0f} h, P3 {p3:.0f} h, "
          f"total training {p2 + p1 + p3:.0f} L4-hours (+ Stage 1, measured after the first pass)")
```

The gate decides per task between 500 and 1,000 epochs, so the real total lies between the two printed lines. Stage 1 adds roughly 45 passes, each from a few minutes to about an hour; check the first one (section 8).

**Decide now.** Multiply the total by the compute units per hour Colab shows for L4. If the cost or the weeks of wall time are too much, use Colab only for P2 (it is the G1 task) and run P1 and P3 on the university cluster. A task must not be split between platforms, because of rule 1.

## 6. Running the campaign (one job at a time, simplest)

Run this in each session after section 4. It works through every pending job in dependency order: P2 first (budget runs, gate, the 14 remaining P2 models and their Stage 1), then P1, then P3, and finally the W3 cost on P2. Finished jobs are skipped, and an interrupted training resumes from its last checkpoint.

```python
!{PY} cluster/jobs.py local --work {WORK} --device cuda
```

The cell prints one line per job as it finishes, for example `train-p2-e1-s0-b500: done (exit 0, 5400s)`. Each job's full output is in `WORK/logs/<job>.log`.

**Stopping cleanly:** use **Runtime → Interrupt execution**. The runner forwards the interrupt to the trainer, which saves a checkpoint before exiting.

**After a disconnect or a session timeout:** start a new session, run section 4, and run the same cell again. At most the epochs since the last checkpoint (up to 10) are repeated. A job interrupted mid-run may be marked `failed`. That is harmless: the next `local` pass retries it, and the trainer resumes.

**A job that fails twice for the same reason is a real failure.** Stop and report it with its log (section 10).

## 7. Faster: two processes on one L4 (optional)

The 1D models are small, so one training job does not fill an L4. Two jobs at once usually raise throughput. Run them **only on disjoint tasks**, and **only after** `verify` and `devcheck` are done.

1. **Run P2 alone first.** Its last job is the W3 cost, which needs a quiet GPU.

   ```python
   !{PY} cluster/jobs.py local --work {WORK} --device cuda --match p2
   ```

   Repeat across sessions until `jobs.py status` shows no pending P2 jobs and `w3cost-p2` is done.

2. **Then run P1 in the background and P3 in the foreground:**

   ```python
   import subprocess
   bg = subprocess.Popen(f"{PY} cluster/jobs.py local --work {WORK} --device cuda --match p1 "
                         f"> {WORK}/local-p1.out 2>&1", shell=True)
   ```

   ```python
   !{PY} cluster/jobs.py local --work {WORK} --device cuda --match p3
   ```

   When the P3 cell ends, check whether P1 is still running with `print(bg.poll())`. `None` means it is still running. Its progress is in `WORK/local-p1.out`.

Never start two processes with the same `--match`, or one with no `--match` while another is running. They would run the same job twice and corrupt its checkpoint. Whether parallel runs actually help: compare `seconds_per_epoch` in two concurrent runs' `report.json` with the timing run.

## 8. Monitoring

In a separate cell, while nothing else is running in the foreground, or between runs:

```python
!{PY} cluster/jobs.py status --work {WORK}
!tail -n 3 {WORK}/logs/train-p2-e1-s0-b500.log      # any job's live log
!nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

Useful facts:
- Training logs print one line every 5 epochs, with the training loss, the validation loss and the elapsed seconds.
- Every finished run has `WORK/emulators/<task>/<run>/report.json`, giving the best epoch, the validation loss, the final-step error, the growth slope, the device and `seconds_per_epoch`.
- After the two budget runs of a task, `WORK/emulators/<task>/budget.json` shows the A07 decision (500 or 1,000 epochs) and why.
- Stage 1 timing is in `WORK/stats/<task>/<tier>_s<seed>/stats_meta.json` (`seconds`).

Check that every run used the L4 (rule 1):

```python
import glob, json
gpus = {json.load(open(f))["hardware"].get("gpu") for f in glob.glob(f"{WORK}/emulators/*/*/settings.json")}
print(gpus)   # must be {'NVIDIA L4'}
```

## 9. Collecting the results and bringing them back

When `jobs.py status` shows `done: 104`:

```python
!bash cluster/collect.sh {WORK} /content/drive/MyDrive/horizon
!ls -la /content/drive/MyDrive/horizon/results-*
```

This writes the following to `MyDrive/horizon/`:
- `results-core-<stamp>.tar`: emulators, statistics without fields, checks, logs, status and the graph;
- `results-fields-<stamp>.tar`: the seed-0 error and W3 fields, about 11 GB;
- `results-<stamp>.sha256`.

Each tar also holds a SHA-256 manifest of its own contents.

Download all three to the laptop, then ingest each tar. Ingest refuses to overwrite and checks every file's hash:

```bash
cd "/Users/adi/Desktop/Projects/Horizon Research/source"
.venv-run/bin/python scripts/12_ingest_cluster.py ~/Downloads/results-core-<stamp>.tar --dest artifacts/cluster/a18
.venv-run/bin/python scripts/12_ingest_cluster.py ~/Downloads/results-fields-<stamp>.tar --dest artifacts/cluster/a18
```

You can also collect early, for example after P2 alone, so G1 can start. Run `collect.sh` at any time. It packs whatever exists, and a later collection gets a new timestamp.

## 10. Stop and report to the Architect

Stop and send the named files instead of working around the problem when any of these happens:
- `verify` or `devcheck` fails: `WORK/logs/<job>.log`, `WORK/checks/*.json`;
- a training stops with `non-finite validation loss`: the run's `curves.json` and log;
- a job fails twice in a row with the same error: its log;
- `nvidia-smi` shows anything other than an L4 in a session where jobs already ran: the GPU check output from section 8;
- Drive runs out of space: `du -sh` of `WORK/*`.

Never fix a failure by editing code, deleting a run directory or re-planning the work directory.

## 11. Quick reference

| Task | Command |
|---|---|
| Session start | Cells 4.1–4.4 |
| Plan (once) | `{PY} cluster/jobs.py plan --data-root data --work {WORK}` |
| Run everything | `{PY} cluster/jobs.py local --work {WORK} --device cuda` |
| Run one task | `… local … --match p2` |
| One specific job | `{PY} cluster/jobs.py run <job-id> --work {WORK} --device cuda` |
| Status | `{PY} cluster/jobs.py status --work {WORK}` |
| Collect | `bash cluster/collect.sh {WORK} /content/drive/MyDrive/horizon` |

Job IDs look like `train-p1-e2-s3`, `train-p2-e1-s0-b1000`, `gate-p3`, `stage1-p2-e1-s0`, `train-p3-e1w-s0` and `w3cost-p2`. The full list is in `WORK/graph.json`.
