# Horizon Research — cluster runbook (A18, 1D production package)

This package trains every 1D emulator the study needs and computes their Stage 1 statistics on a
GPU cluster. It contains the code at one commit, the TRAIN and DEV data for P2, P1 and P3 with
their hashes, and a job graph that runs unattended. Nothing in it touches the confirmation stream.

## What runs

| Step | Jobs | GPU | Notes |
|---|---|---|---|
| verify | 1 | no | data hashes, shapes, finiteness, code hashes, core tests |
| devcheck | 1 | yes | CPU vs GPU float32, every task × tier; step-1 loss within 1e-5 (A06/A18) |
| A07 budget runs | 6 | yes | E1 seed 0 at 500 and 1,000 epochs, per task, in parallel |
| gate | 3 | no | A07 15% rule → `emulators/<task>/budget.json` |
| training | 42 | yes | E1 seeds 1–4, E2 and E3 seeds 0–4, per task, at the gated budget |
| auxiliary (B10, B15) | 5 | yes | E1-dropout on P1–P3, E1-Wang on P2 and P3, seed 0, gated budget |
| Stage 1 | 45 | yes | DEV pool (1,500 trajectories): W0 and default W3 statistics; fields for seed 0 of every tier; W3 ablation variants for P2 E1 seed 0 |
| W3 cost | 1 | yes, exclusive | G1(iii) on P2 E1 seed 0: median of 7 paired W3 / rollout ratios |

104 jobs. Each is idempotent and resumable: training checkpoints every 10 epochs and on SIGTERM;
a finished job is never rerun. All runs use float32 with TF32 disabled, deterministic algorithms and
no autocast or compile; every run records the GPU, driver, CUDA, cuDNN and node.

## Steps

1. **Copy and check the bundle.** `sha256sum -c horizon-a18-<commit>.tar.sha256`, then
   `tar -xf horizon-a18-<commit>.tar` on the cluster's project or scratch file system (≈ 3.2 GB;
   the work directory grows to about 25 GB with fields).
2. **Build the environment** on a login node with internet access:
   `cd horizon-a18-<commit>/source && bash cluster/setup_env.sh` (Python 3.12; load the cluster's
   Python module first if needed). It writes `cluster/pip-freeze-cluster.txt`; keep it.
3. **Configure.** `cp cluster/cluster.example.json cluster/cluster.json` and set the partition,
   account, GPU request (`gres`), and, if possible, a `constraint` so every job lands on the same GPU
   model. Use one GPU type for the whole campaign.
4. **Plan.** `.venv-cluster/bin/python cluster/jobs.py plan --data-root data --work ../work`
5. **Rehearse (optional, ≈ 15 min).** A smoke graph with tiny budgets, in its own work directory:
   `.venv-cluster/bin/python cluster/jobs.py plan --data-root data --work ../work-smoke --smoke`
   then `cluster/jobs.py submit --work ../work-smoke --config cluster/cluster.json`.
   Discard `work-smoke` afterwards; its numbers mean nothing.
6. **Submit.** `.venv-cluster/bin/python cluster/jobs.py submit --work ../work --config cluster/cluster.json`
   (add `--dry-run` first to see every `sbatch` line). Dependencies are passed with
   `--dependency=afterok`, so a failed check stops everything downstream of it.
7. **Watch.** `cluster/jobs.py status --work ../work` and `squeue -u $USER`. Logs: `../work/logs/`.
   - A job near its time limit receives SIGUSR1, checkpoints and requeues itself (same job ID, so
     dependencies survive). If the cluster forbids requeue, set `requeue_on_timeout` to false and
     resubmit unfinished jobs with `jobs.py submit --resubmit`.
   - Without SLURM, run everything on one GPU node with `cluster/jobs.py local --work ../work`.
8. **Collect.** `bash cluster/collect.sh ../work` writes `results-core-<stamp>.tar` and
   `results-fields-<stamp>.tar` with SHA-256 manifests inside. Copy both back.
9. **Ingest on the laptop.**
   `.venv-run/bin/python scripts/12_ingest_cluster.py results-core-<stamp>.tar --dest artifacts/cluster/a18`
   (and the same for the fields tar). Ingest refuses to overwrite and checks every hash.

## Stop conditions (report to the Architect; do not work around)

- `verify` or `devcheck` fails;
- any training job ends with a non-finite validation loss;
- a gate cannot read both budget runs;
- Stage 1 reports non-finite rollouts above what the job's `stats_meta.json` explains;
- anything would require the CONFIRM stream or data other than the bundle.

## Files

- `cluster/jobs.py`: the graph, local runner and SLURM submission.
- `cluster/verify_bundle.py`: the first job.
- `scripts/02_train.py`, `scripts/10_device_check.py`, `scripts/11_a07_gate.py`,
  `scripts/03_rollout_reduce.py`, `scripts/14_w3_cost.py`: what the jobs run.
- `src/trajcp/emulators/tiers.py`: E1/E2/E3 definitions; `device.py`: float32 and determinism.
- `data/DATA_MANIFEST.json`, `CODE_VERSION.json`: the hashes the verify job checks.
