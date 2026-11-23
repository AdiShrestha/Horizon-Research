# P2 engineering analogue of T-data-2

This audit uses the exact signed 7.2.3 engineering FNO state, its W3 rule, both fitted W4 profiles, and each arm's rank-18 calibration threshold at T=100 and α=.10. The freeze script runs the installed K2 verifier against the prior evidence archive with the original public keys, checks the six worker source digests and fitted state digests, and reprocesses the committed calibration reductions to recover each threshold. It reproduces the saved forward probe and both weight fields exactly on one original 12-trajectory chunk per draw. The freeze file locks the seed, all 50 IDs, six solver cases, comparisons, runtimes, weight code and audit code before the solver streams a trajectory. No model or weight fitting takes place in this audit.

The new cohort uses engineering-domain seed 20261126. It has 50 unique RNG keys and float32 initial-state/viscosity fingerprints. The driver compares those fingerprints against all 85 signed prospective trajectories, including both calibration draws, and against all 24 design trajectories whose generator artifacts match their recorded hashes. Every solve within a batch uses the same initial-value problems and physical time grid. It checks exact parameters and keys, and checks that projected initial fields agree to 2 × 10⁻¹⁴ absolute. All fields are projected to 256 grid cells by Fourier truncation with the projected Nyquist removed, using the CPU float64 Exponax solver. The audit uses steps 1–100 at macro-step 0.005 and excludes the initial step.

## Solver and error comparisons

The approved primary engineering comparison holds time discretization at 20 substeps and compares 1024 with 2048 cells. The brief's original 2×/4× grid comparison is reported separately at 20 substeps for both 512 and 1024 cells. The former 512/10 → 1024/20 comparison mixes spatial and temporal changes, so it has a separate row. Time controls compare 10 with 20 substeps at 512 cells and 20 with 80 at 1024 and 2048 cells. The 1024/80 → 2048/80 case checks spatial refinement under the finer time discretization. The grid spacing of reported cells is 1/256 and cell indices are zero-based.

For discrepancy Dᵢ(t,x) and frozen positive weight wᵢ(t,x), the per-trajectory statistic is max(t,x) Dᵢ/wᵢ. The conservative statistic is max(t,x) [maxᵢ Dᵢ / minᵢ wᵢ]. Both extrema inside the ratio refer to the same time and cell, and they may come from different trajectories. The report gives both statistics divided by each draw's frozen q and names the contributing IDs and locations. The conservative statistic dominates the matched-trajectory maximum and determines the strict ε/q < 0.001 decision. W3 and W4 consume the same projected initial conditions and viscosities across every comparison.

The consumer reads at most 12 trajectories per frame, maintains pointwise extremal arrays plus scalar per-trajectory results, and writes no solver fields. Timings distinguish JAX compilation, synchronized solves, frozen W3/W4 evaluation, and total elapsed time including streaming and reduction. The reducer RSS refers to the consumer process.

## Reproduce

Use the Factory 7.2.3 CPU audit lock for the driver and the Horizon G0 lock for the solver. Use the installed development 7.2.3 kernel in a disposable trusted home. The p2-w34-7.2.3 artifacts are in the adjacent Factory dist directory. Generate the original 24-row design cohort with the published settings and attest its files with the original design generation report. Choose a fresh output directory.

```sh
HORIZON_SOURCE="$HORIZON_ROOT/source"
D="$FACTORY_ROOT/dist/p2-w34-7.2.3"
ENG="$FACTORY_ROOT/factory/engineering"
CONFIG="$HORIZON_SOURCE/configs/task/p2_burgers1d_pilot.json"
export PYTHONPATH="$HORIZON_SOURCE/src:$ENG"
"$SOLVER_PYTHON" "$ENG/generate_p2_audit.py" --config "$CONFIG" \
  --out "$WORK/design" --seed 20261124 --counts 0 24 0 0
"$FACTORY_PYTHON" "$HORIZON_SOURCE/scripts/p2_tdata2.py" freeze \
  --bundle "$D-field-audit.zip" --kernel "$K2_KERNEL" \
  --maintainer "$D-maintainer.pub" --custodian "$D-custodian.pub" \
  --design "$WORK/design" --design-provenance "$D-design-generation.json" \
  --out "$WORK/freeze.json"
"$FACTORY_PYTHON" "$HORIZON_SOURCE/scripts/p2_tdata2.py" run \
  --bundle "$D-field-audit.zip" --freeze "$WORK/freeze.json" \
  --solver-python "$SOLVER_PYTHON" --out "$WORK/report.json"
cd "$HORIZON_SOURCE"
.venv/bin/python -m pytest -q
```

The hand oracle gives matched maxima 2 and 4 from two trajectories. Its conservative maximum at step 1, cell 0 combines discrepancy 4 from trajectory 17 and weight 0.25 from trajectory 22, yielding 16 and ε/q = 8 at q = 2. The tests also check streaming partition invariance, rejection of invalid weights or q, and failure at exactly ε/q = 0.001.

## Observed decisions

**The approved 1024/20 candidate failed the conservative criterion for W4 on both draws and for W3 on draw 0.** Every matched-trajectory maximum for this comparison was below 0.001. The pointwise cross-trajectory calculation exposed larger ratios where a different trajectory had a smaller weight at the largest-error location or nearby.

| Arm and draw | Frozen q | Maximum matched ε/q, 1024→2048 | Conservative ε/q, 1024→2048 | Decision | Conservative ε/q, 512→1024 |
|---|---:|---:|---:|---|---:|
| B01, 0 | 2.425533 | 0.000248571 | 0.000248571 | Pass | 0.00973055 |
| W3, 0 | 1.443511 | 0.000173641 | 0.00191188 | Fail | 0.110863 |
| W4, 0 | 2.573546 | 0.000260316 | 0.00255183 | Fail | 0.104412 |
| B01, 1 | 2.580992 | 0.000233599 | 0.000233599 | Pass | 0.00914445 |
| W3, 1 | 3.292819 | 0.0000761212 | 0.000838134 | Pass | 0.0486004 |
| W4, 1 | 2.569364 | 0.000238916 | 0.00234205 | Fail | 0.0952340 |

All six arm/draw combinations failed the original 512→1024 comparison at matched substeps and the older coupled 512/10→1024/20 comparison. The raw spatial maxima were 0.02360176 at 512→1024 (trajectory 26, step 16, cell 64) and 0.000602916 at 1024→2048 (trajectory 26, step 18, cell 63). The matched W3 and W4 maxima at 1024→2048 were on trajectory 4, at step 17 and step 16 respectively, both at cell 65. The conservative W3 maximum paired discrepancy ID 26 with minimum-weight ID 27 (step 41, cell 69). The conservative W4 maximum paired discrepancy ID 26 with minimum-weight ID 8 (step 16, cell 62) on both draws. B01 weights tie everywhere at one, so the first ID is recorded for the minimum-weight tie.

All time-only controls passed for all arms. At 1024 cells the 20→80 raw difference peaked at 9.15334 × 10⁻⁶ (trajectory 7, step 8, cell 99), and the highest time-only conservative ε/q at either 1024 or 2048 cells was 4.39 × 10⁻⁵. Matching both spatial grids at 80 substeps reproduced the same pass/fail pattern. These controls indicate that spatial resolution is the principal contributor to the observed failure at the approved candidate.

| Solver cells / substeps | Compile (s) | Solve (s) |
|---|---:|---:|
| 512 / 10 | 0.889 | 0.545 |
| 512 / 20 | 0.874 | 1.056 |
| 1024 / 20 | 0.847 | 2.126 |
| 2048 / 20 | 0.846 | 4.430 |
| 1024 / 80 | 0.808 | 8.816 |
| 2048 / 80 | 0.794 | 17.537 |

The first run took 41.62 seconds end to end, including 6.54 seconds for weight evaluation. The reducer process peaked at 366.28 MiB RSS. The largest difference in matched projected initial states across cases was 1.89 × 10⁻¹⁵. There was zero overlap between the audit PRNG keys and the regenerated key sets for the three earlier engineering seeds, including the original 65-row pilot.

Keep T-data-2 and scientific submission open. The approved engineering candidate would need another spatial refinement check on an independently seeded cohort before supporting the weighted criterion, for example a properly fixed 2048→4096 comparison with a matched temporal control. Scientific T-data-2 requires the official solver, horizon, frozen scientific models and weight settings, registered calibration thresholds, and audit cohorts for each task. A finite two-grid comparison gives empirical evidence of refinement and cannot establish a uniform bound on error relative to the exact continuum solution. A continuum coverage statement needs a separately established continuum error bound under its stated assumptions.

The complete freeze, the full per-trajectory report, and the verification summary are in the source artifacts directory as p2_tdata2_freeze.json, p2_tdata2_report.json, and p2_tdata2_verification.json. A second run reproduced every threshold, discrepancy ratio, trajectory ID, and reported worst location exactly. The Horizon suite passed 102 tests, including the hand oracle, with Ruff and reducer mypy checks passing.
