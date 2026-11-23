# P2 2048-cell bounded engineering refinement

The previous 1024/20 audit remains a failure for the weighted criterion. This qualification freezes 2048/20 as the candidate and 4096/20 as its comparator. It also freezes 20→80 time controls at both grids and a same-time comparison at 80 substeps. The 1024/20 solve is included only to measure the cost baseline on the same trajectories and to supply the same projected initial field for W3/W4 as in the prior audit. No grid beyond 4096 is used.

The two cohorts are the 50 prior audit trajectories (paired, seed 20261126) and 50 fresh trajectories (independent engineering qualification, seed 20261127). The paired cohort uses the same key sequence as the previous audit. The driver verifies zero key and projected-input fingerprint overlap between the two cohorts. Both have zero key overlap with the earlier pilot, design and frozen calibration data. The input-fingerprint check separately compares against all 109 prior design and prospective inputs from the signed engineering workflow. The binding includes the hashes of the previous failed audit, the signed 7.2.3 evidence archive, the verified ledger, worker code, source state, W3 settings, and both frozen W4 profiles and thresholds. The freeze replays the prior bundle in the installed development kernel, checks exact fresh-worker probe equivalence against the committed reload probe and reproduces the W3/W4 weights exactly on one earlier chunk per fit draw. There is no refitting or recalibration.

Both cohorts use the same float64 CPU Burgers solver and physical horizon as the previous audit (100 steps with macro-step 0.005). Every grid is Fourier-projected to 256 cells. An independent identity pass checks RNG keys and projected initial-state/viscosity fingerprints without looking at refinement discrepancies. The field-streaming pass verifies exact matching parameters and keys across solver settings and checks the projected initial fields agree to an absolute 2 × 10⁻¹⁴. The solver sends batches of at most 12 trajectories and writes no field arrays to disk. The two weight computations use identical frozen settings and the same 256-cell input for all comparisons.

The per-trajectory maximum is max(t,x) |Δᵢ(t,x)|/wᵢ(t,x). The conservative maximum takes maxᵢ |Δᵢ| and minᵢ wᵢ pointwise before division and maximization over (t,x). Each is divided by the relevant frozen q. The reported conservative decision uses the strict threshold ε/q < 0.001. Steps are numbered from one and cells from zero. Solver and consumer peak RSS are separate process high water marks. Compilation is timed separately for each of the two batch shapes, and solve-only cost ratios use a 1024/20 solve of the same cohort in the same run.

## Reproduce

Use the installed development Factory 7.2.3 kernel in a disposable trusted home, the pinned Factory worker environment for the driver, and the Horizon G0 environment for the solver. The p2-w34-7.2.3 artifacts are in the adjacent Factory dist directory. The current source artifacts directory supplies the preserved previous audit records.

```sh
HORIZON_SOURCE="$HORIZON_ROOT/source"
D="$FACTORY_ROOT/dist/p2-w34-7.2.3"
ENG="$FACTORY_ROOT/factory/engineering"
CONFIG="$HORIZON_SOURCE/configs/task/p2_burgers1d_pilot.json"
export PYTHONPATH="$HORIZON_SOURCE/src:$ENG"
"$SOLVER_PYTHON" "$ENG/generate_p2_audit.py" --config "$CONFIG" \
  --out "$WORK/design" --seed 20261124 --counts 0 24 0 0
"$FACTORY_PYTHON" "$HORIZON_SOURCE/scripts/p2_refine_2048.py" freeze \
  --bundle "$D-field-audit.zip" --kernel "$K2_KERNEL" \
  --maintainer "$D-maintainer.pub" --custodian "$D-custodian.pub" \
  --design "$WORK/design" --design-provenance "$D-design-generation.json" \
  --out "$WORK/freeze.json"
"$FACTORY_PYTHON" "$HORIZON_SOURCE/scripts/p2_refine_2048.py" run \
  --bundle "$D-field-audit.zip" --freeze "$WORK/freeze.json" \
  --solver-python "$SOLVER_PYTHON" --out "$WORK/report.json"
"$FACTORY_PYTHON" "$HORIZON_SOURCE/scripts/p2_refine_2048.py" run \
  --bundle "$D-field-audit.zip" --freeze "$WORK/freeze.json" \
  --solver-python "$SOLVER_PYTHON" --out "$WORK/repeat.json"
cd "$HORIZON_SOURCE"
.venv/bin/python -m pytest -q
```

The existing hand-calculated oracle and boundary test cover the conservative reducer. The independent-cohort result is an engineering decision for the frozen T=100 pilot. Official T-data-2 requires the registered scientific horizon, models, weights, thresholds and task cohorts. A two-grid empirical comparison supplies no uniform continuum error guarantee.

## Observed results

**2048/20 passed the bounded engineering comparison in both cohorts.** All six arm/draw combinations met the strict conservative decision on the fresh cohort. The paired cohort also passed. The values below are ε/q for the spatial comparison at 20 substeps. Each cell gives the matched maximum followed by the conservative maximum.

| Arm and fit draw | Frozen q | Paired matched / conservative | Fresh matched / conservative |
|---|---:|---:|---:|
| B01, 0 | 2.425533 | 3.39974e-6 / 3.39974e-6 | 4.22124e-8 / 4.22124e-8 |
| W3, 0 | 1.443511 | 2.17063e-6 / 2.30923e-5 | 8.09555e-8 / 3.76508e-7 |
| W4, 0 | 2.573546 | 3.19731e-6 / 3.40145e-5 | 1.24278e-7 / 5.59285e-7 |
| B01, 1 | 2.580992 | 3.19496e-6 / 3.19496e-6 | 3.96699e-8 / 3.96699e-8 |
| W3, 1 | 3.292819 | 9.51566e-7 / 1.01232e-5 | 3.54894e-8 / 1.65054e-7 |
| W4, 1 | 2.569364 | 2.94278e-6 / 3.13067e-5 | 1.14061e-7 / 5.13308e-7 |

All paired matched maxima were on trajectory 26 at step 17, cell 63. At that location the paired weighted conservative maxima combined discrepancy ID 26 with minimum-weight ID 9. All fresh matched maxima were on ID 28 at step 16, cell 248. The fresh W4 conservative maximum combined discrepancy ID 28 with weight ID 10 at that same position, while W3 reached its conservative maximum one step and one cell later with the same IDs. B01's constant minimum weight is tied across trajectories.

All time controls passed. The largest conservative ε/q on the paired cohort was 4.39e-5 for either 20→80 control and 3.40e-5 for the spatial comparison at 80 substeps. The corresponding maxima on the fresh cohort were 3.86e-5 for either time control and 5.60e-7 for spatial refinement at 80 substeps. Temporal refinement therefore leaves the spatial decisions unchanged. The separately preserved 1024/20 failure remains the evidence for that older candidate.

| Solver cells / substeps | Paired compile / solve (s) | Fresh compile / solve (s) | Solve cost vs 1024/20, paired / fresh |
|---|---:|---:|---:|
| 1024 / 20 | 0.924 / 2.299 | 0.890 / 2.177 | 1.00× / 1.00× |
| 2048 / 20 | 0.965 / 4.583 | 0.917 / 4.598 | 1.99× / 2.11× |
| 4096 / 20 | 0.894 / 7.674 | 0.885 / 7.424 | 3.34× / 3.41× |
| 2048 / 80 | 0.888 / 17.805 | 0.810 / 17.709 | 7.74× / 8.13× |
| 4096 / 80 | 0.804 / 29.260 | 0.816 / 29.580 | 12.73× / 13.59× |

The paired and fresh cohorts took 67.54 s and 67.25 s, respectively, including 6.60 s and 6.80 s of weight evaluation. The solver RSS peaked at 439.55 MiB for the paired cohort and 426.48 MiB for the fresh cohort. The long-lived consumer peaked at 376.28 MiB. The first run took 137.36 s overall, including the identity check. Both processes streamed the same 12-trajectory cap. No solver arrays were persisted.

## Verification and evidence

A full repeat took 136.38 s and reproduced all 2,400 per-trajectory result rows, thresholds, aggregate ratios, worst IDs and locations exactly. A separate check recomputed each reported ratio from its saved discrepancy, weight and q, checked the aggregate maxima and the strict pass/fail decisions, and confirmed that all 50 paired metadata records and the 2048 temporal control match the previous audit exactly. The 102-test Horizon suite and Ruff checks passed.

The source artifacts directory contains p2_refine_2048_freeze.json, p2_refine_2048_report.json and p2_refine_2048_verification.json. The freeze SHA-256 is 8a6a0726e35d3824a21b0702ca0030592e2f5f6f6a28f710e176634d3a5c3bef. The previous audit artifacts and the Horizon registration and preregistration files retain their original hashes.
