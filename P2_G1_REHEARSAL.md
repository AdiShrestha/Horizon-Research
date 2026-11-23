# P2 G1 engineering rehearsal at T=200

This is a synthetic engineering rehearsal of the three §14 G1 criteria. The frozen horizon is 200 macro-steps at Δt=.005, so the last evaluated time is 1.0. The CPU float64 Exponax Burgers solver uses 2048 cells and 20 substeps. The separately seeded 50-trajectory numerical audit compares against 4096/20 at matched parameters, physical times and 256-cell Fourier projection. It reports both steps 1–200 and steps 101–200. A step offset of 100 in the JSON report means to add 100 to the reported window-relative step. The earlier failed 1024/20 weighted audit and the T=100 2048/20 pass remain separate artifacts.

## Frozen design and isolation

The design has one 32-trajectory training set and two disjoint panels of 24 fit, 29 calibration and 30 test trajectories each. Together with the audit cohort this is 248 independently keyed synthetic initial-value problems. Domain seeds 20261128–20261135 were frozen with all budgets, the FNO training seed, probe seed, profile definitions, evaluation, and timing protocol. The solver verifies common initial states across both numerical grids. The driver checks exact key and canonical float32 initial-state/viscosity disjointness between current roles and checks RNG keys against the 100 previously audited P2 inputs.

The CPU float32 periodic FNO uses the previous engineering architecture with three spectral blocks, width 12 and 12 modes. It trains for 120 teacher-forced Adam updates on training labels only. This rehearsal uses one training seed and does not reproduce the full scientific E1 protocol. Training writes a data-only float32 NPY state. The reload process checks probe equivalence exactly. Finite differences use eight seeded modes-1–8 unit-RMS fields at 0.001 times initial RMS, with smoothing of three periodic cells, a per-step relative floor of 0.02 and a training-only absolute floor of 10⁻⁶ × training field SD. Forecast scoring shares the unperturbed member of the batched W3 rollout across all four arms. Inference pads inputs to a fixed batch size of 12 so the final smaller chunks use the same model shape.

- **B01** uses flat unit weights.
- **W1** is W1-emp, the isotonic median of per-step L∞ errors, with a training-SD and first-step-relative floor. Its β=.50 was fixed before running.
- **W3** uses the intrinsic sensitivity field with the frozen smoothing and floors and has no fitted state.
- **W4** multiplies W3 by the time profile selected on D_fit from the q, reciprocal-LP and parametric candidates. Two inner D_fit folds of 12 yield a finite pseudo-calibration rank of 12 at α=.10. The selected family is refit on all 24 fit rows.

Blind processes get copied input-only directories and the common checkpoint. The fit processes receive only their own fit scope, and the calibration routine commits thresholds before the test evaluator reads any test labels. This study-side harness uses explicit file views and separate processes under a disposable workflow. It does not expand Factory 7.2.3's T=100 kernel contract or claim its OS confinement for this rehearsal. The SHA-256 manifest binds the plan, generated role files, model state, profiles, blind fields, and calibration JSON. CPU float64 field reductions are checked by the separate Factory scalar field oracle after the manifest seals. A new process verifies the same manifest and recomputes the published measures from the retained synthetic fields.

## Frozen decision rules

The full-field SUP score uses one calibration score per trajectory. At α=.10 with n=29, k=27. Under independent and continuous calibration/test scores conditional on the trained and fitted procedure, the count in 30 test trajectories is Beta-Binomial(30,27,3). Its central 95% marginal quantiles are 22–30, verified independently with exact rational probabilities. The discrete interval contains about 97.59% of the nominal law. The engineering criterion checks all eight arm/draw cells against that interval. Their decisions are jointly dependent through shared data within a draw and shared model across draws. Distinct draws use disjoint cohorts. The report also gives descriptive Bonferroni-wide reference quantiles of 19–30 for eight comparisons. Two panels can check this tolerance but offer little power to assess the full distribution; tied calibration scores are reported.

Width is the mean of the per-trajectory, per-step ratios of the full band L2 norm to truth L2 norm, averaged over all 200 steps and then averaged equally over the two draws. The frozen width criterion is the best W1/W3/W4 reduction against B01 being at least 20%. Timing uses the median of seven paired W3/one-rollout ratios after two warm-ups, alternating the operation order. Both run on the same CPU, Torch precision, 12-input batch, model, and 200 steps. W3 timing includes the eight perturbations alongside the unperturbed member, finite-difference RMS, smoothing and floor. The threshold is at most 10×.

## Reproduce

The Factory CPU lock supplies Torch 2.15.0, NumPy 2.5.0, SciPy 1.18.0 and jaxtyping. The separate Horizon G0 lock supplies Exponax 0.2.1, JAX and JAXLIB 0.12.0.1, NumPy 2.5.0 and SciPy 1.18.1. Source and both prior audit artifacts remain part of the freeze verification. Choose fresh output paths.

```sh
H="$HORIZON_ROOT/source"
export PYTHONPATH="$H/src:$FACTORY_ROOT/factory:$FACTORY_ROOT/factory/engineering"
"$FACTORY_PYTHON" "$H/scripts/p2_g1.py" freeze --out "$WORK/freeze.json"
"$FACTORY_PYTHON" "$H/scripts/p2_g1.py" run --freeze "$WORK/freeze.json" \
  --solver-python "$SOLVER_PYTHON" --evidence "$WORK/evidence" --report "$WORK/report.json"
"$FACTORY_PYTHON" "$H/scripts/p2_g1.py" verify --evidence "$WORK/evidence" \
  --report "$WORK/report.json" --out "$WORK/replay.json"
cd "$H"
.venv/bin/python -m pytest -q
```

## Observed G1 decisions

**Coverage passed the frozen finite-test tolerance. Width improvement and W3 runtime failed.** The 20% width-reduction target would require a mean relative width of at most 9.713 versus the observed B01 mean of 12.142. W3 was the narrowest weighted arm at 16.595, or 36.68% wider than B01. W1 averaged 23.317 (92.04% wider) and W4 averaged 17.908 (47.49% wider).

| Draw | Arm | Frozen q | Covered / 30 | Mean relative width (steps 1–200) |
|---|---|---:|---:|---:|
| 0 | B01 | 2.299224 | 27 | 12.593 |
| 0 | W1-emp | 4.472152 | 26 | 24.465 |
| 0 | W3 | 3.277776 | 25 | 16.170 |
| 0 | W4 | 4.693759 | 25 | 17.727 |
| 1 | B01 | 2.269262 | 29 | 11.690 |
| 1 | W1-emp | 5.235889 | 28 | 22.168 |
| 1 | W3 | 3.654641 | 30 | 17.020 |
| 1 | W4 | 3.537902 | 28 | 18.089 |

The calibration scores had no ties in these samples. The W4 q profile at β=.80 won selection on draw 0, and the parametric profile won on draw 1. All counts fall within 22–30. This wide acceptance interval and the shared model/within-panel pairing limit the interpretation to a descriptive engineering check.

The median standalone rollout took 0.1419 s and the median W3 computation took 2.7376 s. The median of the seven measured paired ratios was **19.37×**, with a range of 18.65×–20.70×. These are eager CPU float32 timings for the complete W3 operation with batch size 12. No MPS, JVP, or accelerator cost qualification is inferred.

## Numerical audit and replay

All eight arm/draw combinations passed the strict 0.001 conservative ε/q threshold on the 50 audit trajectories, both across steps 1–200 and in the additional steps 101–200. The maximum conservative ratio across the full horizon was **7.194 × 10⁻⁷** (draw 1 W4, discrepancy ID 32, weight ID 2, step 56, cell 115). For steps 101–200 the maximum was **6.772 × 10⁻¹⁰** (draw 0 W3, discrepancy ID 32, weight ID 15, step 193, cell 124). The report also records each arm/draw matched maximum and all per-trajectory values. These are empirical matched-grid diagnostics at the frozen engineering settings. Official solver qualification and a continuum error bound remain open.

The workflow took 175.15 s through retained-field audit. Solver work totaled 59.26 s excluding 6.96 s of compilation. For the audit cohort alone, the 2048 and 4096 solves took 8.69 s and 14.88 s. The maximum worker RSS was 547.56 MiB in W4/W1 fitting. The evaluator peaked at 314.94 MiB. The evidence manifest binds 260,683,058 bytes (248.61 MiB) of generated inputs, fields, checkpoints and metadata, stored in the ignored source data directory. This is a bounded fixture retention policy. The separate scalar oracle replayed 472 calibration/test trajectory–arm pairs and agreed to at most 2.13 × 10⁻¹⁴ in relative width.

The width and cost failures remain in the engineering evidence. Read them before planning changes to floors, envelopes or W3 execution. Any follow-up settings need a prospective engineering freeze and fresh evaluation. Official G0, G1, the full model/seed protocol, accelerator cost and scientific submission remain open.

The replay from a clean working directory against the retained workspace evidence matched the original replay exactly and rejected a changed published threshold. The Horizon suite passed 102 tests and Ruff passed on the rehearsal scripts. The SHA-256 freeze is fa4ec041ac662612933dffe00207bb0701fc33b6de61f17a625877e5c3af0087. The published files are p2_g1_freeze.json, p2_g1_report.json and p2_g1_verification.json in source/artifacts. The retained manifest and synthetic field files are under source/data/p2_g1_rehearsal.

For exact replay of this historical freeze, use the Factory 7.2.3 development distribution referenced by the original evidence. Factory 7.2.4 extends the independently audited kernel module whose full 7.2.3 file digest is included in this freeze.
