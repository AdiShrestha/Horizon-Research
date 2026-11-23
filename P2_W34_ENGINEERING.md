# P2 §5 engineering pilot

The CPU float32 model runs eight independently sampled Gaussian Fourier perturbations per initial condition. It omits the DC mode to preserve the mean and uses modes 1–8, unit grid RMS, and amplitude `1e-3 * rms(u0)`. The perturbation key combines a fixed seed with the canonical float32 initial field and viscosity. The circulation of pseudonyms, execution order, and split memberships therefore do not change the perturbations. The worker rolls out the unperturbed state alongside all eight perturbations. It accumulates finite-difference RMS in float64, applies periodic spatial Gaussian smoothing, then a per-trajectory/per-step median floor and a separately stored `1e-6 * training_field_std` absolute floor. Training standard deviation uses all post-initial training reference cells and steps, with one float32 scalar in the signed data-only model state.

For each of twelve smoothing/floor settings, a 24-row **design-only** cohort compared q, LP and parametric time profiles in seeded, content-order-invariant twofold selection. Each 12-row pseudo-calibration fold uses conformal rank 12 at α=.10. For q, the grid is β ∈ {.50, .80, .90, .95, .99}, with isotonic regression of the empirical 1−β quantile. The LP uses the existing HiGHS sparse score-coefficient solver and its exact reciprocal as the band profile. The parametric candidate fits median step statistics by least squares on the log scale with a nonnegative growth exponent, bounded fit parameters, and the brief’s time-envelope floor. These are engineering choices for the parametric fit criterion and bounds. Selection minimizes row-weighted mean full relative width over the pseudo-calibration folds, then refits the selected candidate on all D_fit rows. The paired pilot freezes smoothing and the relative floor before generating the fresh pool. Each protected D_fit split can select its own profile family and β by the locked rule. Workers outside D_fit receive no labels. A W3 worker receives no fitted profile, and W4 can read only its draw’s committed profile.

## Fit-only diagnosis of the earlier fixture

The six earlier D_fit trajectories reproduce the signed profile exactly. Its multipliers at steps 1, 25 and 100 were 0.04735, 0.77727 and 7.87016. Replacing the ad hoc absolute floor with the training-SD rule alone changed the fit empirical 90th-percentile SUP proxy from 14.738 to 14.869. The old relative floor replaced about 0.079% of cells and the additive absolute floor was smaller than the relative-floored field at every cell. Smoothing alone at one cell reduced the score proxy to 5.707. Changing the two fixed sine/cosine directions to eight seeded low-mode probes alone reduced it to 3.657. These contrasts describe changes to the sensitivity field on those six rows, holding the other W3 settings fixed. The old hybrid profile brought the step-25 fit width proxy from 35.66 to 4.30 and raised the step-100 proxy from 46.94 to 57.35. This implicates small local denominators and the late time multiplier on the old D_fit sample. The proxy uses an empirical percentile because six trajectories cannot yield a finite conformal rank at α=.10. The results cannot allocate causal contributions to held-out width, separate probe count from the basis change, or isolate interactions among smoothing, floors, and the profile.

## Reproduction

Use the Factory 7.2.3 development checkout and the two pinned runtimes. The 7.2.2 signed synthetic oracle bundle contains the prior fitted state and run manifest. Set `OLD_RUN` to its verified `runs/725` directory after extracting that disposable engineering bundle. Use unique output paths for each run.

```sh
HORIZON_SOURCE="$HORIZON_ROOT/source"
ENG="$FACTORY_ROOT/factory/engineering"
GEN="$ENG/generate_p2_audit.py"
CONFIG="$HORIZON_SOURCE/configs/task/p2_burgers1d_pilot.json"
export PYTHONPATH="$HORIZON_SOURCE/src"
"$SOLVER_PYTHON" "$GEN" --config "$CONFIG" --out "$WORK/old"
"$SOLVER_PYTHON" "$GEN" --config "$CONFIG" --out "$WORK/design" --seed 20261124 --counts 0 24 0 0
export PYTHONPATH="$HORIZON_SOURCE/src:$ENG:$FACTORY_ROOT/factory"
"$FACTORY_PYTHON" "$HORIZON_SOURCE/scripts/p2_w34_settings.py" \
  --old-cohort "$WORK/old" --old-run "$OLD_RUN" --design "$WORK/design" --out "$WORK/frozen.json"
PYTHONPATH="$HORIZON_SOURCE/src" "$SOLVER_PYTHON" "$GEN" --config "$CONFIG" \
  --out "$WORK/fresh" --seed 20261125 --counts 32 24 19 10
"$FACTORY_PYTHON" "$HORIZON_SOURCE/scripts/p2_w34_qualification.py" \
  --cohort "$WORK/fresh" --freeze "$WORK/frozen.json" --out "$WORK/qualification"
cd "$HORIZON_SOURCE"
.venv/bin/python -m pytest -q
```

Use the Horizon G0 lock for Exponax/JAX and the Factory `requirements-p2-cpu-audit.lock` for Torch/Factory. The qualification creates a disposable owner home and maintainer key, verifies the installed K2 kernel, completes the ordinary `advance` path, and uses a clean verifier directory with explicitly pinned public keys. Code digests from the prospective freeze are checked before copying the worker into the study, then Factory registers the transitive local modules. The retained fields are checked against the signed W4 profile and the independent field oracle. Replay verifies field-to-score arithmetic and the W3 × profile relation. It does not rerun the FNO or refit the model and W4 candidate selection.

## Scope

The prospective design uses one FNO training seed, two D_fit draws and one paired calibration/test split per fit draw. The brief's five/20 draw protocol, scale, other tasks/horizons/scores, P5 smoothing selection, T-data-2, JVP/μ/log variants, MPS, and CPU/GPU cost equivalence require further work. The audited path deliberately retains bounded fields rather than qualifying the production cache policy. This fixture is not the official G1 experiment, and it leaves preregistration, production signing and scientific submission gated.

## Observed prospective results

T=100, SUP, α=.10, rank 18 of 19. Coverage counts are for 10 test trajectories per draw. Relative widths use the study's full-width convention.

| Draw | Arm | Covered | Threshold | Mean relative width | Relative width at 25 | Relative width at 100 | Integrated width |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | B01 | 7/10 | 2.425533 | 8.949 | 7.016 | 12.772 | 4.851 |
| 0 | W3 | 6/10 | 1.443511 | 6.267 | 3.402 | 13.376 | 2.019 |
| 0 | W4 | 9/10 | 2.573546 | 10.333 | 3.005 | 30.513 | 2.792 |
| 1 | B01 | 8/10 | 2.580992 | 9.825 | 8.201 | 13.307 | 5.162 |
| 1 | W3 | 10/10 | 3.292819 | 9.639 | 6.045 | 17.652 | 3.420 |
| 1 | W4 | 10/10 | 2.569364 | 6.795 | 2.498 | 17.652 | 2.054 |

The design selected smoothing of three cells and η=.02. The parametric profile won both protected D_fit selections. Unit tests enforce the LP coefficient/reciprocal relation from score to calibrated band, including global rescaling. The installed run selected neither LP nor q, so the held-out results qualify the selected parametric variant. The test coverage counts and widths are descriptive across two overlapping draws.

The installed confirm-and-audit step took 74.09 s. The training worker took 2.18 s and the two fitting workers took 8.92 s total. Peak per-worker RSS was 513.95 MiB in fitting. Field retention used 35,637,504 bytes (33.99 MiB) with at most 12 trajectories per chunk. The signed archive used 45,501,836 bytes. The CPU solver took 0.460 s to compile and 1.406 s to solve all 85 fresh trajectories.

The clean verifier checked all 174 trajectory–arm pairs. The maximum absolute direct-versus-cached SUP difference was 4.44 × 10⁻¹⁶ and the maximum independent reduction/metric difference was 1.42 × 10⁻¹⁴. Reloading the serialized model reproduced the fixed forward probe exactly. The unperturbed embedding in the nine-way float32 sensitivity batch differed from the standalone scored forecast by at most 3.34 × 10⁻⁶. The sensitivity finite differences use the unperturbed member of their own batch. The direct and cached audit both score the same retained standalone forecast. Factory replay verifies the retained arrays and arithmetic, so it does not establish regeneration invariance across different inference batch layouts.

Installed runtime versions were CPython 3.12.8, Torch 2.15.0, NumPy 2.5.0, SciPy 1.18.0 and jaxtyping 0.3.11. The separate solver runtime used Exponax 0.2.1, JAX 0.12.0.1 and float64. The frozen design, detailed measurements and pinned-key replay output are stored in the adjacent Factory's `dist/` with prefix `p2-w34-7.2.3-`.
