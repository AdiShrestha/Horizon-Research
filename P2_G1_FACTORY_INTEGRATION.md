# Factory 7.2.4 P2 T=200 engineering integration

Factory has a separate audited single-task mode for P2 G1. It requires T=200, the 256-cell field, SUP, α=.10, and one shared CPU FNO for B01, fitted W1-emp, intrinsic W3 and fitted W4. The mode retains bounded fields, binds the model state, frozen profiles and timing samples to the signed run, and checks the G1 coverage, width and cost criteria after the family seals. The T=100 audited route and the separately registered four-task T=100 primary analysis retain their prior contracts.

## Reproduction and qualification scope

The harness binds the prior synthetic study-side freeze and all of its retained files before creating a disposable owner home, development keys, and a fresh registration. It copies the original 32 training rows in their original order, and combines the previous draw-0 fit, calibration and test rows into the exact 83-unit pool required by Factory. Factory takes two private splits from that pool. The prior study-side run instead had distinct pools from different role seed domains. Thus the comparison serves as a regression of the implementation and inputs. Its native draw summaries need not equal the previous study-side summaries. Training seeds, weights, model precision, test quotas, horizon and thresholds rules are carried over. The measured native benchmark uses an unlabeled train-feature batch. The earlier benchmark used an audit-feature batch.

The normal installed CLI reaches attestation, owner approval, confirmation and completion. The independent harness estimates all calibration scores and test bands from exactly the retained native inputs, fields and signed draw memberships. The kernel's clean replay also checks W1 field/profile equality, W4 field/profile × W3 equality, and cached versus direct W4 SUP. Timing replay recomputes the cost statistic from seven committed matched observations after two warmups. It does not rerun host timing. The archived 4096-grid solver diagnostic remains the previous study-side T=200 engineering evidence. It is outside this native lifecycle.

```sh
H="$HORIZON_ROOT/source"
export PYTHONPATH="$H/src:$FACTORY_ROOT/factory:$FACTORY_ROOT/factory/engineering"
"$FACTORY_PYTHON" "$H/scripts/p2_g1_factory.py" --factory "$FACTORY_ROOT" \
  --prior "$H/data/p2_g1_rehearsal" --out "$WORK/native"
"$FACTORY_PYTHON" "$H/scripts/p2_g1_w3_diagnostics.py" \
  --freeze "$H/data/p2_g1_rehearsal/freeze.json" \
  --state "$H/data/p2_g1_rehearsal/train/state.npy" \
  --fit0 "$H/data/p2_g1_rehearsal/data/fit0" \
  --fit1 "$H/data/p2_g1_rehearsal/data/fit1" --out "$WORK/diagnostics.json"
cd "$FACTORY_ROOT"
PYTHONPATH=factory "$FACTORY_PYTHON" -m pytest -q factory/tests/test_p2_g1_protocol.py \
  factory/sf7/tests/test_trajectory_field_audit.py factory/sf7/tests/test_trajectory_workflow.py \
  factory/sf7/tests/test_trajectory_primary.py factory/sf7/tests/test_lifecycle.py \
  -k 'p2_g1_protocol or field_audit or installed_cli or interruption_and_stale or primary_cli or open_to_audit_through_the_cli'
```

On the original two D_fit scopes the locked W3 increased the fit-only score quantile by 1.598× and 1.405× while reducing the uncalibrated width coefficient to 0.942× and 0.912× of flat. The resulting fit-only width proxies were 1.505× and 1.282× of flat. Three-cell smoothing materially reduced the raw score tails. The relative floor was never active on those smoothed D_fit fields, and the absolute floor was smaller than the relative-floored value in every cell. This identifies the normalized-score quantile as the larger factor in the width proxy on those inputs. It does not establish which change would reduce widths at valid held-out coverage.

## Observed native run

The native FNO checkpoint was byte-identical to the study-side one. Factory reused the original 32 training inputs and the original 83 rows from panel 0. Same-role cross-draw overlap was 9 fit, 11 calibration, and 14 test trajectories. Counts and timing are descriptive engineering measurements with shared model and draw dependence.

| Private draw | Arm | Frozen threshold | Covered / 30 |
|---|---|---:|---:|
| 0 | B01 | 2.433274 | 30 |
| 0 | W1 | 4.976754 | 27 |
| 0 | W3 | 3.477858 | 26 |
| 0 | W4 | 3.389302 | 30 |
| 1 | B01 | 2.433274 | 29 |
| 1 | W1 | 3.945480 | 24 |
| 1 | W3 | 4.018297 | 30 |
| 1 | W4 | 3.376156 | 28 |

Coverage passed the fixed 22–30 screen in all eight cells at calibration rank 27 of 29. Mean full relative widths over 200 steps were 13.024 for B01, 25.310 for W1, 17.947 for W3 and 18.638 for W4. The smallest weighted width was W3, 37.80% wider than B01, so the 20% reduction target failed. The confined paired W3 timing ratio was 20.78× for 12 trajectories, above the 10× target. Other verification workloads were active during part of this native run. The prior study-side 19.37× timing and its width failure are retained as distinct observations.

The ordinary installed confirm and audit step took 270.93 s. All 47 confined CPU stage receipts passed the scope checks, including train and D_fit label scopes, blind forecasts, zero-fit W3, own-profile-only W1/W4 and an unlabeled benchmark. Peak worker RSS was 551.86 MiB in fitting. The run retained 193,336,320 bytes of forecast/envelope fields under the 192 MiB retention cap. The signed export was 162,418,152 bytes. The external reference checked 472 trajectory–arm pairs and matched full relative widths within 1.42 × 10⁻¹⁴. The clean Factory verifier independently checked those fields, with maximum cached SUP deviation 8.88 × 10⁻¹⁶ and maximum field-to-metric deviation 2.84 × 10⁻¹⁴.

The source-side regression remains reproducible with the extracted 7.2.3 development distribution. Its freeze binds the full 7.2.3 oracle source file, so the extended 7.2.4 file has a different digest even though the original scalar field arithmetic remains intact. The native harness checks the original evidence hash and all unchanged study algorithm hashes before applying its own audited kernel.

## Failure cases and bounded remediation

A clean K2 installation from the development package reproduced the 472 native field checks. A disposable test-signed modified bundle halved the lowest B01 calibration score, kept the conformal rank, threshold, test rows, and aggregates fixed, and rebuilt the run and ledger signature chain. The verifier rejected it specifically because its signed reduction disagreed with the field-derived score. The regression suite passed 11 selected tests, including the explicit T=200 mode requirement, remaining T=100 boundaries, a late-step-only miss, insufficient fit folds, a short retention budget, invalid timing, interrupted-family retirement, stale receipts and the scalar route. The Horizon suite passed 102 tests.

The isolated synthetic timing run used the same CPU float32 model, 12 input trajectories, identical perturbations and 200 steps. One rollout took a median 0.1353 s. The joint nine-member computation took 20.74× by median paired ratio. Median phase times were 0.53 ms for perturbations, 2943 ms for rollout, 11.57 ms for RMS reduction, 2.98 ms for smoothing and 3.67 ms for floors. The rollout phase dominated. Groups of three took 28.06× and a sequence of nine 12-trajectory calls took 9.15×. The sequential version changed the unperturbed forecast by at most 1.29 × 10⁻⁶ and W3 weights by at most 2.16 × 10⁻⁴ absolute, with 99th-percentile relative delta 1.63 × 10⁻⁴. The concurrently measured preliminary run put the same schedules at 23.89×, 28.68× and 10.13×, respectively. The sequential path is a candidate for a prospective timing check. There is no verified ≤10× result for confined execution on a fresh cohort.

A bounded follow-up could freeze two decisions before generating a fresh P2 pool.

1. Compare the existing joint path and the sequential schedule under the same 12-trajectory CPU worker budget. Keep the eight perturbations, floor and smoother fixed. Check forecast and weight parity at a prespecified tolerance, replay each schedule's calibrated band, and check the 10× gate under isolated and operational timing. Use the schedule that actually produces the scored field, including its unperturbed member.
2. Try one shape shrinkage candidate with square-root spatial contrast. At each trajectory and time divide W3 by its spatial RMS, square-root that dimensionless shape, normalize it back to unit RMS, and multiply by the original spatial RMS. Compare it with the locked W3 shape using D_fit-only folds and identical W4 selection rules if used for W4. Freeze the rule before the new calibration/test cohort. This directly addresses high normalized errors at locally small weights while bounding the candidate set. Keep B01, the existing weights, α, rank, width gate and prior negative results as the prespecified references.

Neither candidate was applied to the opened calibration or test panels. The official analysis, independently administered production custody and accelerator qualification remain open.
