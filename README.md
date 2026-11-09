# Horizon-Research

Research code for conformal prediction bands around autoregressive neural PDE emulators. The goal is a band that contains the reference solution at every grid point and every step of a long emulator rollout simultaneously, with a finite-sample, distribution-free coverage guarantee, and that stays narrow enough to be useful over hundreds of steps.

## Status

Early development. Code is added in stages, starting with the calibration core, its tests and the reference-data generators. Experimental results will be released together with the accompanying preprint.

## Planned layout

| Path | Contents |
|---|---|
| `src/trajcp/` | Python package: reference-data generation, emulators, batched rollouts, conformal calibration, evaluation |
| `scripts/` | Pipeline entry points and theory simulations |
| `tests/` | Unit and property-based tests |
| `configs/` | Experiment configurations |
| `docs/` | Methods notes, data cards and reproduction instructions |
| `source/` | Self-contained code for one registered comparison study |

## Software

Python 3.12. Emulators are trained with PyTorch (eager mode; Apple-silicon GPUs supported). Reference solutions are computed with JAX on the CPU in double precision. All calibration arithmetic runs in double precision on the CPU. Setup instructions will be added with the first package release.

## Licence

BSD 3-Clause; see `LICENSE`.
