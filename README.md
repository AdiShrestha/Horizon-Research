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

Python 3.12. Emulators are trained with PyTorch (eager mode; Apple-silicon GPUs supported). Reference solutions are computed with JAX on the CPU in double precision. All calibration arithmetic runs in double precision on the CPU.

## Setup

The environment is managed with [uv](https://docs.astral.sh/uv/) and pinned in `uv.lock` (macOS arm64 and Linux x86-64).

```bash
uv sync --frozen   # create .venv from uv.lock
make test          # fast test suite (CPU, float64 core)
make lint          # ruff, mypy on the conformal core, dtype guard
```

Conventions the code and tests enforce:

- `src/trajcp/conformal/` works on float64 NumPy arrays on the CPU only. `make lint` rejects `float32`, `.cuda(`, `torch.compile(` and `autocast` in that package.
- Emulators run in eager float32; `torch.compile` and `torch.autocast` are not used.
- Python writes no bytecode (the Makefile exports `PYTHONDONTWRITEBYTECODE=1`), and `make test` fails if bytecode is found under `source/`; `make clean-source-bytecode` removes it.
- `source/requirements.lock` is the hash-pinned dependency list of the registered comparison study; `make export-lock` regenerates it from `uv.lock`, and a test fails if the two disagree.

## Licence

BSD 3-Clause; see `LICENSE`.
