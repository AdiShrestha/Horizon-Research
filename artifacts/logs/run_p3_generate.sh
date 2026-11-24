#!/bin/bash
# A15/A18: P3 TRAIN (1,000) then DEV (1,500) at the A15 burn-in.
cd "/Users/adi/Desktop/Projects/Horizon Research/source"
export PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
echo "commit $(git rev-parse --short HEAD) (+ uncommitted 01_generate_p3.py, 09_p3_burnin.py, ks.py additions)"
date; .venv-run/bin/python scripts/01_generate_p3.py --stream TRAIN --split train --n 1000 --out artifacts/data/p3; echo "exit $?"
date; .venv-run/bin/python scripts/01_generate_p3.py --stream DEV --split dev --n 1500 --out artifacts/data/p3; echo "exit $?"
date
