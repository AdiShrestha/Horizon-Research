#!/bin/bash
# IC-02 WP-C step 2 (A07): E1 seed 0 at B=500 then 2B=1000 epochs, MPS, batch 64.
cd "/Users/adi/Desktop/Projects/Horizon Research/source"
export PYTHONPATH=src
for E in 500 1000; do
  date; .venv-run/bin/python scripts/02_train_e1.py --data artifacts/data/p2 --out artifacts/emulators/p2/e1_seed0_b$E --seed 0 --epochs $E --device mps --batch 64 --peak-lr 1e-3
  date
done
