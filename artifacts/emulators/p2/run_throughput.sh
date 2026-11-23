#!/bin/bash
# IC-02 WP-C step 1: timing sweep (quiet machine, one GPU job at a time).
cd "/Users/adi/Desktop/Projects/Horizon Research/source"
export PYTHONPATH=src
OUT=artifacts/emulators/p2/throughput_raw.jsonl
for dev in mps cpu; do for b in 64 256 1024; do
  .venv-run/bin/python scripts/05_throughput.py --data artifacts/data/p2 --out $OUT --mode epoch --device $dev --batch $b --epochs 3
done; done
