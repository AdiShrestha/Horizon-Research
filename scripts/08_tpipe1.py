"""T-pipe-1 (IC-02 WP-E step 5): recompute cached Stage 1 statistics from the stored fields.

For 20 showcase trajectories (evenly spaced indices) recompute M, e_l2, m_{w0,w3}, p{gamma}_{w0,w3},
wbar and wl2 from <stats>/fields/{error,w3_default}.npy with independent code (the frozen
``conformal.weights.step_maxima`` for the step maxima, sorting for the quantiles) and compare with
<stats>/stats.parquet. Tolerance 1e-6 relative, fixed before the first run. Writes
<stats>/tpipe1.json and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from trajcp.conformal.weights import step_maxima
from trajcp.rollout.reduce import GAMMAS

TOL = 1e-6


def worst_relative(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    same_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    ok = np.where(same_inf, 0.0, np.abs(a - b) / np.maximum(np.abs(b), 1e-300))
    return float(np.nanmax(ok))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()
    table = pl.read_parquet(args.stats / "stats.parquet")
    err_all = np.load(args.stats / "fields/error.npy", mmap_mode="r")
    w3_all = np.load(args.stats / "fields/w3_default.npy", mmap_mode="r")
    n = err_all.shape[0]
    pick = np.unique(np.linspace(0, n - 1, min(20, n)).round().astype(int))
    err = np.asarray(err_all[pick], dtype=np.float64)
    w3 = np.asarray(w3_all[pick], dtype=np.float64)
    ones = np.ones_like(err)
    cells = err.shape[-1]
    results: dict[str, float] = {}

    def col(name: str) -> np.ndarray:
        return np.stack(table[name].to_numpy())[pick]

    results["M"] = worst_relative(col("M"), np.abs(err).max(-1))
    results["e_l2"] = worst_relative(col("e_l2"), np.sqrt(np.mean(err**2, -1)))
    for tag, w in (("w0", ones), ("w3", w3)):
        results[f"m_{tag}"] = worst_relative(col(f"m_{tag}"), step_maxima(err, w))
        r = np.sort(np.abs(err) / w, axis=-1)
        for gamma in GAMMAS:
            j = math.ceil(round((1.0 - gamma) * cells, 12))
            results[f"p{gamma:g}_{tag}"] = worst_relative(
                col(f"p{gamma:g}_{tag}"), r[..., j - 1]
            )
        results[f"wbar_{tag}"] = worst_relative(col(f"wbar_{tag}"), w.mean(-1))
        results[f"wl2_{tag}"] = worst_relative(
            col(f"wl2_{tag}"), np.sqrt(np.mean(w**2, -1))
        )
    report = {
        "trajectories": pick.tolist(),
        "tolerance": TOL,
        "worst_relative": results,
        "pass": bool(max(results.values()) <= TOL),
    }
    (args.stats / "tpipe1.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"max": max(results.values()), "pass": report["pass"]}))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
