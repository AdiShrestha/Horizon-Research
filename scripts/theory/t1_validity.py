"""Finite-sample validity of trajectory-level split conformal calibration (theorem T1 checks).

For each score model and calibration size n, R splits each draw n calibration and m = 600 test
trajectories independently, calibrate at alpha = 0.1 and record:

  (a) mean coverage against k/(n+1), inside the 99.9% band from the Beta-Binomial variance
      [Var(C) + E C(1 - C)/m] / R; for one test trajectory per split also the binomial band;
  (b) calibration-conditional coverage C = F(q) against Beta(k, n+1-k) by KS. F is exact for
      uniform scores; for field scores F is the empirical CDF of a shared independent reference
      sample of --reference scores (standard error about sqrt(0.09 / reference) near 0.9);
  (c) covered counts out of 600 against Beta-Binomial(600, k, n+1-k) by KS with a Monte Carlo
      null (the law is discrete);
  (d) prefix uniformity (field models): with the single threshold, the event "every prefix of a
      test trajectory is covered" equals "the full trajectory is covered" (mismatches must be 0),
      and each prefix horizon is covered at least at the full-horizon rate.

Tolerance: every check at the 0.1% level (band z = 3.29, p-values > 0.001), fixed before the
first run. Seeds [31415926, 101, model, n] (theory checks only).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import stats

from trajcp.conformal import core
from trajcp.theory import fields

ALPHA, M_TEST, Z = 0.1, 600, float(stats.norm.ppf(0.9995))
HORIZONS = (10, 25, 50)
MODELS = {"uniform": 0, "ar1": 1, "gp": 2}


def step_draws(model: str, rng: np.random.Generator, n: int) -> np.ndarray:
    if model == "uniform":
        return fields.uniform_scores(rng, n)[:, None]
    return fields.step_scores(rng, n, length=0.0 if model == "ar1" else 3.0)


def discrete_ks_pvalue(counts: np.ndarray, law: stats.rv_frozen, rng: np.random.Generator) -> float:
    support = np.arange(M_TEST + 1)
    cdf = law.cdf(support)

    def stat(x: np.ndarray) -> float:
        emp = np.cumsum(np.bincount(x, minlength=M_TEST + 1)) / x.size
        return float(np.abs(emp - cdf).max())

    observed = stat(counts)
    null = [stat(np.asarray(law.rvs(size=counts.size, random_state=rng))) for _ in range(1000)]
    return (1 + sum(v >= observed for v in null)) / 1001


def run(model: str, n: int, splits: int, reference: int, chunk: int) -> dict[str, object]:
    rng = np.random.default_rng([31415926, 101, MODELS[model], n])
    k = core.conformal_rank(n, ALPHA)
    ref = None
    if model != "uniform":
        ref = np.sort(
            step_draws(model, np.random.default_rng([31415926, 102, MODELS[model]]), reference).max(
                1
            )
        )
    counts = np.empty(splits, dtype=np.int64)
    first = np.empty(splits, dtype=bool)
    conditional = np.empty(splits)
    prefix_mismatch = 0
    prefix_cover = np.zeros(len(HORIZONS), dtype=np.int64)
    for start in range(0, splits, chunk):
        size = min(chunk, splits - start)
        steps = step_draws(model, rng, size * (n + M_TEST)).reshape(size, n + M_TEST, -1)
        final = steps.max(axis=2)
        for s in range(size):
            q = core.calibrate(final[s, :n], ALPHA)
            test = final[s, n:]
            counts[start + s] = int(np.count_nonzero(test <= q))
            first[start + s] = bool(test[0] <= q)
            conditional[start + s] = (
                min(q, 1.0) if ref is None else np.searchsorted(ref, q, "right") / ref.size
            )
            if steps.shape[2] > 1:
                prefix = np.maximum.accumulate(steps[s, n:], axis=1)
                prefix_mismatch += int(np.count_nonzero(np.all(prefix <= q, axis=1) != (test <= q)))
                prefix_cover += [int(np.count_nonzero(prefix[:, h - 1] <= q)) for h in HORIZONS]
    target = k / (n + 1)
    beta = stats.beta(k, n + 1 - k)
    law = stats.betabinom(M_TEST, k, n + 1 - k)
    mean = counts.sum() / (splits * M_TEST)
    band = Z * math.sqrt(float(law.var()) / splits) / M_TEST
    band_one = Z * math.sqrt(target * (1 - target) / splits)
    out: dict[str, object] = {
        "model": model,
        "n": n,
        "k": k,
        "splits": splits,
        "m": M_TEST,
        "target": target,
        "a_mean_coverage": mean,
        "a_band": band,
        "a_pass": abs(mean - target) <= band,
        "a_m1_coverage": float(first.mean()),
        "a_m1_binomial_band": band_one,
        "a_m1_pass": abs(first.mean() - target) <= band_one,
        "b_ks_p": float(stats.kstest(conditional, beta.cdf).pvalue),
        "b_conditional_mean": float(conditional.mean()),
        "b_conditional_sd": float(conditional.std()),
        "b_beta_sd": float(beta.std()),
        "b_reference": 0 if ref is None else reference,
        "c_ks_p_mc": discrete_ks_pvalue(counts, law, rng),
    }
    out["b_pass"] = out["b_ks_p"] > 0.001
    out["c_pass"] = out["c_ks_p_mc"] > 0.001
    if model != "uniform":
        out["d_prefix_mismatches"] = prefix_mismatch
        out["d_prefix_coverage"] = dict(
            zip(map(str, HORIZONS), (prefix_cover / (splits * M_TEST)).tolist(), strict=True)
        )
        # Integer comparison: each prefix covered-count against the full-horizon covered-count.
        out["d_pass"] = prefix_mismatch == 0 and all(
            int(c) >= int(counts.sum()) for c in prefix_cover
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=10_000)
    parser.add_argument("--reference", type=int, default=10_000_000)
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "uniform:20",
            "uniform:50",
            "uniform:100",
            "uniform:600",
            "ar1:100",
            "ar1:600",
            "gp:100",
            "gp:600",
        ],
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for config in args.configs:
        model, n = config.split(":")
        start = time.perf_counter()
        r = run(model, int(n), args.splits, args.reference, args.chunk)
        r["seconds"] = time.perf_counter() - start
        results.append(r)
        print(json.dumps(r, default=lambda o: o.item()), flush=True)
        (args.out / "t1.json").write_text(json.dumps(results, indent=2, default=lambda o: o.item()))


if __name__ == "__main__":
    main()
