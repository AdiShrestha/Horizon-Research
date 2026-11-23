"""Solver process for the bounded P2 refinement and audit identity check."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys

from p2_tdata2_solver import stream

import jax
import numpy as np

from trajcp.data.exponax_gen import BurgersPilot, PILOT_SEED_TAG, batched_burgers


def identities(plan):
    all_keys = {}
    for seed, count in ((20261123, 65), (20261124, 24), (20261125, 85)):
        stem = jax.random.fold_in(jax.random.PRNGKey(seed), PILOT_SEED_TAG)
        keys = jax.vmap(lambda i: jax.random.fold_in(stem, i))(np.arange(count, dtype=np.uint32))
        all_keys[str(seed)] = np.asarray(keys).tolist()
    cohorts = {}
    spec = BurgersPilot(**plan["solver_config"])
    for name, seed in plan["cohorts"].items():
        # Only initial states enter the check. One macro-step avoids compiling the full horizon.
        fn = batched_burgers(replace(spec, steps=1, master_seed=seed), spatial_refinement=2,
                             temporal_refinement=2)
        ids = np.array(plan["audit_ids"], dtype=np.uint32)
        fields, (nu, _, _, keys) = jax.device_get(fn(ids))
        fingerprints = []
        for u, p in zip(fields[:, 0, 0], nu, strict=True):
            canonical = np.asarray(u, dtype="<f4").tobytes() + np.asarray(p, dtype="<f4").tobytes()
            fingerprints.append(sha256(canonical).hexdigest())
        cohorts[name] = {"fingerprints": fingerprints, "keys": keys.tolist()}
    return {"prior_keys": all_keys, "cohorts": cohorts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("identities", "stream"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.mode == "stream":
        stream(BurgersPilot(**plan["solver_config"]), plan)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = {"solver_peak_rss_bytes": int(rss if sys.platform == "darwin" else 1024 * rss)}
    else:
        result = identities(plan)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
