"""Generate isolated engineering roles for the frozen T=200 P2 rehearsal."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from p2_tdata2_solver import BurgersPilot, batched_burgers, jax, np


def generate(plan, out):
    out.mkdir()
    config = BurgersPilot(**plan["solver"])
    timings = {}
    for role, spec in plan["roles"].items():
        directory = out / role
        directory.mkdir()
        ids = np.arange(spec["count"], dtype=np.uint32)
        cases = (4, 8) if role == "audit" else (4,)
        timings[role] = {}
        for spatial in cases:
            fn = batched_burgers(replace(config, master_seed=spec["seed"]), spatial_refinement=spatial,
                                 temporal_refinement=2)
            compiled, time_compile, time_solve = {}, 0., 0.
            for start in range(0, len(ids), plan["chunk"]):
                batch = ids[start:start + plan["chunk"]]
                if len(batch) not in compiled:
                    t = time.perf_counter()
                    compiled[len(batch)] = fn.lower(batch).compile()
                    time_compile += time.perf_counter() - t
                t = time.perf_counter()
                fields, (nu, _, _, keys) = jax.device_get(compiled[len(batch)](batch))
                time_solve += time.perf_counter() - t
                assert fields.shape == (len(batch), 201, 1, 256) and fields.dtype == np.float64
                assert np.isfinite(fields).all()
                suffix = f"{start:03d}.npz"
                if spatial == 4:
                    np.savez(directory / ("input-" + suffix), ids=batch, initial=fields[:, 0, 0],
                             nu=nu[:, None], keys=keys)
                    np.save(directory / f"truth-{start:03d}.npy", fields[:, 1:, 0], allow_pickle=False)
                else:
                    with np.load(directory / ("input-" + suffix), allow_pickle=False) as before:
                        np.testing.assert_array_equal(keys, before["keys"])
                        np.testing.assert_array_equal(nu[:, None], before["nu"])
                        np.testing.assert_allclose(fields[:, 0, 0], before["initial"], rtol=0, atol=2e-14)
                    np.save(directory / f"compare-{start:03d}.npy", fields[:, 1:, 0], allow_pickle=False)
            timings[role][str(config.solver_grid*spatial)] = {"compile_seconds": time_compile, "solve_seconds": time_solve}
    (out / "solver-timings.json").write_text(json.dumps(timings, indent=2)+"\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("freeze", type=Path)
    cli.add_argument("out", type=Path)
    args = cli.parse_args()
    generate(json.loads(args.freeze.read_text()), args.out)
