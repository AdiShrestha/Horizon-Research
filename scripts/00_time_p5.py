"""Time 100 P5-shaped Exponax solves using synthetic vorticity fixtures on CPU."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"

import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def benchmark():
    grid, output_grid, horizon, substeps = 128, 64, 100, 5
    count, batch_size = 100, 5
    macro_dt = 0.01
    domain = 2 * np.pi
    stepper = ex.stepper.KolmogorovFlowVorticity(
        2, domain, grid, macro_dt / substeps, diffusivity=0.001, drag=-0.1,
        injection_mode=4, order=4, dealiasing_fraction=2 / 3,
    )
    coordinate = jnp.arange(grid, dtype=jnp.float64) * domain / grid
    modes = ((1, 2), (2, 1), (3, -2), (4, 1))
    waves = jnp.stack([i * coordinate[:, None] + j * coordinate[None, :] for i, j in modes])
    basis = jnp.concatenate((jnp.cos(waves), jnp.sin(waves)), axis=0)

    def project(spectrum):
        half = output_grid // 2
        crop = jnp.concatenate((spectrum[..., :half, :half + 1], spectrum[..., -half:, :half + 1]), axis=-2)
        crop = crop.at[..., half, :].set(0).at[..., -1].set(0)
        return jnp.fft.irfft2(crop / grid**2, s=(output_grid, output_grid), norm="forward")

    def solve(index):
        key = jax.random.fold_in(jax.random.PRNGKey(0x50354231), index)
        amplitudes = 0.1 * jax.random.normal(key, (len(basis),), dtype=jnp.float64)
        initial = jnp.einsum("i,ixy->xy", amplitudes, basis)[None]

        def macro_step(state, _):
            state = jax.lax.fori_loop(0, substeps, lambda _, value: stepper.step_fourier(value), state)
            return state, project(state)

        initial_spectrum = jnp.fft.rfft2(initial)
        _, values = jax.lax.scan(macro_step, initial_spectrum, None, length=horizon)
        return values

    ids = np.arange(count, dtype=np.uint32)
    started = time.perf_counter()
    run = jax.jit(jax.vmap(solve)).lower(ids[:batch_size]).compile()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    peak = 0.0
    for start in range(0, count, batch_size):
        result = np.asarray(run(ids[start:start + batch_size]))
        if result.dtype != np.float64 or not np.isfinite(result).all():
            raise ValueError("P5 benchmark returned invalid float64 values")
        peak = max(peak, float(abs(result).max()))
    elapsed = time.perf_counter() - started
    return {
        "status": "measured", "intended_use": "synthetic P5-shaped timing fixture only",
        "backend": jax.default_backend(), "dtype": str(result.dtype),
        "trajectories": count, "batch_size": batch_size, "solver_grid": grid,
        "output_grid": output_grid, "horizon": horizon, "substeps": substeps,
        "macro_dt": macro_dt, "domain_extent": domain, "diffusivity": 0.001,
        "drag": -0.1, "forcing_mode": 4, "compile_seconds": compile_seconds,
        "generate_seconds": elapsed, "trajectories_per_second": count / elapsed,
        "max_absolute_vorticity": peak,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/p5_timing.json")
    args = parser.parse_args()
    result = json.dumps(benchmark(), indent=2, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result)
    print(result, end="")
