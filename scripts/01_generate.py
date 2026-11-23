"""Generate and audit a standalone P2 engineering pilot."""

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trajcp.data.audit import audit_refinement
from trajcp.data.exponax_gen import PILOT_SEED_TAG, BurgersPilot, batched_burgers

ROOT = Path(__file__).resolve().parents[1]


def generate(config, out, count, batch_size, audit_count):
    if not 0 < audit_count <= count <= 2**32 or not 0 < batch_size <= count:
        raise ValueError("invalid count, batch size, or audit count")
    solve = batched_burgers(config)
    timings = {}
    ids = np.arange(count, dtype=np.uint32)
    audit_ids = np.linspace(0, count - 1, audit_count, dtype=np.uint32)
    coarse_audit = np.empty((audit_count, config.steps + 1, 1, config.grid))
    metadata = []
    projection_grid = 2 * np.pi * np.fft.rfftfreq(config.grid, config.domain_extent / config.grid)
    max_mean_drift = 0.0
    max_energy_increase = 0.0
    max_storage_error = 0.0
    terminal_rms_ratios = []
    started = time.perf_counter()
    base = solve.lower(np.resize(ids, batch_size)).compile()
    timings["coarse_compile_seconds"] = time.perf_counter() - started
    saved = np.lib.format.open_memmap(
        out / "trajectories.npy", mode="w+", dtype=np.float32,
        shape=(count, config.steps + 1, 1, config.grid),
    )
    started = time.perf_counter()
    for lo in range(0, count, batch_size):
        batch = ids[lo:lo + batch_size]
        values, (nus, slopes, amplitudes, keys) = jax.device_get(base(np.resize(batch, batch_size)))
        values = values[:len(batch)]
        if values.dtype != np.float64 or not np.isfinite(values).all():
            raise ValueError("solver returned invalid float64 states")
        for offset, trajectory_id in enumerate(batch):
            initial = values[offset, 0, 0]
            spectrum = np.fft.rfft(initial, norm="forward")
            power = abs(spectrum)**2
            gradient = np.fft.irfft(1j * projection_grid * spectrum, n=config.grid, norm="forward")
            metadata.append({
                "namespace": config.namespace,
                "trajectory_id": int(trajectory_id),
                "key_word_0": int(keys[offset, 0]), "key_word_1": int(keys[offset, 1]),
                "nu": float(nus[offset]), "spectral_slope": float(slopes[offset]),
                "initial_rms": float(amplitudes[offset]),
                "initial_gradient_rms": float(np.sqrt(np.mean(gradient**2))),
                "initial_sup": float(abs(initial).max()),
                "spectral_centroid": float(np.dot(np.arange(power.size), power) / power.sum()),
            })
        selected = np.flatnonzero(np.isin(audit_ids, batch))
        coarse_audit[selected] = values[audit_ids[selected] - lo]
        saved[lo:lo + len(batch)] = values
        max_storage_error = max(max_storage_error, float(abs(saved[lo:lo + len(batch)] - values).max()))
        mean = values.mean(axis=-1)
        max_mean_drift = max(max_mean_drift, float(abs(mean - mean[:, :1]).max()))
        energy = np.mean(values**2, axis=-1)
        max_energy_increase = max(max_energy_increase, float(np.diff(energy, axis=1).max()))
        terminal_rms_ratios.extend(np.sqrt(energy[:, -1, 0] / energy[:, 0, 0]).tolist())
    saved.flush()
    del saved
    pq.write_table(pa.Table.from_pylist(metadata), out / "metadata.parquet")
    timings["coarse_generate_and_store_seconds"] = time.perf_counter() - started
    timings["coarse_trajectories_per_second"] = count / timings["coarse_generate_and_store_seconds"]

    refined_solver = batched_burgers(config, spatial_refinement=2, temporal_refinement=2)
    started = time.perf_counter()
    audit_runner = refined_solver.lower(audit_ids).compile()
    timings["refined_compile_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    refined, (_, _, _, audit_keys) = jax.device_get(audit_runner(audit_ids))
    timings["refined_solve_seconds"] = time.perf_counter() - started
    expected_keys = [[metadata[i]["key_word_0"], metadata[i]["key_word_1"]] for i in audit_ids]
    np.testing.assert_array_equal(audit_keys, expected_keys)
    np.testing.assert_allclose(coarse_audit[:, 0], refined[:, 0], rtol=0, atol=2e-14)
    # A different batch size and order must preserve IDs, parameters and fields.
    selected = audit_ids[[audit_count - 1, 0]]
    repeated, (repeated_nu, repeated_slope, repeated_rms, repeated_keys) = jax.device_get(solve(selected))
    np.testing.assert_array_equal(repeated_keys, [expected_keys[-1], expected_keys[0]])
    for values, field in ((repeated_nu, "nu"), (repeated_slope, "spectral_slope"), (repeated_rms, "initial_rms")):
        np.testing.assert_array_equal(values, [metadata[index][field] for index in selected])
    ordering_error = float(abs(repeated - coarse_audit[[audit_count - 1, 0]]).max())
    np.testing.assert_allclose(repeated, coarse_audit[[audit_count - 1, 0]], rtol=0, atol=1e-12)

    delta, per_trajectory = audit_refinement(coarse_audit, refined)
    np.save(out / "refinement_delta.npy", delta, allow_pickle=False)
    held_out_id, time_index, _, cell = np.unravel_index(
        abs(refined - coarse_audit).argmax(), refined.shape
    )
    # Diagnose the largest observed discrepancy using one extra paired solve
    # in time and another paired solve in space.
    worst_id = audit_ids[held_out_id:held_out_id + 1]
    started = time.perf_counter()
    time_refined, _ = jax.device_get(batched_burgers(config, temporal_refinement=2)(worst_id))
    further_refined, _ = jax.device_get(
        batched_burgers(config, spatial_refinement=4, temporal_refinement=2)(worst_id)
    )
    timings["additional_refinement_seconds_including_compile"] = time.perf_counter() - started
    return {
        "intended_use": "local engineering only, excluded from study cohorts",
        "config": config.__dict__,
        "seed_mapping": f"fold_in(fold_in(PRNGKey(master_seed), {PILOT_SEED_TAG}), trajectory_id)",
        "trajectory_ids": [0, count - 1],
        "snapshot_shape": [count, config.steps + 1, 1, config.grid],
        "batch_size": batch_size,
        "solver": "Exponax conservative Burgers ETDRK4, 2/3 dealiasing",
        "backend": jax.default_backend(),
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("jax", "jaxlib", "exponax", "numpy", "pyarrow")},
        "timings": timings,
        "validity": {
            "all_finite": True, "max_mean_drift": max_mean_drift,
            "max_positive_step_mean_square_change": max_energy_increase,
            "max_float32_storage_error": max_storage_error,
            "max_reordered_rebatched_float64_error": ordering_error,
            "terminal_over_initial_rms_min_median_max": np.quantile(terminal_rms_ratios, [0, .5, 1]).tolist(),
        },
        "refinement": {
            "trajectory_ids": audit_ids.tolist(),
            "solver_grids": [config.solver_grid, 2 * config.solver_grid],
            "substeps": [config.substeps, 2 * config.substeps],
            "comparison_grid": config.grid,
            "pointwise_max_file": "refinement_delta.npy",
            "per_trajectory_sup_errors": per_trajectory.tolist(),
            "worst_location": {
                "trajectory_id": int(audit_ids[held_out_id]), "macro_step": int(time_index),
                "cell": int(cell),
            },
            "epsilon_num_w0": float(delta.max()),
            "worst_case_diagnostics": {
                "time_only_at_base_grid": float(abs(coarse_audit[held_out_id] - time_refined[0]).max()),
                "space_only_base_to_refined": float(abs(time_refined[0] - refined[held_out_id]).max()),
                "space_only_refined_to_8x": float(abs(refined[held_out_id] - further_refined[0]).max()),
                "extra_solver_grids": [config.solver_grid, 4 * config.solver_grid],
                "extra_substeps": 2 * config.substeps,
            },
            "epsilon_num_over_qhat": None,
            "t_data_2_status": "pending calibrated qhat and final weight field",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/task/p2_burgers1d_pilot.json")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/p2_engineering_pilot")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--audit-count", type=int, default=10)
    args = parser.parse_args()
    config = BurgersPilot(**json.loads(args.config.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        parser.error("--out already exists; give a fresh engineering output directory")
    with tempfile.TemporaryDirectory(dir=args.out.parent, prefix=".p2-pilot-") as scratch:
        staged = Path(scratch) / "pilot"
        staged.mkdir()
        report = generate(config, staged, args.count, args.batch_size, args.audit_count)
        report["sha256"] = {}
        for path in sorted(staged.iterdir()):
            with path.open("rb") as stream:
                report["sha256"][path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        (staged / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        staged.rename(args.out)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
