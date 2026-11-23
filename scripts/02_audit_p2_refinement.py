"""Compare matched spatial grids and a timestep control on the ten P2 audit IDs."""

import argparse
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"

import jax
import numpy as np
import pyarrow.parquet as pq

from trajcp.data.audit import audit_refinement
from trajcp.data.exponax_gen import BurgersPilot, batched_burgers

ROOT = Path(__file__).resolve().parents[1]


def as_rows(coarse, refined):
    _, sup_errors = audit_refinement(coarse, refined)
    difference = abs(refined - coarse)
    peaks = difference.reshape(len(difference), -1).argmax(axis=1)
    steps, _, cells = np.unravel_index(peaks, difference.shape[1:])
    return [
        {"peak_error": float(error), "macro_step": int(step), "cell": int(cell)}
        for error, step, cell in zip(sup_errors, steps, cells, strict=True)
    ]


def measure(config, ids, grid_factor, time_factor):
    solve = batched_burgers(config, spatial_refinement=grid_factor, temporal_refinement=time_factor)
    start = time.perf_counter()
    executable = solve.lower(ids).compile()
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    values, params = jax.device_get(executable(ids))
    solve_seconds = time.perf_counter() - start
    if values.dtype != np.float64 or not np.isfinite(values).all():
        raise ArithmeticError("P2 refinement must produce finite CPU float64 fields")
    return values, params, {
        "solver_grid": config.solver_grid * grid_factor,
        "substeps": config.substeps * time_factor,
        "solver_dt": config.macro_dt / (config.substeps * time_factor),
        "batch": len(ids), "compile_seconds": compile_seconds,
        "solve_seconds": solve_seconds, "trajectories_per_second": len(ids) / solve_seconds,
    }


def diagnostic(pilot):
    original = json.loads((pilot / "report.json").read_text())
    config = BurgersPilot(**original["config"])
    ids = np.array(original["refinement"]["trajectory_ids"], dtype=np.uint32)
    if len(ids) != 10 or len(np.unique(ids)) != len(ids):
        raise ValueError("expected the ten distinct IDs in the original pilot audit")
    metadata = pq.read_table(pilot / "metadata.parquet").to_pylist()
    by_id = {row["trajectory_id"]: row for row in metadata}
    rows = [by_id[int(i)] for i in ids]
    keys = [[row["key_word_0"], row["key_word_1"]] for row in rows]
    nus = [row["nu"] for row in rows]
    snapshots = np.load(pilot / "trajectories.npy", mmap_mode="r")

    runs, fields = {}, {}
    for grid_factor, time_factor in ((1, 1), (1, 2), (1, 8), (2, 2), (2, 8), (4, 2), (4, 8)):
        values, (run_nus, _, _, run_keys), run = measure(config, ids, grid_factor, time_factor)
        np.testing.assert_array_equal(run_keys, keys)
        np.testing.assert_array_equal(run_nus, nus)
        np.testing.assert_allclose(values[:, 0], snapshots[ids, 0], rtol=0, atol=1.2e-7)
        grid = run["solver_grid"]
        substeps = run["substeps"]
        runs[f"{grid}/{substeps}"] = run
        fields[grid, substeps] = values

    base, middle, high = (config.solver_grid * factor for factor in (1, 2, 4))
    original_steps, control, fine = (config.substeps * factor for factor in (1, 2, 8))
    np.testing.assert_array_equal(fields[base, original_steps].astype(np.float32), snapshots[ids])
    reproduced_delta, _ = audit_refinement(fields[base, original_steps], fields[middle, control])
    np.testing.assert_array_equal(reproduced_delta, np.load(pilot / "refinement_delta.npy"))
    for field in fields.values():
        np.testing.assert_allclose(field[:, 0], fields[base, fine][:, 0], rtol=0, atol=2e-14)

    comparisons = {
        "space_512_1024": ((base, fine), (middle, fine)),
        "space_1024_2048": ((middle, fine), (high, fine)),
        "time_512": ((base, control), (base, fine)),
        "time_1024": ((middle, control), (middle, fine)),
        "time_2048": ((high, control), (high, fine)),
    }
    errors = {name: as_rows(fields[left], fields[right]) for name, (left, right) in comparisons.items()}
    summaries = {
        name: {"trajectory_id": int(ids[worst]), "nu": nus[worst], **peaks[worst]}
        for name, peaks in errors.items()
        for worst in [max(range(len(peaks)), key=lambda i: peaks[i]["peak_error"])]
    }
    trajectory_results = []
    for offset, (trajectory_id, nu) in enumerate(zip(ids, nus, strict=True)):
        peaks = {name: values[offset] for name, values in errors.items()}
        trajectory_results.append({
            "trajectory_id": int(trajectory_id), "nu": nu, **peaks,
            "spatial_peak_decreased": (
                peaks["space_1024_2048"]["peak_error"] < peaks["space_512_1024"]["peak_error"]
            ),
        })
    return {
        "intended_use": "paired spatial refinement of the same ten engineering pilot trajectories",
        "reproduce_from_workspace_root": [
            "uv venv --no-project --python 3.12.8 /tmp/horizon-g0",
            "uv pip install --python /tmp/horizon-g0/bin/python -r source/requirements-g0.lock",
            "PYTHONPATH=source/src /tmp/horizon-g0/bin/python source/scripts/02_audit_p2_refinement.py "
            "--out /tmp/horizon-p2-refinement-repeat.json",
        ],
        "environment_lock": "source/requirements-g0.lock",
        "engineering_interpretation": (
            "Use 1024 cells and 20 substeps as the provisional setting for further paired pilot "
            "work, with 2048 cells as the comparison grid. Assess the final resolution "
            "against the calibrated T-data-2 criterion."
        ),
        "python": platform.python_version(), "backend": jax.default_backend(),
        "versions": {name: version(name) for name in ("exponax", "jax", "jaxlib", "numpy", "pyarrow")},
        "config": original["config"],
        "seed_mapping": original["seed_mapping"],
        "verified_original_fields_and_audit_bitwise": True,
        "comparison_grid": config.grid,
        "matched_substeps": fine,
        "time_control_substeps": [control, fine],
        "error_definition": "maximum absolute projected difference over macro-steps and cells",
        "solve_seconds_relative_to_512_10": {
            label: case["solve_seconds"] / runs[f"{base}/{original_steps}"]["solve_seconds"]
            for label, case in runs.items()
        },
        "timings": runs,
        "worst_cases": summaries,
        "spatial_peak_decreased_count": sum(row["spatial_peak_decreased"] for row in trajectory_results),
        "trajectories": trajectory_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=ROOT / "artifacts/p2_engineering_pilot")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/p2_spatial_refinement.json")
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.pilot.resolve()):
        parser.error("write the diagnostic outside the original pilot")
    report = diagnostic(args.pilot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({
        "worst_cases": report["worst_cases"],
        "spatial_peak_decreased_count": report["spatial_peak_decreased_count"],
        "solve_seconds_relative_to_512_10": report["solve_seconds_relative_to_512_10"],
        "report": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
