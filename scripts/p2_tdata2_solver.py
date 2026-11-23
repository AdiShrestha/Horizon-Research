"""Stream paired projected CPU float64 refinements in chunks to the audit driver."""
import argparse
from io import BytesIO
import json
import os
import struct
import sys
import time
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"

import jax  # noqa: E402
import numpy as np  # noqa: E402

from trajcp.data.exponax_gen import BurgersPilot, batched_burgers  # noqa: E402


def emit(payload):
    with BytesIO() as buf:
        np.savez(buf, **payload)
        data = buf.getbuffer()
        sys.stdout.buffer.write(struct.pack("<Q", len(data)))
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        data.release()


def stream(config, frozen):
    cases = {f"g{g}_s{s}": batched_burgers(config, spatial_refinement=g // config.solver_grid,
                                           temporal_refinement=s // config.substeps)
             for g, s in frozen["solver_cases"]}
    compiled = {}
    ids = np.asarray(frozen["audit_ids"], dtype=np.uint32)
    for start in range(0, len(ids), frozen["chunk"]):
        indices = ids[start:start + frozen["chunk"]]
        payload, metadata, timings = {"ids": indices}, None, {}
        reference = None
        for label, solve in cases.items():
            cache_key = label, len(indices)
            elapsed_compile = 0.
            if cache_key not in compiled:
                began = time.perf_counter()
                compiled[cache_key] = solve.lower(indices).compile()
                elapsed_compile = time.perf_counter() - began
            began = time.perf_counter()
            values, params = jax.device_get(compiled[cache_key](indices))
            elapsed_solve = time.perf_counter() - began
            assert values.dtype == np.float64 and np.isfinite(values).all()
            initial = values[:, 0, 0]
            if metadata is None:
                metadata, reference = params, initial
            for left, right in zip(params, metadata, strict=True):
                np.testing.assert_array_equal(left, right)
            input_delta = float(np.max(np.abs(initial - reference)))
            np.testing.assert_allclose(initial, reference, rtol=0, atol=2e-14)
            payload[label] = values[:, 1:, 0]
            if label == frozen["weight_input_case"]:
                payload["initial"] = initial
            timings[label] = [elapsed_compile, elapsed_solve, input_delta]
        for name, array in zip(("nu", "slope", "amplitude", "keys"), metadata, strict=True):
            payload[name] = array
        payload["timings"] = np.array(list(timings.values()))
        emit(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("freeze", type=Path)
    args = parser.parse_args()
    frozen = json.loads(args.freeze.read_text())
    stream(BurgersPilot(**frozen["solver_config"]), frozen)
