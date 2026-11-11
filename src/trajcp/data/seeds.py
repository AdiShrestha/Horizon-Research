"""Seed streams and stateless per-trajectory keys.

key = fold_in(PRNGKey(master + task_index * 1_000_003), i): trajectory i is identical whatever the
batching or ordering. Each stream serves one purpose only. The confirmation stream is refused unless
the caller passes ``allow_confirm=True``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

STREAMS = {
    "TRAIN": 20261104,
    "DEV": 20261105,
    "AUX": 20261106,
    "SUBSTUDY": 2026111009,
    "CONFIRM": 20261231,
}
TASK_INDEX = {"p1": 1, "p2": 2, "p3": 3, "p4": 4, "p5": 5}
_OFFSET = 1_000_003


def stream_key(stream: str, task: str, allow_confirm: bool = False) -> jax.Array:
    if stream not in STREAMS:
        raise KeyError(f"unknown stream {stream!r}")
    if stream == "CONFIRM" and not allow_confirm:
        raise PermissionError("the confirmation stream is closed")
    return jax.random.PRNGKey(STREAMS[stream] + TASK_INDEX[task] * _OFFSET)


def trajectory_key(stream: str, task: str, index: int, allow_confirm: bool = False) -> jax.Array:
    return jax.random.fold_in(stream_key(stream, task, allow_confirm), index)


# Component tags: append new names, never renumber. A trajectory key is never split directly;
# each random component of a trajectory draws from its own component key.
COMPONENTS = {"ic": 0, "mu": 1, "perturbation": 2, "noise": 3}
_COMPONENT_BASE = 1 << 20


def component_key(key: jax.Array, name: str) -> jax.Array:
    """Independent key for one named random component of a trajectory."""
    if name not in COMPONENTS:
        raise KeyError(f"unknown component {name!r}")
    return jax.random.fold_in(key, _COMPONENT_BASE + COMPONENTS[name])
