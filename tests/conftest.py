"""Split the suite across the two interpreters (A12).

``.venv`` is the core and oracle environment (no torch, no jax); ``.venv-run`` is the durable run
environment. Ported test files that need torch or jax are collected only where those import.
"""

from __future__ import annotations

import importlib.util

_NEEDS = {
    "test_data.py": ("jax", "exponax"),
    "test_training.py": ("torch", "safetensors"),
    "test_mps.py": ("torch",),
    "test_oracles.py": ("mapie",),
}

collect_ignore = [
    name
    for name, modules in _NEEDS.items()
    if any(importlib.util.find_spec(m) is None for m in modules)
]
