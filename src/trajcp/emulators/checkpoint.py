"""Resumable training state as SafeTensors plus JSON, written atomically (standing rule 5).

Tensors (model parameters and optimizer moments) go to ``last.safetensors`` on the CPU; everything
else (epoch, curves, best value, elapsed seconds, resumes, optimizer hyper-parameters and step
counters) goes to ``last_state.json``. No pickle is used for model or optimizer state. The state
file is written after the tensor file and carries the epoch, so a crash between the two leaves
the previous consistent pair (the tensor file is written to a temporary name first, then both are
swapped in with ``os.replace``; the epoch stored in each is compared on load).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


def _flatten(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, torch.Tensor]:
    tensors = {
        f"model.{k}": v.detach().cpu().contiguous()
        for k, v in model.state_dict().items()
    }
    for index, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            tensors[f"opt.{index}.{name}"] = (
                torch.as_tensor(value).detach().cpu().contiguous()
            )
    return tensors


def save_state(
    directory: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    meta: dict[str, Any],
) -> None:
    """Write last.safetensors and last_state.json atomically; ``meta`` must be JSON-serializable."""
    tensors = _flatten(model, optimizer)
    groups = optimizer.state_dict()["param_groups"]
    payload = {**meta, "param_groups": groups, "tensor_epoch": meta["epoch"]}
    t_tmp, j_tmp = directory / "last.safetensors.tmp", directory / "last_state.json.tmp"
    save_file(tensors, str(t_tmp), metadata={"epoch": str(meta["epoch"])})
    j_tmp.write_text(json.dumps(payload))
    os.replace(t_tmp, directory / "last.safetensors")
    os.replace(j_tmp, directory / "last_state.json")


def load_state(
    directory: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any] | None:
    """Restore model and optimizer in place; return the metadata, or None when no state exists."""
    state_path, tensor_path = (
        directory / "last_state.json",
        directory / "last.safetensors",
    )
    if not (state_path.exists() and tensor_path.exists()):
        return None
    payload = json.loads(state_path.read_text())
    tensors = load_file(str(tensor_path))
    model.load_state_dict(
        {k[6:]: v for k, v in tensors.items() if k.startswith("model.")}
    )
    opt_state: dict[int, dict[str, torch.Tensor]] = {}
    for key, value in tensors.items():
        if key.startswith("opt."):
            _, index, name = key.split(".", 2)
            opt_state.setdefault(int(index), {})[name] = value.to(device)
    optimizer.load_state_dict(
        {"state": opt_state, "param_groups": payload["param_groups"]}
    )
    return payload
