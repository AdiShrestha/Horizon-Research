"""Device configuration for production training (A18).

float32 semantics everywhere: on CUDA, TF32 is disabled for matmul and cuDNN, cuDNN autotuning is
off and deterministic algorithms are requested (``CUBLAS_WORKSPACE_CONFIG`` is set before CUDA
initializes). On MPS the brief's §12.2 rules hold (no fast math, no fallback). No autocast and no
``torch.compile`` on any device.
"""

from __future__ import annotations

import os
import platform
import subprocess

import torch


def configure(name: str) -> torch.device:
    if name.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True, warn_only=True)
    elif name == "mps":
        for var in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"):
            if os.environ.get(var):
                raise SystemExit(f"{var} must be unset")
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.65")
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS requested but not available")
    elif name != "cpu":
        raise SystemExit(f"unknown device {name!r}")
    return torch.device(name)


def describe(device: torch.device) -> dict[str, object]:
    """Hardware and library facts recorded with every run."""
    info: dict[str, object] = {
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        info |= {
            "gpu": props.name,
            "gpu_memory_bytes": props.total_memory,
            "capability": f"{props.major}.{props.minor}",
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        }
        try:
            info["driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        except OSError:
            info["driver"] = None
    for var in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURMD_NODENAME", "PBS_JOBID"):
        if os.environ.get(var):
            info[var.lower()] = os.environ[var]
    return info
