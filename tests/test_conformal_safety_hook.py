import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "scripts/check_conformal_safety.py"


def invoke(path):
    return subprocess.run([sys.executable, str(HOOK), str(path)], capture_output=True, text=True)


def test_allows_cpu_float64_and_documentation(tmp_path):
    path = tmp_path / "safe.py"
    path.write_text(
        '"""float32"""\n'
        '# torch.compile(model) and data.cuda() are excluded here.\n'
        'import numpy as np\n'
        'import torch\n'
        'from jaxtyping import Float64\n'
        'def apply(x: Float64[np.ndarray, "n"]):\n'
        '    with torch.autocast(device_type="cpu", enabled=False):\n'
        '        return np.asarray(x, dtype=np.float64)\n'
        'memo = "Use float64 instead of float32."\n'
    )
    result = invoke(path)
    assert result.returncode == 0
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("np.asarray(x, dtype=np.float32)", "float32"),
        ('np.asarray(x, dtype="float32")', "float32"),
        ('x: "np.float32"', "float32"),
        ("from numpy import float32 as single", "float32"),
        ("from jaxtyping import Float32", "float32"),
        ("array.cuda()", "CUDA calls"),
        ("tensor.cuda(\n    0\n)", "CUDA calls"),
        ("torch.compile(model)", "torch.compile"),
        ("import torch as th\nth.compile(model)", "torch.compile"),
        ("from torch import compile as optimize\noptimize(model)", "torch.compile"),
        ("torch.autocast('mps')", "MPS autocast"),
        ('torch.autocast(device_type="mps")', "MPS autocast"),
        ('import torch.amp as amp\namp.autocast("mps:0")', "MPS autocast"),
        ("from torch.amp import autocast as cast\ncast('mps')", "MPS autocast"),
        ("def incomplete(", "cannot check Python source"),
    ],
)
def test_rejects_unsafe_or_invalid_source(tmp_path, source, message):
    path = tmp_path / "fixture.py"
    path.write_text(source + "\n")
    result = invoke(path)
    assert result.returncode == 1
    assert str(path) in result.stderr
    assert message in result.stderr
