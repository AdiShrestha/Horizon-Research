"""The conformal-core guard rejects single precision and accelerator code, and nothing else."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts" / "check_conformal_dtypes.py"


def _load_guard() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_conformal_dtypes", GUARD)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


guard = _load_guard()

CLEAN = [
    "import numpy as np\nx = np.zeros(3, dtype=np.float64)\ny = x.astype('float64')\n",
    "import math\nk = math.ceil(round(10 * 0.9, 12))\n",
    '"""Never use float32, .cuda( or torch.compile( here."""\nvalue = 1.0  # not float32\n',
    "def f(half: float) -> float:\n    return 2.0 * half\n",
    "class A:\n    half = 1.0\n\nA().half\n",
    "message = 'expected float64 input but received a different dtype'\n",
]

VIOLATIONS = [
    ("import numpy as np\nx = np.float32(1.0)\n", "float32"),
    ("import numpy as np\nx = np.zeros(3, dtype='float32')\n", "float32"),
    ("import numpy as np\nx = np.zeros(3, dtype=np.dtype('<f4'))\n", "<f4"),
    ("import numpy as np\nx = np.zeros(3).astype(np.float16)\n", "float16"),
    ("import numpy as np\nx = np.zeros(3, dtype=np.complex64)\n", "complex64"),
    ("import numpy as np\nx = np.single(1.0)\n", "single"),
    ("import numpy as np\nx = np.half(1.0)\n", "half"),
    ("import torch\nx = torch.zeros(2, dtype=torch.float32)\n", "float32"),
    ("import torch\nx = torch.zeros(2, dtype=torch.bfloat16)\n", "bfloat16"),
    ("from numpy import float32\n", "float32"),
    ("from torch import bfloat16\n", "bfloat16"),
    ("import torch\nx = torch.zeros(2).cuda()\n", "cuda"),
    ("x = tensor.cuda()\n", "cuda"),
    ("import torch\nf = torch.compile(g)\n", "torch.compile"),
    ("import torch\n\n@torch.compile\ndef f(x):\n    return x\n", "torch.compile"),
    ("import torch as th\nf = th.compile(g)\n", "torch.compile"),
    ("from torch import compile\n", "compile"),
    ("import torch\nwith torch.autocast('mps'):\n    pass\n", "autocast"),
    ("import torch\nwith torch.amp.autocast('cpu'):\n    pass\n", "autocast"),
    ("from torch.amp import autocast\n", "autocast"),
]


def test_known_clean_snippets_pass() -> None:
    for snippet in CLEAN:
        assert guard.violations_in_source(snippet, "snippet.py") == [], snippet


@pytest.mark.parametrize(("snippet", "token"), VIOLATIONS)
def test_known_violations_are_reported(snippet: str, token: str) -> None:
    found = guard.violations_in_source(snippet, "snippet.py")
    assert found, snippet
    assert any(token in line for line in found), (token, found)
    assert all(line.startswith("snippet.py:") for line in found)


def test_unparsable_file_is_a_violation() -> None:
    found = guard.violations_in_source("def broken(:\n", "bad.py")
    assert found and "cannot parse" in found[0]


def test_violations_report_line_numbers() -> None:
    found = guard.violations_in_source("import numpy as np\n\n\nx = np.float32(2.0)\n", "f.py")
    assert found[0].startswith("f.py:4:")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(GUARD), *args], capture_output=True, text=True, cwd=ROOT
    )


def test_command_line_exit_codes(tmp_path: Path) -> None:
    clean = tmp_path / "clean.py"
    clean.write_text(CLEAN[0], encoding="utf-8")
    dirty = tmp_path / "dirty.py"
    dirty.write_text(VIOLATIONS[0][0], encoding="utf-8")

    assert _run(str(clean)).returncode == 0
    assert _run(str(dirty)).returncode == 1
    assert _run(str(tmp_path)).returncode == 1  # directories are searched recursively
    assert _run(str(tmp_path / "missing.py")).returncode == 2  # fails closed


def test_the_conformal_package_is_clean() -> None:
    result = _run(str(ROOT / "src" / "trajcp" / "conformal"))
    assert result.returncode == 0, result.stdout + result.stderr
