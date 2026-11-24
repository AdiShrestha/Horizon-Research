"""First job on the cluster (A18): verify the data and code before anything trains.

Checks: every array listed in DATA_MANIFEST.json exists with the recorded SHA-256, shape and dtype
and is finite; every <split>_settings.json hash agrees; the code tree matches CODE_VERSION.json
(file hashes); the core and pipeline tests pass under this interpreter. Writes --out (JSON) and
exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TESTS = ["tests/test_core.py", "tests/test_extended.py", "tests/test_tiers.py",
         "tests/test_stage1.py", "tests/test_checkpoint.py", "tests/test_data.py"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    report: dict[str, object] = {"data": [], "code": None, "tests": None}
    ok = True
    manifest = json.loads((args.data_root / "DATA_MANIFEST.json").read_text())
    for entry in manifest["arrays"]:
        path = args.data_root / entry["path"]
        row = {"path": entry["path"], "exists": path.exists()}
        if path.exists():
            row["sha256_ok"] = sha256(path) == entry["sha256"]
            arr = np.load(path, mmap_mode="r")
            row["shape_ok"] = list(arr.shape) == entry["shape"] and str(arr.dtype) == entry["dtype"]
            row["finite"] = bool(np.isfinite(arr).all())
            settings = json.loads(path.with_name(path.stem + "_settings.json").read_text())
            row["settings_hash_ok"] = settings["sha256"] == entry["sha256"]
        row["pass"] = all(v for k, v in row.items() if k not in ("path",))
        ok &= row["pass"]
        report["data"].append(row)
        print(json.dumps(row), flush=True)
    code_manifest = ROOT / "CODE_VERSION.json"
    if code_manifest.exists():
        code = json.loads(code_manifest.read_text())
        bad = [p for p, h in code["files"].items() if not (ROOT / p).exists() or sha256(ROOT / p) != h]
        report["code"] = {"commit": code["commit"], "files": len(code["files"]), "mismatched": bad}
        ok &= not bad
    if not args.skip_tests:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", *TESTS], cwd=ROOT,
                              capture_output=True, text=True,
                              env={"PYTHONPATH": str(ROOT / "src"), "JAX_PLATFORMS": "cpu",
                                   "JAX_ENABLE_X64": "1", "PATH": "/usr/bin:/bin"})
        report["tests"] = {"exit": proc.returncode, "tail": proc.stdout.strip().splitlines()[-3:]}
        ok &= proc.returncode == 0
        print(proc.stdout[-2000:], flush=True)
    report["pass"] = bool(ok)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    if not ok:
        raise SystemExit("bundle verification failed")


if __name__ == "__main__":
    main()
