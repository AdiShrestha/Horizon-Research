"""Build the A18 cluster bundle from a clean commit (run on the laptop).

    .venv-run/bin/python cluster/build_bundle.py --dest ../cluster_bundle

Creates <dest>/horizon-a18-<commit>/ containing ``source/`` (git archive of HEAD plus
CODE_VERSION.json with per-file SHA-256) and ``source/data/<task>/`` (TRAIN and DEV arrays, metadata,
settings and data cards for P2, P1 and P3, with DATA_MANIFEST.json), then a single uncompressed tar
next to it and its SHA-256. Refuses to run with uncommitted changes in src/, scripts/, cluster/ or
tests/, and checks every array against the hash in its settings file.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TASKS = ("p2", "p1", "p3")
SPLITS = ("train", "dev")
EXTRA = ("DATA_CARD.md", "datacard_stats.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "artifacts" / "data")
    parser.add_argument("--no-tar", action="store_true")
    args = parser.parse_args()
    if git("status", "--porcelain", "--untracked-files=no", "src", "scripts", "cluster", "tests").strip():
        raise SystemExit("commit the code first: the bundle is built from HEAD")
    commit = git("rev-parse", "HEAD").strip()
    name = f"horizon-a18-{commit[:7]}"
    out = (args.dest / name).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists")
    src = out / "source"
    src.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT,
                             capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(src, filter="data")
    files = {
        str(p.relative_to(src)): sha256(p)
        for p in sorted(src.rglob("*"))
        if p.is_file() and p.suffix in {".py", ".txt", ".json", ".sh", ".toml", ".lock", ".md"}
        and not str(p.relative_to(src)).startswith("artifacts/")
    }
    (src / "CODE_VERSION.json").write_text(json.dumps(
        {"commit": commit, "dirty": False, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
         "files": files}, indent=2))
    data_out = src / "data"
    arrays = []
    for task in TASKS:
        (data_out / task).mkdir(parents=True)
        for split in SPLITS:
            settings = json.loads((args.data_root / task / f"{split}_settings.json").read_text())
            npy = args.data_root / task / f"{split}.npy"
            digest = sha256(npy)
            if digest != settings["sha256"]:
                raise SystemExit(f"{npy}: hash differs from its settings file")
            for f in (f"{split}.npy", f"{split}_meta.parquet", f"{split}_settings.json"):
                shutil.copy2(args.data_root / task / f, data_out / task / f)
            arr = np.load(npy, mmap_mode="r")
            arrays.append({"path": f"{task}/{split}.npy", "sha256": digest, "shape": list(arr.shape),
                           "dtype": str(arr.dtype), "stream": settings["stream"], "n": settings["n"]})
            print(f"{task}/{split}: {arr.shape} {digest[:12]}", flush=True)
        for f in EXTRA:
            if (args.data_root / task / f).exists():
                shutil.copy2(args.data_root / task / f, data_out / task / f)
    (data_out / "DATA_MANIFEST.json").write_text(json.dumps(
        {"commit": commit, "arrays": arrays, "note": "TRAIN and DEV streams only; CONFIRM is never shipped"},
        indent=2))
    if not args.no_tar:
        tar_path = out.with_suffix(".tar")
        with tarfile.open(tar_path, "w") as tar:
            tar.add(out, arcname=name)
        digest = sha256(tar_path)
        tar_path.with_suffix(".tar.sha256").write_text(f"{digest}  {tar_path.name}\n")
        print(f"{tar_path} {tar_path.stat().st_size / 2**30:.2f} GiB sha256 {digest}")
    print(f"bundle: {out}")


if __name__ == "__main__":
    main()
