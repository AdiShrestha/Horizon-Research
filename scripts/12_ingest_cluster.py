"""Ingest a cluster results tar on the laptop (A18): extract, then verify every file against the
SHA-256 manifest packed inside it. Never overwrites an existing destination.

    .venv-run/bin/python scripts/12_ingest_cluster.py results-core-<stamp>.tar --dest artifacts/cluster/<run>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tar", type=Path)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.tar) as tar:
        names = tar.getnames()
        clash = [n for n in names if (args.dest / n).exists()]
        if clash:
            raise SystemExit(f"{len(clash)} files already exist in {args.dest}, e.g. {clash[0]}")
        tar.extractall(args.dest, filter="data")
    manifests = sorted(args.dest.glob("RESULTS_MANIFEST_*.sha256"))
    bad, checked = [], 0
    for manifest in manifests:
        for line in manifest.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            checked += 1
            if sha256(args.dest / name) != digest:
                bad.append(name)
    report = {"tar": str(args.tar), "files_checked": checked, "mismatched": bad, "pass": not bad}
    (args.dest / f"INGEST_{args.tar.stem}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if bad:
        raise SystemExit("hash mismatch")


if __name__ == "__main__":
    main()
