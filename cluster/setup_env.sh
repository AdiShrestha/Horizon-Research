#!/usr/bin/env bash
# Build the cluster runtime once, on a login node with internet access (A18).
# Usage: bash cluster/setup_env.sh [python executable, default python3.12]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${1:-python3.12}"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || {
  echo "Python 3.12 is required (try: module load python/3.12, or pass the interpreter path)"; exit 1; }
"$PY" -m venv "$HERE/.venv-cluster"
"$HERE/.venv-cluster/bin/python" -m pip install --upgrade pip
"$HERE/.venv-cluster/bin/python" -m pip install -r "$HERE/cluster/requirements-cluster.txt"
"$HERE/.venv-cluster/bin/python" -m pip freeze > "$HERE/cluster/pip-freeze-cluster.txt"
"$HERE/.venv-cluster/bin/python" - <<'PY'
import torch, jax, numpy
print("torch", torch.__version__, "cuda build", torch.version.cuda, "cuda available here:", torch.cuda.is_available())
print("jax", jax.__version__, "numpy", numpy.__version__)
PY
echo "Environment ready: $HERE/.venv-cluster (freeze in cluster/pip-freeze-cluster.txt)"
