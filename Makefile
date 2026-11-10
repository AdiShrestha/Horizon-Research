# Developer shortcuts. Python never writes bytecode here, so source/ stays free of .pyc files.
export PYTHONDONTWRITEBYTECODE := 1
export PYTHONPATH := src$(if $(PYTHONPATH),:$(PYTHONPATH),)

UV     ?= uv
PYTHON ?= .venv/bin/python
PY     := $(PYTHON) -B
CODE   := src scripts tests

.DEFAULT_GOAL := help
.PHONY: help sync export-lock test test-nightly lint format dtype-guard precommit check-source-bytecode clean-source-bytecode

help:
	@echo "sync                    create or update .venv from uv.lock"
	@echo "export-lock             regenerate source/requirements.lock from uv.lock"
	@echo "test                    fast test suite (CPU, float64 core)"
	@echo "test-nightly            full suite: 10,000-split simulations, 2,000 property examples"
	@echo "lint                    ruff, mypy on the conformal core, dtype guard"
	@echo "format                  apply ruff formatting"
	@echo "precommit               run the pre-commit configuration on every file"
	@echo "clean-source-bytecode   delete __pycache__ and .pyc files under source/"

# macOS can flag files inside .venv as hidden, and Python skips hidden .pth files, which
# silently breaks the editable install; clear the flag after every sync.
sync:
	$(UV) sync --frozen --group oracles
	@if [ "$$(uname)" = "Darwin" ]; then chflags -R nohidden .venv; fi

# Hash-pinned dependency list of the registered comparison study under source/.
# Regenerate only after a dependency change, and before that study is frozen.
export-lock:
	$(UV) export --format requirements-txt --frozen --no-emit-project > source/requirements.lock

test: check-source-bytecode
	$(PY) -m pytest -m "not nightly"

test-nightly: check-source-bytecode
	HYPOTHESIS_PROFILE=nightly $(PY) -m pytest

lint: dtype-guard
	$(PY) -m ruff check $(CODE)
	$(PY) -m ruff format --check $(CODE)
	$(PY) -m mypy

format:
	$(PY) -m ruff check --fix $(CODE)
	$(PY) -m ruff format $(CODE)

# The conformal core is float64, CPU-only: no float32, .cuda(, torch.compile( or autocast.
dtype-guard:
	$(PY) scripts/check_conformal_dtypes.py src/trajcp/conformal

precommit:
	$(UV) tool run pre-commit run --all-files

check-source-bytecode:
	@if find source \( -name __pycache__ -o -name '*.pyc' -o -name '*.pyo' \) 2>/dev/null | grep -q . ; then \
		echo "bytecode found under source/; run: make clean-source-bytecode" >&2; exit 1; fi

clean-source-bytecode:
	@find source -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@find source -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete 2>/dev/null || true
