"""Reject single-precision and accelerator code inside the conformal core.

All conformal arithmetic runs in float64 on the CPU. This check parses every Python file
it is given (directories are searched recursively, defaulting to ``src/trajcp/conformal``)
and reports

* any use of ``float32``, ``float16``, ``bfloat16`` or ``complex64`` (names, attributes and
  dtype strings such as ``"float32"``),
* any ``.cuda(`` call,
* any use of ``torch.compile`` or of an ``autocast`` context.

Docstrings and comments are not inspected, so explanations of the rule stay legal. A file
that cannot be parsed counts as a violation and a path that does not exist is an error:
the check fails closed.

Usage: ``python scripts/check_conformal_dtypes.py [PATH ...]``; exit status 0 when clean,
1 on violations, 2 on a missing path.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

DEFAULT_TARGET = Path("src/trajcp/conformal")

# Names that may not appear as identifiers or attributes (np.float32, torch.bfloat16, ...).
FORBIDDEN_NAMES = frozenset({"float32", "float16", "bfloat16", "complex64"})
# NumPy aliases of the same types; only meaningful as attributes (np.single, np.half, ...).
FORBIDDEN_ALIASES = frozenset({"single", "half", "csingle"})
# Complete string constants that name or encode a forbidden dtype.
FORBIDDEN_STRINGS = frozenset(
    {"float32", "float16", "bfloat16", "complex64", "f4", "<f4", ">f4", "=f4", "f2", "c8"}
)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Identity of every constant node that is a module, class or function docstring."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def violations_in_source(source: str, filename: str = "<string>") -> list[str]:
    """Return ``file:line:col: message`` strings for every violation in ``source``."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [f"{filename}:{exc.lineno or 0}:{exc.offset or 0}: cannot parse file ({exc.msg})"]

    docstrings = _docstring_nodes(tree)
    torch_names = {"torch"}
    array_names = {"np", "numpy", "jnp", "torch"}  # roots under which dtype aliases are checked
    found: list[tuple[int, int, str]] = []

    def add(node: ast.AST, message: str) -> None:
        found.append((getattr(node, "lineno", 0), getattr(node, "col_offset", 0), message))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                bound = alias.asname or top
                if top == "torch":
                    torch_names.add(bound)
                if top in {"torch", "numpy", "jax"}:
                    array_names.add(bound)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "torch" or module.startswith("torch."):
                for alias in node.names:
                    if alias.name in {"compile", "autocast"}:
                        add(node, f"import of torch.{alias.name} is not allowed")
                    if alias.name in FORBIDDEN_NAMES:
                        add(node, f"import of {alias.name} is not allowed")
            elif module in {"numpy", "jax.numpy"}:
                for alias in node.names:
                    if alias.name in FORBIDDEN_NAMES | FORBIDDEN_ALIASES:
                        add(node, f"import of {alias.name} is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                add(node, f"{node.id} is not allowed (conformal arithmetic is float64)")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_NAMES or (
                node.attr in FORBIDDEN_ALIASES and _root_name(node) in array_names
            ):
                add(node, f".{node.attr} is not allowed (conformal arithmetic is float64)")
            elif node.attr == "autocast":
                add(node, "autocast is not allowed")
            elif node.attr == "compile" and _root_name(node) in torch_names:
                add(node, "torch.compile is not allowed")
            elif node.attr == "cuda":
                add(node, ".cuda is not allowed (conformal arithmetic runs on the CPU)")
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and node.value.strip().lower() in FORBIDDEN_STRINGS
        ):
            add(node, f"dtype string {node.value!r} is not allowed (float64 only)")

    return [f"{filename}:{line}:{col}: {message}" for line, col, message in sorted(found)]


def python_files(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("*.py"))
        else:
            yield path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", type=Path, help="files or directories to check")
    args = parser.parse_args(argv)
    targets = args.paths or [DEFAULT_TARGET]

    missing = [str(p) for p in targets if not p.exists()]
    if missing:
        print("path not found: " + ", ".join(missing), file=sys.stderr)
        return 2

    problems: list[str] = []
    checked = 0
    for file in python_files(targets):
        checked += 1
        problems.extend(violations_in_source(file.read_text(encoding="utf-8"), str(file)))

    for line in problems:
        print(line)
    if problems:
        print(f"{len(problems)} violation(s) in {checked} file(s)", file=sys.stderr)
        return 1
    print(f"conformal dtype guard: {checked} file(s) clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
