"""Reject the conformal CPU float64 package's disallowed syntax in Python files."""

import argparse
import ast
import re
import sys
import tokenize
from pathlib import Path

FLOAT32_DTYPE = re.compile(r"(?:\w+\.)*float32", re.IGNORECASE)


def imported_names(nodes):
    """Resolve direct namespace aliases and renamed function imports."""
    names = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                names[alias.asname or root] = alias.name if alias.asname else root
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return names


def qualified_name(expression, imports):
    if isinstance(expression, ast.Name):
        return imports.get(expression.id, expression.id)
    if isinstance(expression, ast.Attribute):
        base = qualified_name(expression.value, imports)
        return f"{base}.{expression.attr}" if base else ""
    return ""


def is_mps(expression):
    return isinstance(expression, ast.Constant) and isinstance(expression.value, str) and (
        expression.value.split(":")[0].lower() == "mps"
    )


def violations(tree):
    nodes = list(ast.walk(tree))
    imports = imported_names(nodes)
    # Standalone string expressions include docstrings and carry no dtype.
    standalone_strings = {
        node.value for node in nodes if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }
    for node in nodes:
        dtype = None
        if isinstance(node, ast.Name):
            dtype = node.id
        elif isinstance(node, ast.Attribute):
            dtype = node.attr
        elif isinstance(node, ast.alias):
            dtype = node.name
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node not in standalone_strings:
            dtype = node.value
        if dtype and FLOAT32_DTYPE.fullmatch(dtype):
            yield node.lineno, node.col_offset + 1, "float32 is forbidden in the conformal package"

        if isinstance(node, ast.Call):
            name = qualified_name(node.func, imports)
            cuda_method = isinstance(node.func, ast.Attribute) and node.func.attr == "cuda"
            if cuda_method or name == "torch.cuda":
                yield node.lineno, node.col_offset + 1, "CUDA calls are forbidden in the conformal package"
            elif name == "torch.compile":
                yield node.lineno, node.col_offset + 1, "torch.compile is forbidden in the conformal package"
            elif name in {"torch.autocast", "torch.amp.autocast"}:
                mps_device = (node.args and is_mps(node.args[0])) or any(
                    kw.arg == "device_type" and is_mps(kw.value) for kw in node.keywords
                )
                if mps_device:
                    yield node.lineno, node.col_offset + 1, "MPS autocast is forbidden in the conformal package"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    failed = False
    for path in args.paths:
        try:
            with tokenize.open(path) as stream:
                tree = ast.parse(stream.read(), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as error:
            print(f"{path}: cannot check Python source: {error}", file=sys.stderr)
            failed = True
            continue
        for line, column, message in sorted(set(violations(tree))):
            print(f"{path}:{line}:{column}: {message}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
