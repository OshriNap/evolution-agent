"""Code sandbox: AST validation and safe execution.

Ported from reference code_compiler.py, adapted for generic evolution.
"""

from __future__ import annotations

import ast
import logging
import math
from collections import Counter, defaultdict, deque
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Forbidden AST names (modules and builtins)
_FORBIDDEN_NAMES = frozenset({
    "os", "sys", "subprocess", "socket", "http", "urllib",
    "pathlib", "shutil", "ctypes", "signal", "threading",
    "multiprocessing", "pickle", "shelve", "tempfile",
    "importlib", "runpy", "code", "codeop",
})

_FORBIDDEN_CALLS = frozenset({
    "exec", "eval", "compile", "open", "getattr", "setattr",
    "delattr", "globals", "locals", "vars", "dir",
    "breakpoint", "exit", "quit", "input",
    "__import__",
})

# Safe builtins for the exec namespace
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "max": max,
    "min": min,
    "sum": sum,
    "len": len,
    "sorted": sorted,
    "enumerate": enumerate,
    "range": range,
    "zip": zip,
    "round": round,
    "pow": pow,
    "float": float,
    "int": int,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "True": True,
    "False": False,
    "None": None,
    "isinstance": isinstance,
    "hasattr": hasattr,
    "any": any,
    "all": all,
    "map": map,
    "filter": filter,
    "reversed": reversed,
    "ord": ord,
    "chr": chr,
    "hash": hash,
    "divmod": divmod,
    "print": lambda *a, **kw: None,  # silenced print
}


class _ASTLinter(ast.NodeVisitor):
    """Walk AST to check for forbidden constructs."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        names = ", ".join(a.name for a in node.names)
        self.errors.append(f"Import not allowed: import {names}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.errors.append(f"Import not allowed: from {node.module} import ...")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.errors.append(f"Dunder access not allowed: .{node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _FORBIDDEN_NAMES:
            self.errors.append(f"Forbidden module reference: {node.id}")
        if node.id in _FORBIDDEN_CALLS:
            self.errors.append(f"Forbidden builtin: {node.id}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
            self.errors.append(f"Forbidden call: {node.func.id}()")
        if isinstance(node.func, ast.Attribute) and node.func.attr in _FORBIDDEN_CALLS:
            self.errors.append(f"Forbidden call: .{node.func.attr}()")
        self.generic_visit(node)


def lint_code(code: str) -> list[str]:
    """Parse and lint a code string. Returns list of error messages."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error at line {e.lineno}: {e.msg}"]

    linter = _ASTLinter()
    linter.visit(tree)
    return linter.errors


def check_function_exists(code: str, function_name: str) -> list[str]:
    """Check that the code defines a function with the given name."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [f"Cannot parse code to find function '{function_name}'"]

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return []

    return [f"Function '{function_name}' not found in code"]


class CodeSandbox:
    """Compile and execute LLM-generated Python code safely."""

    def __init__(self, extra_builtins: dict[str, Any] | None = None) -> None:
        self._extra_builtins = extra_builtins or {}

    def validate(self, code: str, function_name: str | None = None) -> list[str]:
        """Validate code via AST linting. Returns errors (empty = ok)."""
        errors = lint_code(code)
        if function_name and not errors:
            errors.extend(check_function_exists(code, function_name))
        return errors

    def compile_function(
        self, code: str, function_name: str,
    ) -> Callable | None:
        """Compile code and extract a named function.

        Returns None if code is unsafe or function not found.
        """
        errors = self.validate(code, function_name)
        if errors:
            for err in errors:
                logger.warning("Sandbox lint error: %s", err)
            return None

        namespace: dict[str, Any] = {
            "__builtins__": {**_SAFE_BUILTINS, **self._extra_builtins},
            "math": math,
            "deque": deque,
            "Counter": Counter,
            "defaultdict": defaultdict,
        }

        try:
            exec(code, namespace)  # noqa: S102
        except Exception as e:
            logger.warning("Code exec failed: %s", e)
            return None

        fn = namespace.get(function_name)
        if not callable(fn):
            logger.warning("Code did not define callable '%s'", function_name)
            return None

        return fn

    def execute_code(self, code: str) -> dict[str, Any]:
        """Execute code and return the resulting namespace (minus builtins)."""
        errors = lint_code(code)
        if errors:
            raise ValueError(f"Code validation failed: {errors}")

        namespace: dict[str, Any] = {
            "__builtins__": {**_SAFE_BUILTINS, **self._extra_builtins},
            "math": math,
            "deque": deque,
            "Counter": Counter,
            "defaultdict": defaultdict,
        }

        exec(code, namespace)  # noqa: S102
        return {k: v for k, v in namespace.items() if k != "__builtins__"}
