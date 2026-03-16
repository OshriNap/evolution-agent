"""Tests for code sandbox."""

from evolution_agent.evaluation.sandbox import CodeSandbox, lint_code, check_function_exists


def test_lint_clean_code():
    errors = lint_code("def f(x):\n    return x + 1")
    assert errors == []


def test_lint_syntax_error():
    errors = lint_code("def f(x:\n    return x")
    assert len(errors) == 1
    assert "Syntax error" in errors[0]


def test_lint_import_forbidden():
    errors = lint_code("import os")
    assert any("Import" in e for e in errors)


def test_lint_forbidden_call():
    errors = lint_code("x = eval('1+1')")
    assert any("eval" in e for e in errors)


def test_lint_forbidden_module_ref():
    errors = lint_code("x = subprocess.run([])")
    assert any("subprocess" in e for e in errors)


def test_lint_dunder_forbidden():
    errors = lint_code("x.__class__")
    assert any("__class__" in e for e in errors)


def test_check_function_exists_found():
    errors = check_function_exists("def sort_list(arr):\n    return sorted(arr)", "sort_list")
    assert errors == []


def test_check_function_exists_missing():
    errors = check_function_exists("def other(x):\n    pass", "sort_list")
    assert any("sort_list" in e for e in errors)


def test_sandbox_validate():
    s = CodeSandbox()
    assert s.validate("def f(): return 1") == []
    assert s.validate("def f(): return 1", "f") == []
    assert len(s.validate("import os")) > 0
    assert len(s.validate("def g(): pass", "f")) > 0


def test_sandbox_compile_function():
    s = CodeSandbox()
    fn = s.compile_function("def f(x):\n    return x * 2", "f")
    assert fn is not None
    assert fn(5) == 10


def test_sandbox_compile_unsafe():
    s = CodeSandbox()
    fn = s.compile_function("import os\ndef f(): return os.getcwd()", "f")
    assert fn is None


def test_sandbox_compile_missing_function():
    s = CodeSandbox()
    fn = s.compile_function("def g(): return 1", "f")
    assert fn is None


def test_sandbox_compile_with_math():
    s = CodeSandbox()
    fn = s.compile_function("def f(x):\n    return math.sqrt(x)", "f")
    assert fn is not None
    assert fn(4) == 2.0


def test_sandbox_execute_code():
    s = CodeSandbox()
    ns = s.execute_code("x = 42\ny = x * 2")
    assert ns["x"] == 42
    assert ns["y"] == 84
