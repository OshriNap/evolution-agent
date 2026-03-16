"""Tests for parameter extraction and tuning."""

import pytest

from evolution_agent.evaluation.parameter_tuner import (
    ParamSpec,
    extract_params,
    _inject_params,
)

try:
    from evolution_agent.evaluation.parameter_tuner import tune_parameters, OPTUNA_AVAILABLE
except ImportError:
    OPTUNA_AVAILABLE = False

from evolution_agent.evaluation.sandbox import CodeSandbox


def test_extract_simple_params():
    code = '''
def solve(x, p=None):
    if p is None:
        p = {"alpha": 1.0, "beta": 0.5, "max_iter": 100}
    return x * p["alpha"]
'''
    params = extract_params(code)
    assert len(params) == 3
    names = {p.name for p in params}
    assert names == {"alpha", "beta", "max_iter"}

    alpha = next(p for p in params if p.name == "alpha")
    assert alpha.default == 1.0
    assert alpha.is_int is False

    max_iter = next(p for p in params if p.name == "max_iter")
    assert max_iter.default == 100.0
    assert max_iter.is_int is True


def test_extract_no_params():
    code = "def solve(x):\n    return x * 2"
    params = extract_params(code)
    assert params == []


def test_extract_negative_params():
    code = '''
def f(x, p=None):
    if p is None:
        p = {"offset": -5.0}
    return x + p["offset"]
'''
    params = extract_params(code)
    assert len(params) == 1
    assert params[0].default == -5.0


def test_extract_zero_param():
    code = '''
def f(x, p=None):
    if p is None:
        p = {"bias": 0.0}
    return x + p["bias"]
'''
    params = extract_params(code)
    assert len(params) == 1
    assert params[0].low < 0 and params[0].high > 0


def test_inject_params():
    code = '''def f(x, p=None):
    if p is None:
        p = {"alpha": 1.0, "beta": 0.5}
    return x * p["alpha"] + p["beta"]'''

    modified = _inject_params(code, {"alpha": 2.5, "beta": 0.8})
    assert "2.5" in modified
    assert "0.8" in modified


def test_default_range_positive():
    spec = ParamSpec(name="x", default=1.0, low=0.1, high=10.0)
    assert spec.low == 0.1
    assert spec.high == 10.0


def test_default_range_int():
    params = extract_params('''
def f(p=None):
    if p is None:
        p = {"n": 10}
''')
    assert len(params) == 1
    p = params[0]
    assert p.is_int
    assert p.low < 10 and p.high > 10


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_tune_simple_function():
    """Tune a simple quadratic: maximize -(x-3)^2 + 10."""
    code = '''
def f(x, p=None):
    if p is None:
        p = {"a": 1.0}
    return -(x - p["a"]) ** 2 + 10
'''
    sandbox = CodeSandbox()

    def fitness_fn(fn):
        # Evaluate at x=3 — optimal when p["a"] == 3
        result = fn(3.0)
        return result, {"value": result}

    result = tune_parameters(
        code=code,
        compile_fn=sandbox.compile_function,
        fitness_fn=fitness_fn,
        function_name="f",
        n_trials=20,
        sampler="tpe",
        direction="maximize",
    )

    assert result is not None
    assert result.best_fitness > result.default_fitness
    # Should find a ≈ 3.0
    assert abs(result.best_params["a"] - 3.0) < 1.0


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_tune_with_cmaes():
    code = '''
def f(x, p=None):
    if p is None:
        p = {"a": 0.0, "b": 0.0}
    return -((x - p["a"]) ** 2 + (x - p["b"]) ** 2)
'''
    sandbox = CodeSandbox()

    def fitness_fn(fn):
        result = fn(2.0)
        return result, {}

    result = tune_parameters(
        code=code,
        compile_fn=sandbox.compile_function,
        fitness_fn=fitness_fn,
        function_name="f",
        n_trials=30,
        sampler="cmaes",
        direction="maximize",
    )

    assert result is not None
    assert result.n_trials > 0
