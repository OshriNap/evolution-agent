"""Parameter tuner: extract tunable params from evolved code, optimize with Optuna.

The evolved function uses a `p` dict for tunable numeric constants:

    def solve(input, p=None):
        if p is None:
            p = {"alpha": 1.0, "beta": 0.5, "max_iter": 100}
        ...

The tuner:
1. Parses the code AST to extract parameter names, types, and defaults
2. Defines search ranges from defaults (e.g. default * 0.1 to default * 10)
3. Runs Optuna trials to find optimal values
4. Returns best params + best fitness

Supports TPE (Bayesian optimization), CMA-ES, and random samplers.
"""

from __future__ import annotations

import ast
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class ParamSpec:
    """One tunable parameter extracted from code."""

    name: str
    default: float
    low: float
    high: float
    is_int: bool = False
    log_scale: bool = False


@dataclass
class TuningResult:
    """Result of parameter tuning for one code variant."""

    best_fitness: float
    best_params: dict[str, float]
    n_trials: int
    default_fitness: float  # fitness with default params
    improvement: float  # best - default
    all_trials: list[dict[str, Any]] = field(default_factory=list)


def extract_params(code: str) -> list[ParamSpec]:
    """Extract tunable parameters from `p = {...}` default dict in code.

    Looks for patterns like:
        p = {"alpha": 1.0, "beta": 0.5}
    or:
        if p is None:
            p = {"alpha": 1.0}
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    params: list[ParamSpec] = []

    for node in ast.walk(tree):
        # Look for: p = {"key": value, ...}
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "p":
                    if isinstance(node.value, ast.Dict):
                        params = _extract_from_dict(node.value)
                        if params:
                            return params

    return params


def _extract_from_dict(node: ast.Dict) -> list[ParamSpec]:
    """Extract ParamSpecs from a Dict AST node."""
    params: list[ParamSpec] = []
    for key, value in zip(node.keys, node.values):
        if key is None:
            continue
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            continue
        name = key.value

        if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            default = float(value.value)
            is_int = isinstance(value.value, int)
            low, high = _default_range(default, is_int)
            params.append(ParamSpec(
                name=name,
                default=default,
                low=low,
                high=high,
                is_int=is_int,
                log_scale=default > 0 and high / max(low, 1e-10) > 100,
            ))
        elif isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
            # Handle negative numbers: -1.0
            if isinstance(value.operand, ast.Constant) and isinstance(value.operand.value, (int, float)):
                default = -float(value.operand.value)
                is_int = isinstance(value.operand.value, int)
                low, high = _default_range(default, is_int)
                params.append(ParamSpec(
                    name=name, default=default, low=low, high=high, is_int=is_int,
                ))
    return params


def _default_range(default: float, is_int: bool) -> tuple[float, float]:
    """Compute search range from a default value."""
    if default == 0:
        return (-1.0, 1.0) if not is_int else (-10, 10)

    abs_val = abs(default)

    if is_int:
        # For integers, search +/- 5x or at least +/- 5
        spread = max(int(abs_val * 5), 5)
        return (default - spread, default + spread)

    # For floats, search 0.1x to 10x (same sign)
    if default > 0:
        return (default * 0.1, default * 10.0)
    else:
        return (default * 10.0, default * 0.1)  # reversed because negative


def tune_parameters(
    code: str,
    compile_fn: Callable[[str, str], Any],
    fitness_fn: Callable[..., tuple[float, dict[str, float]]],
    function_name: str,
    n_trials: int = 30,
    sampler: str = "tpe",  # "tpe" (BO), "cmaes", "random"
    direction: str = "maximize",
    param_specs: list[ParamSpec] | None = None,
    timeout_s: float = 60.0,
) -> TuningResult | None:
    """Tune parameters in evolved code using Optuna.

    Args:
        code: The evolved code string containing `p = {...}`
        compile_fn: Function to compile code → callable (sandbox.compile_function)
        fitness_fn: Fitness function that takes (compiled_fn,) → (fitness, metrics)
        function_name: Name of the function to extract
        n_trials: Number of Optuna trials
        sampler: "tpe" (Bayesian), "cmaes", or "random"
        direction: "maximize" or "minimize"
        param_specs: Override extracted params (optional)
        timeout_s: Total timeout for tuning

    Returns:
        TuningResult or None if code has no tunable params or compilation fails
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not installed, skipping parameter tuning")
        return None

    # Extract params if not provided
    specs = param_specs or extract_params(code)
    if not specs:
        return None

    # Compile the base function to verify it works
    base_fn = compile_fn(code, function_name)
    if base_fn is None:
        return None

    # Evaluate with defaults first
    try:
        default_fitness, _ = fitness_fn(base_fn)
    except Exception as e:
        logger.debug("Default params evaluation failed: %s", e)
        return None

    # Build code template that accepts params
    # The function already uses `p` dict, so we inject params before calling
    def _make_fn_with_params(trial_params: dict[str, float]):
        """Create a wrapper that injects params into the function call."""
        fn = compile_fn(code, function_name)
        if fn is None:
            return None

        import functools
        import inspect

        # Check if function accepts `p` argument
        sig = inspect.signature(fn)
        if "p" in sig.parameters:
            return functools.partial(fn, p=trial_params)
        else:
            # Function uses p internally with defaults, need to recompile
            # with modified defaults
            modified_code = _inject_params(code, trial_params)
            return compile_fn(modified_code, function_name)

    # Create Optuna study
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if sampler == "cmaes":
        s = CmaEsSampler(seed=42)
    elif sampler == "random":
        s = RandomSampler(seed=42)
    else:
        s = TPESampler(n_startup_trials=min(5, n_trials // 3), seed=42)

    study = optuna.create_study(
        direction=direction,
        sampler=s,
    )

    all_trials: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, float] = {}
        for spec in specs:
            if spec.is_int:
                params[spec.name] = trial.suggest_int(
                    spec.name, int(spec.low), int(spec.high),
                )
            elif spec.log_scale and spec.low > 0:
                params[spec.name] = trial.suggest_float(
                    spec.name, spec.low, spec.high, log=True,
                )
            else:
                params[spec.name] = trial.suggest_float(
                    spec.name, spec.low, spec.high,
                )

        fn = _make_fn_with_params(params)
        if fn is None:
            return float("-inf") if direction == "maximize" else float("inf")

        try:
            fitness, metrics = fitness_fn(fn)
            all_trials.append({"params": params, "fitness": fitness})
            return fitness
        except Exception:
            return float("-inf") if direction == "maximize" else float("inf")

    study.optimize(objective, n_trials=n_trials, timeout=timeout_s)

    try:
        best = study.best_trial
        return TuningResult(
            best_fitness=best.value,
            best_params=best.params,
            n_trials=len(study.trials),
            default_fitness=default_fitness,
            improvement=best.value - default_fitness,
            all_trials=all_trials,
        )
    except ValueError:
        return TuningResult(
            best_fitness=default_fitness,
            best_params={s.name: s.default for s in specs},
            n_trials=0,
            default_fitness=default_fitness,
            improvement=0.0,
        )


def _inject_params(code: str, params: dict[str, float]) -> str:
    """Replace default parameter values in code with tuned values.

    Finds `p = {"key": default, ...}` and replaces with tuned values.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    # Find the p = {...} assignment and rebuild it
    lines = code.split("\n")

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "p":
                    if isinstance(node.value, ast.Dict):
                        # Build new dict string
                        items = []
                        for key, val in zip(node.value.keys, node.value.values):
                            if key and isinstance(key, ast.Constant):
                                name = key.value
                                if name in params:
                                    v = params[name]
                                    items.append(f'"{name}": {v}')
                                else:
                                    # Keep original
                                    items.append(
                                        f'"{name}": '
                                        + ast.get_source_segment(code, val)
                                    )

                        new_dict = "{" + ", ".join(items) + "}"

                        # Get indentation
                        line_idx = node.lineno - 1
                        original_line = lines[line_idx]
                        indent = original_line[: len(original_line) - len(original_line.lstrip())]

                        # Handle multi-line dict assignments
                        end_line = node.end_lineno
                        lines[line_idx:end_line] = [f"{indent}p = {new_dict}"]

                        return "\n".join(lines)

    return code
