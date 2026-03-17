"""Parameter tuner: extract tunable params from evolved code and optimize.

The evolved function uses a `p` dict for tunable numeric constants:

    def solve(input, p=None):
        if p is None:
            p = {"alpha": 1.0, "beta": 0.5, "max_iter": 100}
        ...

The tuner:
1. Parses the code AST to extract parameter names, types, and defaults
2. Defines search ranges from defaults (e.g. default * 0.1 to default * 10)
3. Optimizes params via coordinate descent + golden-section search (fast)
   or Optuna BO/CMA-ES (when requested)
4. Returns best params + best fitness
"""

from __future__ import annotations

import ast
import functools
import inspect
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# Golden ratio for golden-section search
_PHI = (1 + math.sqrt(5)) / 2
_RESPHI = 2 - _PHI  # ~0.382


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


# ---------------------------------------------------------------------------
# Fast tuner: coordinate descent + golden-section search
# ---------------------------------------------------------------------------

def _golden_section_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    maximize: bool = True,
    tol: float = 1e-3,
    max_evals: int = 20,
    is_int: bool = False,
) -> tuple[float, float]:
    """Golden-section search for a 1D function.

    Returns (best_x, best_fitness).
    """
    evals = 0

    # Evaluate boundaries first
    fa = f(int(round(a)) if is_int else a)
    fb = f(int(round(b)) if is_int else b)
    evals += 2

    # Set up interior points
    x1 = a + _RESPHI * (b - a)
    x2 = b - _RESPHI * (b - a)
    if is_int:
        x1 = int(round(x1))
        x2 = int(round(x2))
    f1 = f(x1)
    f2 = f(x2)
    evals += 2

    while evals < max_evals and (b - a) > tol:
        if is_int and abs(b - a) < 2:
            break

        if (f1 > f2) == maximize:
            # Best is in [a, x2] — narrow from the right
            b = x2
            fb = f2
            x2 = x1
            f2 = f1
            x1 = a + _RESPHI * (b - a)
            if is_int:
                x1 = int(round(x1))
            f1 = f(x1)
        else:
            # Best is in [x1, b] — narrow from the left
            a = x1
            fa = f1
            x1 = x2
            f1 = f2
            x2 = b - _RESPHI * (b - a)
            if is_int:
                x2 = int(round(x2))
            f2 = f(x2)
        evals += 1

    # Return best of all evaluated points
    candidates = [(a, fa), (b, fb), (x1, f1), (x2, f2)]
    if maximize:
        best_x, best_f = max(candidates, key=lambda c: c[1])
    else:
        best_x, best_f = min(candidates, key=lambda c: c[1])

    return (int(round(best_x)) if is_int else best_x, best_f)


def tune_parameters_fast(
    code: str,
    compile_fn: Callable[[str, str], Any],
    fitness_fn: Callable[..., tuple[float, dict[str, float]]],
    function_name: str,
    direction: str = "maximize",
    param_specs: list[ParamSpec] | None = None,
    n_sweeps: int = 2,
    max_evals_per_param: int = 8,
    timeout_s: float = 60.0,
) -> TuningResult | None:
    """Tune parameters via coordinate descent + golden-section search.

    Much faster than Optuna for continuous params with cheap evaluations.
    Optimizes one parameter at a time (coordinate descent), using
    golden-section search for each 1D sub-problem. Repeats for n_sweeps.
    Skips params that don't affect fitness on first probe.
    """
    specs = param_specs or extract_params(code)
    if not specs:
        return None

    compiled_fn = compile_fn(code, function_name)
    if compiled_fn is None:
        return None

    sig = inspect.signature(compiled_fn)
    fn_accepts_p = "p" in sig.parameters

    # Evaluate with defaults
    default_params = {s.name: s.default for s in specs}
    try:
        if fn_accepts_p:
            default_fitness, _ = fitness_fn(functools.partial(compiled_fn, p=default_params))
        else:
            default_fitness, _ = fitness_fn(compiled_fn)
    except Exception as e:
        logger.debug("Default params evaluation failed: %s", e)
        return None

    maximize = direction == "maximize"
    best_params = dict(default_params)
    best_fitness = default_fitness
    total_evals = 1
    t0 = time.monotonic()

    # Measure single eval cost to set budget
    eval_t0 = time.monotonic()
    _ = fitness_fn(functools.partial(compiled_fn, p=default_params)) if fn_accepts_p else fitness_fn(compiled_fn)
    eval_cost = time.monotonic() - eval_t0
    total_evals += 1

    # Adaptive: if evals are expensive, reduce budget
    if eval_cost > 0.05:  # >50ms per eval
        max_evals_per_param = min(max_evals_per_param, 6)
        n_sweeps = 1

    for sweep in range(n_sweeps):
        for spec in specs:
            if time.monotonic() - t0 > timeout_s:
                break

            current_params = dict(best_params)

            # Quick sensitivity probe: does this param matter?
            if sweep == 0:
                probe_lo = spec.low + 0.25 * (spec.high - spec.low)
                probe_hi = spec.low + 0.75 * (spec.high - spec.low)
                if spec.is_int:
                    probe_lo, probe_hi = int(round(probe_lo)), int(round(probe_hi))

                current_params[spec.name] = probe_lo
                try:
                    fn = functools.partial(compiled_fn, p=dict(current_params)) if fn_accepts_p else compiled_fn
                    f_lo, _ = fitness_fn(fn)
                except Exception:
                    f_lo = best_fitness

                current_params[spec.name] = probe_hi
                try:
                    fn = functools.partial(compiled_fn, p=dict(current_params)) if fn_accepts_p else compiled_fn
                    f_hi, _ = fitness_fn(fn)
                except Exception:
                    f_hi = best_fitness

                total_evals += 2

                # Skip if param has no effect
                if abs(f_lo - f_hi) < 1e-10 and abs(f_lo - best_fitness) < 1e-10:
                    continue

                current_params = dict(best_params)

            def eval_param(val, _spec=spec, _params=current_params):
                _params[_spec.name] = val
                try:
                    if fn_accepts_p:
                        fn = functools.partial(compiled_fn, p=dict(_params))
                    else:
                        modified = _inject_params(code, _params)
                        fn = compile_fn(modified, function_name)
                        if fn is None:
                            return float("-inf") if maximize else float("inf")
                    fitness, _ = fitness_fn(fn)
                    return fitness
                except Exception:
                    return float("-inf") if maximize else float("inf")

            opt_val, opt_fitness = _golden_section_search(
                eval_param,
                spec.low,
                spec.high,
                maximize=maximize,
                is_int=spec.is_int,
                max_evals=max_evals_per_param,
            )

            total_evals += max_evals_per_param
            best_params[spec.name] = opt_val
            if (opt_fitness > best_fitness) == maximize:
                best_fitness = opt_fitness

    return TuningResult(
        best_fitness=best_fitness,
        best_params=best_params,
        n_trials=total_evals,
        default_fitness=default_fitness,
        improvement=best_fitness - default_fitness,
    )


# ---------------------------------------------------------------------------
# Optuna-based tuner (for when BO/CMA-ES is explicitly requested)
# ---------------------------------------------------------------------------

def tune_parameters(
    code: str,
    compile_fn: Callable[[str, str], Any],
    fitness_fn: Callable[..., tuple[float, dict[str, float]]],
    function_name: str,
    n_trials: int = 30,
    sampler: str = "tpe",  # "tpe" (BO), "cmaes", "random", "fast"
    direction: str = "maximize",
    param_specs: list[ParamSpec] | None = None,
    timeout_s: float = 60.0,
) -> TuningResult | None:
    """Tune parameters in evolved code.

    sampler="auto" (default when called with "fast"): measures eval cost
    and picks fast coordinate descent for cheap evals (<10ms) or Optuna
    TPE for expensive ones. Explicit "tpe"/"cmaes"/"random" force Optuna.
    """
    # GPU path: batch Sobol sweep + refinement
    if sampler == "gpu":
        try:
            from evolution_agent.evaluation.gpu_tuner import tune_parameters_gpu
            return tune_parameters_gpu(
                code, compile_fn, fitness_fn, function_name,
                direction=direction, param_specs=param_specs,
            )
        except ImportError:
            logger.warning("GPU tuner unavailable, falling back to fast")
            sampler = "fast"

    # Fast path: coordinate descent + golden section
    if sampler == "fast" or not OPTUNA_AVAILABLE:
        # Auto-detect: probe eval cost, use Optuna if expensive
        if sampler == "fast" and OPTUNA_AVAILABLE:
            specs_check = param_specs or extract_params(code)
            if specs_check:
                fn_check = compile_fn(code, function_name)
                if fn_check is not None:
                    sig_check = inspect.signature(fn_check)
                    t0 = time.monotonic()
                    try:
                        if "p" in sig_check.parameters:
                            fitness_fn(functools.partial(fn_check, p={s.name: s.default for s in specs_check}))
                        else:
                            fitness_fn(fn_check)
                    except Exception:
                        pass
                    eval_ms = (time.monotonic() - t0) * 1000
                    if eval_ms > 10:
                        # Expensive eval — use Optuna with few trials
                        logger.debug("Eval cost %.0fms > 10ms, using Optuna TPE", eval_ms)
                        return tune_parameters(
                            code, compile_fn, fitness_fn, function_name,
                            n_trials=min(n_trials, 15), sampler="tpe",
                            direction=direction, param_specs=param_specs,
                            timeout_s=timeout_s,
                        )
        return tune_parameters_fast(
            code, compile_fn, fitness_fn, function_name,
            direction=direction, param_specs=param_specs,
            timeout_s=timeout_s,
        )

    # Optuna path
    specs = param_specs or extract_params(code)
    if not specs:
        return None

    base_fn = compile_fn(code, function_name)
    if base_fn is None:
        return None

    try:
        default_fitness, _ = fitness_fn(base_fn)
    except Exception as e:
        logger.debug("Default params evaluation failed: %s", e)
        return None

    compiled_fn = compile_fn(code, function_name)
    if compiled_fn is None:
        return None

    sig = inspect.signature(compiled_fn)
    fn_accepts_p = "p" in sig.parameters

    def _make_fn_with_params(trial_params: dict[str, float]):
        """Create a wrapper that injects params into the function call."""
        if fn_accepts_p:
            return functools.partial(compiled_fn, p=trial_params)
        else:
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
