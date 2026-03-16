"""Hybrid evaluator: LLM evolves structure, optimizer tunes parameters.

Wraps a FunctionEvaluator. For each code variant:
1. Evaluate with default params (quick reject if code is broken)
2. If code has tunable `p = {...}` params, run Optuna to find best values
3. Return best fitness (after tuning) and bake optimized params into the code
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from evolution_agent.core.types import EvalResult, OptimizationDirection
from evolution_agent.evaluation.base import BaseEvaluator
from evolution_agent.evaluation.parameter_tuner import (
    TuningResult,
    extract_params,
    tune_parameters,
)
from evolution_agent.evaluation.sandbox import CodeSandbox

logger = logging.getLogger(__name__)


class HybridEvaluator(BaseEvaluator):
    """Evaluator that tunes numeric parameters after structural evaluation.

    The evolved function should use a `p` dict for tunable constants:

        def solve(input, p=None):
            if p is None:
                p = {"alpha": 1.0, "beta": 0.5}
            ...

    Evaluation flow:
    1. Compile + evaluate with default params
    2. If fitness > threshold, run parameter tuning (Optuna)
    3. Return best fitness and store optimized params in metadata
    """

    def __init__(
        self,
        fitness_fn: Callable[..., tuple[float, dict[str, float]]],
        function_name: str,
        function_spec: str,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        timeout_s: float = 30.0,
        # Tuning options
        tuning_trials: int = 30,
        tuning_sampler: str = "tpe",  # "tpe" (BO), "cmaes", "random"
        tuning_timeout_s: float = 30.0,
        tune_threshold: float | None = None,  # min fitness to trigger tuning
    ) -> None:
        self._fitness_fn = fitness_fn
        self._function_name = function_name
        self._function_spec = function_spec
        self._direction = direction
        self._timeout_s = timeout_s
        self._sandbox = CodeSandbox()

        self._tuning_trials = tuning_trials
        self._tuning_sampler = tuning_sampler
        self._tuning_timeout_s = tuning_timeout_s
        self._tune_threshold = tune_threshold

        self._tuning_stats = {"tuned": 0, "skipped": 0, "no_params": 0, "improved": 0}

    async def evaluate(self, code: str) -> EvalResult:
        t0 = time.monotonic()

        # 1. Compile
        fn = self._sandbox.compile_function(code, self._function_name)
        if fn is None:
            return EvalResult(
                fitness=self.worst_fitness(),
                error="Code compilation/validation failed",
                eval_time_s=time.monotonic() - t0,
            )

        # 2. Evaluate with defaults
        try:
            default_fitness, default_metrics = self._fitness_fn(fn)
        except Exception as e:
            return EvalResult(
                fitness=self.worst_fitness(),
                error=f"Default evaluation failed: {e}",
                eval_time_s=time.monotonic() - t0,
            )

        # 3. Check if code has tunable params
        params = extract_params(code)
        if not params:
            self._tuning_stats["no_params"] += 1
            return EvalResult(
                fitness=default_fitness,
                metrics={**default_metrics, "tuned": False, "n_params": 0},
                eval_time_s=time.monotonic() - t0,
            )

        # 4. Check threshold — don't waste tuning on bad code
        if self._tune_threshold is not None:
            if self._direction == OptimizationDirection.MAXIMIZE:
                if default_fitness < self._tune_threshold:
                    self._tuning_stats["skipped"] += 1
                    return EvalResult(
                        fitness=default_fitness,
                        metrics={**default_metrics, "tuned": False,
                                 "n_params": len(params), "skip_reason": "below_threshold"},
                        eval_time_s=time.monotonic() - t0,
                    )
            else:
                if default_fitness > self._tune_threshold:
                    self._tuning_stats["skipped"] += 1
                    return EvalResult(
                        fitness=default_fitness,
                        metrics={**default_metrics, "tuned": False,
                                 "n_params": len(params), "skip_reason": "above_threshold"},
                        eval_time_s=time.monotonic() - t0,
                    )

        # 5. Tune parameters
        self._tuning_stats["tuned"] += 1
        remaining_time = max(5.0, self._tuning_timeout_s - (time.monotonic() - t0))

        result = tune_parameters(
            code=code,
            compile_fn=self._sandbox.compile_function,
            fitness_fn=self._fitness_fn,
            function_name=self._function_name,
            n_trials=self._tuning_trials,
            sampler=self._tuning_sampler,
            direction=self._direction.value,
            timeout_s=remaining_time,
        )

        if result is None:
            return EvalResult(
                fitness=default_fitness,
                metrics={**default_metrics, "tuned": False, "n_params": len(params)},
                eval_time_s=time.monotonic() - t0,
            )

        if result.improvement > 0:
            self._tuning_stats["improved"] += 1

        best_fitness = result.best_fitness
        metrics = {
            **default_metrics,
            "tuned": True,
            "n_params": len(params),
            "default_fitness": result.default_fitness,
            "tuned_fitness": result.best_fitness,
            "tuning_improvement": result.improvement,
            "tuning_trials": result.n_trials,
            "tuning_sampler": self._tuning_sampler,
            **{f"p_{k}": v for k, v in result.best_params.items()},
        }

        logger.info(
            "Tuned %d params (%s): %.4f → %.4f (+%.4f) in %d trials",
            len(params), self._tuning_sampler,
            result.default_fitness, result.best_fitness,
            result.improvement, result.n_trials,
        )

        return EvalResult(
            fitness=best_fitness,
            metrics=metrics,
            eval_time_s=time.monotonic() - t0,
        )

    def get_function_spec(self) -> str:
        return self._function_spec

    def get_direction(self) -> OptimizationDirection:
        return self._direction

    def get_tuning_stats(self) -> dict[str, int]:
        return dict(self._tuning_stats)
