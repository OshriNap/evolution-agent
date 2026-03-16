"""Function evaluator: evaluate code by calling a Python fitness function."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from evolution_agent.core.types import EvalResult, OptimizationDirection
from evolution_agent.evaluation.base import BaseEvaluator
from evolution_agent.evaluation.sandbox import CodeSandbox

logger = logging.getLogger(__name__)


class FunctionEvaluator(BaseEvaluator):
    """Evaluate evolved code by extracting a function and passing it to a fitness callable.

    The fitness function receives the compiled callable and returns
    (fitness, metrics_dict).
    """

    def __init__(
        self,
        fitness_fn: Callable[..., tuple[float, dict[str, float]]],
        function_name: str,
        function_spec: str,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        timeout_s: float = 30.0,
    ) -> None:
        self._fitness_fn = fitness_fn
        self._function_name = function_name
        self._function_spec = function_spec
        self._direction = direction
        self._timeout_s = timeout_s
        self._sandbox = CodeSandbox()

    async def evaluate(self, code: str) -> EvalResult:
        t0 = time.monotonic()

        # Compile the function from code
        fn = self._sandbox.compile_function(code, self._function_name)
        if fn is None:
            return EvalResult(
                fitness=self.worst_fitness(),
                error="Code compilation/validation failed",
                eval_time_s=time.monotonic() - t0,
            )

        # Run the fitness function with timeout
        try:
            loop = asyncio.get_event_loop()
            fitness, metrics = await asyncio.wait_for(
                loop.run_in_executor(None, self._fitness_fn, fn),
                timeout=self._timeout_s,
            )
            return EvalResult(
                fitness=fitness,
                metrics=metrics,
                eval_time_s=time.monotonic() - t0,
            )
        except asyncio.TimeoutError:
            return EvalResult(
                fitness=self.worst_fitness(),
                error=f"Evaluation timed out after {self._timeout_s}s",
                eval_time_s=time.monotonic() - t0,
            )
        except Exception as e:
            return EvalResult(
                fitness=self.worst_fitness(),
                error=str(e),
                eval_time_s=time.monotonic() - t0,
            )

    def get_function_spec(self) -> str:
        return self._function_spec

    def get_direction(self) -> OptimizationDirection:
        return self._direction
