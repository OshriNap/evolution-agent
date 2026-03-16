"""Abstract base evaluator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from evolution_agent.core.types import EvalResult, OptimizationDirection


class BaseEvaluator(ABC):
    """Interface that all evaluators must implement."""

    @abstractmethod
    async def evaluate(self, code: str) -> EvalResult:
        """Evaluate a code string and return fitness + metrics."""

    @abstractmethod
    def get_function_spec(self) -> str:
        """Return a description of the function signature the code must define.

        This is included in mutation prompts so the LLM knows
        what function to generate.
        """

    @abstractmethod
    def get_direction(self) -> OptimizationDirection:
        """Whether higher or lower fitness is better."""

    def is_better(self, a: float, b: float) -> bool:
        """Return True if fitness `a` is better than `b`."""
        if self.get_direction() == OptimizationDirection.MAXIMIZE:
            return a > b
        return a < b

    def worst_fitness(self) -> float:
        """Return the worst possible fitness value."""
        if self.get_direction() == OptimizationDirection.MAXIMIZE:
            return float("-inf")
        return float("inf")
