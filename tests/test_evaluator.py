"""Tests for function evaluator."""

import asyncio

import pytest

from evolution_agent.core.types import OptimizationDirection
from evolution_agent.evaluation.function_eval import FunctionEvaluator


def _sorting_fitness(sort_fn):
    """Simple sorting fitness: correctness on a few test cases."""
    tests = [[3, 1, 2], [5, 4, 3, 2, 1], [], [1]]
    correct = 0
    for t in tests:
        try:
            if sort_fn(list(t)) == sorted(t):
                correct += 1
        except Exception:
            pass
    return correct / len(tests), {"correct": correct, "total": len(tests)}


@pytest.fixture
def evaluator():
    return FunctionEvaluator(
        fitness_fn=_sorting_fitness,
        function_name="sort_list",
        function_spec="def sort_list(arr): sort a list",
        direction=OptimizationDirection.MAXIMIZE,
        timeout_s=5.0,
    )


@pytest.mark.asyncio
async def test_evaluate_correct_code(evaluator):
    code = "def sort_list(arr):\n    return sorted(arr)"
    result = await evaluator.evaluate(code)
    assert result.fitness == 1.0
    assert result.error is None


@pytest.mark.asyncio
async def test_evaluate_wrong_function_name(evaluator):
    code = "def wrong_name(arr):\n    return sorted(arr)"
    result = await evaluator.evaluate(code)
    assert result.fitness == float("-inf")
    assert result.error is not None


@pytest.mark.asyncio
async def test_evaluate_unsafe_code(evaluator):
    code = "import os\ndef sort_list(arr):\n    return sorted(arr)"
    result = await evaluator.evaluate(code)
    assert result.fitness == float("-inf")


@pytest.mark.asyncio
async def test_evaluate_runtime_error(evaluator):
    code = "def sort_list(arr):\n    return 1/0"
    result = await evaluator.evaluate(code)
    # Fitness function catches exceptions per test case → 0/4 correct = 0.0
    assert result.fitness == 0.0


@pytest.mark.asyncio
async def test_evaluate_partial_correctness(evaluator):
    # Returns sorted for non-empty but crashes on empty
    code = "def sort_list(arr):\n    if not arr: raise ValueError\n    return sorted(arr)"
    result = await evaluator.evaluate(code)
    assert 0.0 < result.fitness < 1.0
