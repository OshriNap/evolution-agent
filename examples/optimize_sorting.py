"""Example: evolve a sorting function.

Demonstrates using FunctionEvaluator to evolve a sorting algorithm.
The fitness function measures correctness + speed.

Usage:
    python examples/optimize_sorting.py
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evolution_agent.core.config import load_config
from evolution_agent.core.engine import EvolutionEngine
from evolution_agent.core.types import OptimizationDirection
from evolution_agent.evaluation.function_eval import FunctionEvaluator


# --- Seed code ---
SEED_SORTING = """\
def sort_list(arr):
    \"\"\"Sort a list of numbers in ascending order.\"\"\"
    if len(arr) <= 1:
        return arr
    result = list(arr)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if result[i] > result[j]:
                result[i], result[j] = result[j], result[i]
    return result
"""

FUNCTION_SPEC = """\
def sort_list(arr):
    \"\"\"Sort a list of numbers in ascending order.

    Args:
        arr: A list of numbers (int or float).

    Returns:
        A new sorted list in ascending order.
        Must not modify the input list.
    \"\"\"
"""


# --- Fitness function ---
def evaluate_sorting(sort_fn) -> tuple[float, dict[str, float]]:
    """Evaluate a sorting function on correctness and speed.

    Returns (fitness, metrics_dict).
    Fitness = correctness_score * (1 + speed_bonus)
    """
    test_cases = [
        [],
        [1],
        [2, 1],
        [3, 1, 2],
        [5, 4, 3, 2, 1],
        list(range(20)),
        list(range(20, 0, -1)),
        [random.randint(-100, 100) for _ in range(50)],
        [random.randint(-1000, 1000) for _ in range(100)],
        [1, 1, 1, 1],
        [1, 2, 1, 2, 1],
        [-5, 0, 5, -10, 10],
        [0.5, 0.1, 0.9, 0.3, 0.7],
    ]

    correct = 0
    total = len(test_cases)

    for tc in test_cases:
        try:
            result = sort_fn(list(tc))
            expected = sorted(tc)
            if result == expected:
                correct += 1
        except Exception:
            pass

    correctness = correct / total

    # Speed test (only if correct)
    speed_score = 0.0
    if correctness >= 0.9:
        large = [random.randint(-10000, 10000) for _ in range(500)]
        t0 = time.perf_counter()
        for _ in range(10):
            sort_fn(list(large))
        elapsed = time.perf_counter() - t0

        # Baseline: Python's sorted() takes ~X ms
        t1 = time.perf_counter()
        for _ in range(10):
            sorted(list(large))
        baseline = time.perf_counter() - t1

        # Speed bonus: up to 0.5 for being as fast as baseline
        if elapsed > 0:
            speed_ratio = baseline / elapsed
            speed_score = min(0.5, speed_ratio * 0.5)

    fitness = correctness * (1.0 + speed_score)

    return fitness, {
        "correctness": correctness,
        "speed_score": speed_score,
        "correct_cases": correct,
        "total_cases": total,
    }


async def main() -> None:
    config = load_config(
        str(Path(__file__).parent.parent / "config" / "default.yaml"),
        overrides={
            "population_size": int(os.environ.get("EVOL_POPULATION_SIZE", "6")),
            "max_generations": int(os.environ.get("EVOL_MAX_GENERATIONS", "10")),
            "elite_count": 2,
            "analyzer_every_n_gens": 5,
            "stagnation_limit": 8,
            "ollama_base_url": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            "meta_optimizer_type": os.environ.get("EVOL_META_OPTIMIZER_TYPE", "heuristic"),
        },
    )

    evaluator = FunctionEvaluator(
        fitness_fn=evaluate_sorting,
        function_name="sort_list",
        function_spec=FUNCTION_SPEC,
        direction=OptimizationDirection.MAXIMIZE,
        timeout_s=10.0,
    )

    engine = EvolutionEngine(
        config=config,
        evaluator=evaluator,
        seeds=[SEED_SORTING],
    )

    summary = await engine.run()

    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Generations: {summary['total_generations']}")
    print(f"Best fitness: {summary['best_fitness']:.6f}")
    print(f"Elapsed: {summary['elapsed_s']:.1f}s")
    print("\nBest code:")
    print("-" * 40)
    print(summary.get("best_code", "N/A"))


if __name__ == "__main__":
    asyncio.run(main())
