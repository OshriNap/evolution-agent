"""Example: evolve a metric TSP heuristic.

Evolves a function that constructs a tour for the Travelling Salesman Problem
on 2D Euclidean points. Fitness = baseline_length / tour_length (higher = shorter tour).

The seed is a naive nearest-neighbor heuristic. Evolution should discover
improvements like 2-opt, greedy insertion, or creative hybrids.

Usage:
    python examples/optimize_tsp.py
    EVOL_POPULATION_SIZE=8 EVOL_MAX_GENERATIONS=20 python examples/optimize_tsp.py
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evolution_agent.core.config import load_config
from evolution_agent.core.engine import EvolutionEngine
from evolution_agent.core.types import OptimizationDirection
from evolution_agent.evaluation.function_eval import FunctionEvaluator
from evolution_agent.evaluation.hybrid_eval import HybridEvaluator


# --- Seed: nearest neighbor with tunable start-selection ---
SEED_NEAREST_NEIGHBOR = """\
def solve_tsp(points, p=None):
    # Nearest-neighbor heuristic with tunable parameters
    if p is None:
        p = {"start_bias": 0.0, "lookahead_weight": 1.0}
    n = len(points)
    if n <= 1:
        return list(range(n))
    # Pick starting point (bias toward centroid or index 0)
    cx = sum(x for x, y in points) / n
    cy = sum(y for x, y in points) / n
    start = 0
    if p["start_bias"] > 0.5:
        # Start from point closest to centroid
        start = min(range(n), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2)
    visited = [False] * n
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        last = tour[-1]
        best_dist = float('inf')
        best_next = -1
        for j in range(n):
            if not visited[j]:
                dx = points[last][0] - points[j][0]
                dy = points[last][1] - points[j][1]
                d = (dx * dx + dy * dy) ** 0.5
                score = d * p["lookahead_weight"]
                if score < best_dist:
                    best_dist = score
                    best_next = j
        tour.append(best_next)
        visited[best_next] = True
    return tour
"""

FUNCTION_SPEC = """\
def solve_tsp(points, p=None):
    # Solve the Travelling Salesman Problem for 2D Euclidean points.
    #
    # Args:
    #     points: list of (x, y) tuples, coordinates in [0, 1000] range.
    #             Typically 20-50 points.
    #     p: optional dict of tunable numeric parameters with defaults.
    #        Put all tunable constants in p so they can be auto-optimized.
    #        Example: if p is None: p = {"alpha": 1.0, "threshold": 0.5}
    #        Access via p["alpha"], p["threshold"], etc.
    #
    # Returns:
    #     A list of integer indices [i0, i1, ..., i_{n-1}] representing
    #     the order to visit points. The tour implicitly returns to i0.
    #     Must contain each index exactly once (a permutation of range(n)).
    #
    # Available: math module, range, len, sorted, enumerate, zip, min, max,
    #            list, tuple, set, dict, float, int, abs, sum, round, pow, any, all
    #
    # Goal: minimize total tour length (sum of Euclidean distances between
    #        consecutive points, including return to start).
"""


def _tour_length(points, tour):
    """Total Euclidean tour length including return to start."""
    total = 0.0
    n = len(tour)
    for i in range(n):
        x1, y1 = points[tour[i]]
        x2, y2 = points[tour[(i + 1) % n]]
        total += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return total


def _make_instances(seed=42):
    """Fixed test instances."""
    rng = random.Random(seed)
    instances = []
    for n in [15, 20, 25, 30, 40]:
        pts = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n)]
        instances.append(pts)
    # clustered
    pts = []
    for cx, cy in [(200, 200), (800, 200), (500, 800)]:
        for _ in range(10):
            pts.append((cx + rng.gauss(0, 60), cy + rng.gauss(0, 60)))
    instances.append(pts)
    return instances


TEST_INSTANCES = _make_instances()


def _nn_baseline(points):
    n = len(points)
    visited = [False] * n
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = tour[-1]
        best_d, best_j = float('inf'), 0
        for j in range(n):
            if not visited[j]:
                d = math.dist(points[last], points[j])
                if d < best_d:
                    best_d, best_j = d, j
        tour.append(best_j)
        visited[best_j] = True
    return tour


BASELINE_LENGTHS = [_tour_length(pts, _nn_baseline(pts)) for pts in TEST_INSTANCES]


def evaluate_tsp(solve_fn) -> tuple[float, dict[str, float]]:
    """Evaluate a TSP heuristic.

    Fitness = avg(baseline_length / tour_length) across test instances.
    1.0 = same as nearest-neighbor. >1.0 = better. 0 = broken.
    """
    ratios = []
    errors = 0

    for points, baseline_len in zip(TEST_INSTANCES, BASELINE_LENGTHS):
        try:
            tour = solve_fn(list(points))

            # Valid permutation?
            if sorted(tour) != list(range(len(points))):
                errors += 1
                continue

            length = _tour_length(points, tour)
            ratios.append(baseline_len / length if length > 0 else 0)
        except Exception:
            errors += 1

    if not ratios:
        return 0.0, {"errors": errors}

    avg_ratio = sum(ratios) / len(ratios)
    return avg_ratio, {
        "avg_ratio": round(avg_ratio, 4),
        "best_ratio": round(max(ratios), 4),
        "worst_ratio": round(min(ratios), 4),
        "valid": len(ratios),
        "errors": errors,
        "instance_ratios": ratios,  # behavioral embedding for curiosity
    }


async def main() -> None:
    config = load_config(
        str(Path(__file__).parent.parent / "config" / "default.yaml"),
        overrides={
            "population_size": int(os.environ.get("EVOL_POPULATION_SIZE", "8")),
            "max_generations": int(os.environ.get("EVOL_MAX_GENERATIONS", "15")),
            "elite_count": 2,
            "analyzer_every_n_gens": int(os.environ.get("EVOL_ANALYZER_EVERY", "5")),
            "analyzer_model": os.environ.get("EVOL_ANALYZER_MODEL", "claude-code:sonnet"),
            "stagnation_limit": 10,
            "ollama_base_url": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            "meta_optimizer_type": os.environ.get("EVOL_META_OPTIMIZER_TYPE", "heuristic"),
            "max_concurrent_mutations": 1,  # Ollama serves one at a time
            "max_concurrent_evals": 1,  # sequential to save memory with hybrid tuning
            "eval_timeout_s": 10.0,
        },
    )

    use_hybrid = os.environ.get("EVOL_HYBRID", "1") == "1"
    sampler = os.environ.get("EVOL_SAMPLER", "fast")  # fast (coord descent), tpe (BO), cmaes, random
    tuning_trials = int(os.environ.get("EVOL_TUNING_TRIALS", "15"))

    curiosity_weight = float(os.environ.get("EVOL_CURIOSITY", "0.0"))

    if use_hybrid:
        evaluator = HybridEvaluator(
            fitness_fn=evaluate_tsp,
            function_name="solve_tsp",
            function_spec=FUNCTION_SPEC,
            direction=OptimizationDirection.MAXIMIZE,
            timeout_s=30.0,
            tuning_trials=tuning_trials,
            tuning_sampler=sampler,
            tuning_timeout_s=15.0,
            tune_threshold=0.5,  # don't tune obviously broken code
            curiosity_weight=curiosity_weight,
        )
        mode = f"Hybrid mode: LLM structure + {sampler.upper()} parameter tuning ({tuning_trials} trials)"
        if curiosity_weight > 0:
            mode += f" + curiosity (λ={curiosity_weight})"
        print(mode)
    else:
        evaluator = FunctionEvaluator(
            fitness_fn=evaluate_tsp,
            function_name="solve_tsp",
            function_spec=FUNCTION_SPEC,
            direction=OptimizationDirection.MAXIMIZE,
            timeout_s=10.0,
        )
        print("LLM-only mode (no parameter tuning)")

    engine = EvolutionEngine(
        config=config,
        evaluator=evaluator,
        seeds=[SEED_NEAREST_NEIGHBOR],
    )

    summary = await engine.run()

    print("\n" + "=" * 60)
    print("TSP EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Generations: {summary['total_generations']}")
    print(f"Best fitness: {summary['best_fitness']:.6f}")
    print(f"  (1.0 = nearest-neighbor, higher = shorter tours)")
    print(f"Elapsed: {summary['elapsed_s']:.1f}s")
    if use_hybrid and hasattr(evaluator, 'get_tuning_stats'):
        stats = evaluator.get_tuning_stats()
        print(f"Tuning stats: {stats}")
    print("\nBest heuristic:")
    print("-" * 40)
    print(summary.get("best_code", "N/A"))


if __name__ == "__main__":
    asyncio.run(main())
