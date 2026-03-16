"""Example: evolve a prompt template.

Demonstrates using SubprocessEvaluator to evolve text (prompt templates).
The evaluator script tests the prompt against sample tasks.

Usage:
    python examples/optimize_prompt.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evolution_agent.core.config import load_config
from evolution_agent.core.engine import EvolutionEngine
from evolution_agent.core.types import OptimizationDirection
from evolution_agent.evaluation.function_eval import FunctionEvaluator


# --- Seed prompt ---
SEED_PROMPT = '''\
def generate_prompt(task_description):
    """Generate a prompt for an LLM to solve a task."""
    return f"""You are a helpful assistant. Please complete the following task:

{task_description}

Think step by step and provide a clear answer."""
'''

FUNCTION_SPEC = """\
def generate_prompt(task_description):
    \"\"\"Generate a prompt string for an LLM to solve a given task.

    Args:
        task_description: A string describing the task to solve.

    Returns:
        A string containing the complete prompt to send to an LLM.
        The prompt should instruct the LLM to solve the task effectively.
    \"\"\"
"""


# --- Simple fitness function ---
def evaluate_prompt(prompt_fn) -> tuple[float, dict[str, float]]:
    """Evaluate a prompt template on structural quality metrics.

    This is a simplified evaluator that scores prompts based on
    structural properties (length, clarity markers, etc.)
    without actually calling an LLM.
    """
    test_tasks = [
        "Calculate the area of a circle with radius 5",
        "Write a haiku about programming",
        "Explain quantum computing in simple terms",
        "Debug this Python code: x = [1,2,3]; print(x[5])",
        "Summarize the key differences between TCP and UDP",
    ]

    total_score = 0.0
    metrics: dict[str, float] = {}

    for i, task in enumerate(test_tasks):
        try:
            prompt = prompt_fn(task)
        except Exception:
            continue

        if not isinstance(prompt, str):
            continue

        score = 0.0

        # Contains the task
        if task in prompt:
            score += 0.2

        # Has structure (multiple lines)
        lines = prompt.strip().split("\n")
        if len(lines) >= 3:
            score += 0.1

        # Has role/persona
        role_words = ["assistant", "expert", "you are", "your role"]
        if any(w in prompt.lower() for w in role_words):
            score += 0.1

        # Has thinking instruction
        think_words = ["step by step", "think", "reason", "analyze", "consider"]
        if any(w in prompt.lower() for w in think_words):
            score += 0.15

        # Has output format instruction
        format_words = ["format", "output", "answer", "respond", "provide"]
        if any(w in prompt.lower() for w in format_words):
            score += 0.1

        # Reasonable length (not too short, not too long)
        if 100 < len(prompt) < 2000:
            score += 0.1

        # Has sections or markers
        if any(m in prompt for m in ["##", "**", "---", "1.", "- "]):
            score += 0.1

        # Penalty for very short prompts
        if len(prompt) < 50:
            score -= 0.2

        # Penalty for not including the task
        if task not in prompt:
            score -= 0.3

        total_score += max(0.0, score)

    fitness = total_score / len(test_tasks)
    metrics["avg_per_task"] = fitness

    return fitness, metrics


async def main() -> None:
    config = load_config(
        str(Path(__file__).parent.parent / "config" / "default.yaml"),
        overrides={
            "population_size": 8,
            "max_generations": 20,
            "elite_count": 2,
            "analyzer_every_n_gens": 5,
            "stagnation_limit": 10,
        },
    )

    evaluator = FunctionEvaluator(
        fitness_fn=evaluate_prompt,
        function_name="generate_prompt",
        function_spec=FUNCTION_SPEC,
        direction=OptimizationDirection.MAXIMIZE,
        timeout_s=10.0,
    )

    engine = EvolutionEngine(
        config=config,
        evaluator=evaluator,
        seeds=[SEED_PROMPT],
    )

    summary = await engine.run()

    print("\n" + "=" * 60)
    print("PROMPT EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Generations: {summary['total_generations']}")
    print(f"Best fitness: {summary['best_fitness']:.6f}")
    print(f"Elapsed: {summary['elapsed_s']:.1f}s")
    print("\nBest prompt template:")
    print("-" * 40)
    print(summary.get("best_code", "N/A"))


if __name__ == "__main__":
    asyncio.run(main())
