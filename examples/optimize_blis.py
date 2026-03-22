"""Example: evolve LLM serving routing policies using BLIS simulator.

Evolves a Python function that computes routing scores for each instance
in a multi-instance LLM serving cluster. Fitness = minimizing tail latency
(e2e_p99) while maintaining throughput.

The evolved function receives instance state (queue depth, KV utilization,
batch size, etc.) and returns a routing score. BLIS simulator evaluates
the policy across multiple workload profiles.

Usage:
    python examples/optimize_blis.py
    EVOL_POPULATION_SIZE=8 EVOL_MAX_GENERATIONS=20 python examples/optimize_blis.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evolution_agent.integrations.blis_runner import BLISRunner, BLISConfig

# ── BLIS configuration ────────────────────────────────────────────

BLIS_BINARY = str(Path(__file__).parent.parent / "bin" / "blis")

BASE_CONFIG = BLISConfig(
    binary_path=BLIS_BINARY,
    model="meta-llama/llama-3.1-8b-instruct",
    num_requests=200,
    seed=42,
    base_args={
        "num_instances": 4,
        "total_kv_blocks": 50000,
        "max_num_running_reqs": 256,
    },
)

# Workload profiles to test against
WORKLOAD_PROFILES = {
    "steady": {
        "rate": 10,
        "prompt_tokens": 512,
        "output_tokens": 128,
    },
    "bursty": {
        "rate": 30,
        "prompt_tokens": 256,
        "output_tokens": 64,
    },
    "heavy": {
        "rate": 5,
        "prompt_tokens": 2048,
        "output_tokens": 512,
    },
}


# ── Fitness function ──────────────────────────────────────────────

def evaluate_routing_policy(fn) -> tuple[float, dict[str, float]]:
    """Evaluate a routing scorer function using BLIS.

    The evolved function should return routing scorer weights as a string
    like "queue-depth:2.5,kv-utilization:1.8,load-balance:3.0".

    Fitness = harmonic mean of per-workload normalized scores.
    Higher is better (lower latency = higher fitness).
    """
    runner = BLISRunner(BASE_CONFIG)

    try:
        scorer_config = fn()
    except Exception as e:
        return 0.0, {"error": str(e), "valid": False}

    if not isinstance(scorer_config, dict):
        return 0.0, {"error": "must return dict", "valid": False}

    # Extract routing config
    routing_scorers = scorer_config.get("routing_scorers", "queue-depth:1")
    scheduler = scorer_config.get("scheduler", "fcfs")
    priority_policy = scorer_config.get("priority_policy", "constant")
    max_batch = scorer_config.get("max_num_running_reqs", 256)

    workload_scores = {}
    workload_metrics = {}
    errors = 0

    for profile_name, profile_params in WORKLOAD_PROFILES.items():
        params = {
            "routing_policy": "weighted",
            "routing_scorers": routing_scorers,
            "scheduler": scheduler,
            "priority_policy": priority_policy,
            "max_num_running_reqs": int(max_batch),
            **profile_params,
        }

        result = runner.run(params)

        if not result.success or result.completed_requests == 0:
            errors += 1
            workload_scores[profile_name] = 0.0
            workload_metrics[f"{profile_name}_e2e_p99"] = 999999.0
            workload_metrics[f"{profile_name}_tps"] = 0.0
            continue

        # Normalize: lower latency = higher score
        e2e_p99 = max(result.e2e_p99_ms, 1.0)
        completion_rate = result.completed_requests / BASE_CONFIG.num_requests
        score = (1000.0 / e2e_p99) * completion_rate

        workload_scores[profile_name] = score
        workload_metrics[f"{profile_name}_e2e_p99"] = result.e2e_p99_ms
        workload_metrics[f"{profile_name}_tps"] = result.tokens_per_sec
        workload_metrics[f"{profile_name}_ttft_p99"] = result.ttft_p99_ms
        workload_metrics[f"{profile_name}_completed"] = result.completed_requests

    if errors == len(WORKLOAD_PROFILES):
        return 0.0, {"valid": False, "errors": errors, **workload_metrics}

    # Harmonic mean of scores across workloads (rewards balanced performance)
    scores = [s for s in workload_scores.values() if s > 0]
    if not scores:
        fitness = 0.0
    else:
        fitness = len(scores) / sum(1.0 / s for s in scores)

    metrics = {
        "valid": True,
        "errors": errors,
        "workload_scores": list(workload_scores.values()),
        **workload_scores,
        **workload_metrics,
    }
    return fitness, metrics


# ── Function spec and seeds ───────────────────────────────────────

FUNCTION_SPEC = """\
def configure_routing(p=None):
    # Configure the routing policy for a multi-instance LLM serving cluster.
    #
    # Args:
    #     p: optional dict of tunable numeric parameters with defaults.
    #        Put all tunable constants in p so they can be auto-optimized.
    #
    # Returns:
    #     A dict with routing configuration:
    #     {
    #         "routing_scorers": str,  # comma-separated "scorer:weight" pairs
    #         "scheduler": str,        # "fcfs" or "sjf" (shortest job first)
    #         "priority_policy": str,  # "constant" or "slo-based"
    #         "max_num_running_reqs": int,  # max concurrent batch size
    #     }
    #
    # Available scorers (combine any subset with weights):
    #     queue-depth      — prefer instances with shorter queues
    #     kv-utilization   — prefer instances with lower KV cache pressure
    #     load-balance     — distribute requests evenly across instances
    #     prefix-affinity  — route to instance with cached prefix (reduces TTFT)
    #
    # Available globals (do NOT import anything):
    #     math, range, len, min, max, int, float, abs, sum, round, str
    #
    # The cluster has 4 instances serving llama-3.1-8b.
    # Workloads vary: steady (10 rps), bursty (30 rps), heavy (5 rps, long prompts).
    # Goal: minimize e2e_p99 latency across all workload profiles.
    #
    # Strategy hints:
    #   - queue-depth is critical under high load (bursty)
    #   - kv-utilization matters for heavy prompts (KV cache pressure)
    #   - load-balance prevents hotspots
    #   - prefix-affinity reduces TTFT but can cause imbalance
    #   - sjf scheduler helps under mixed prompt lengths
    #   - Higher max_num_running_reqs = more throughput but more memory pressure
"""

SEED_BASIC = """\
def configure_routing(p=None):
    # Simple equal-weight routing
    if p is None:
        p = {"qd": 1.0, "kv": 1.0, "lb": 1.0}
    scorers = f"queue-depth:{p['qd']:.2f},kv-utilization:{p['kv']:.2f},load-balance:{p['lb']:.2f}"
    return {"routing_scorers": scorers, "scheduler": "fcfs",
            "priority_policy": "constant", "max_num_running_reqs": 256}
"""

SEED_QUEUE_HEAVY = """\
def configure_routing(p=None):
    # Queue-depth focused routing with SJF
    if p is None:
        p = {"qd": 5.0, "kv": 1.0, "lb": 2.0, "pa": 0.5, "batch": 200.0}
    scorers = (f"queue-depth:{p['qd']:.2f},kv-utilization:{p['kv']:.2f},"
               f"load-balance:{p['lb']:.2f},prefix-affinity:{p['pa']:.2f}")
    return {"routing_scorers": scorers, "scheduler": "sjf",
            "priority_policy": "slo-based", "max_num_running_reqs": int(p["batch"])}
"""

SEED_ADAPTIVE = """\
def configure_routing(p=None):
    # Balanced routing with all scorers
    if p is None:
        p = {"qd": 3.0, "kv": 2.0, "lb": 1.5, "pa": 1.0, "batch": 256.0}
    scorers = (f"queue-depth:{p['qd']:.2f},kv-utilization:{p['kv']:.2f},"
               f"load-balance:{p['lb']:.2f},prefix-affinity:{p['pa']:.2f}")
    return {"routing_scorers": scorers, "scheduler": "sjf",
            "priority_policy": "constant", "max_num_running_reqs": int(p["batch"])}
"""

SEEDS = [SEED_BASIC, SEED_QUEUE_HEAVY, SEED_ADAPTIVE]


# ── Main ──────────────────────────────────────────────────────────

async def main() -> None:
    from evolution_agent.core.config import load_config
    from evolution_agent.core.engine import EvolutionEngine
    from evolution_agent.core.types import OptimizationDirection
    from evolution_agent.evaluation.function_eval import FunctionEvaluator
    from evolution_agent.evaluation.hybrid_eval import HybridEvaluator

    config = load_config(
        str(Path(__file__).parent.parent / "config" / "default.yaml"),
        overrides={
            "population_size": int(os.environ.get("EVOL_POPULATION_SIZE", "8")),
            "max_generations": int(os.environ.get("EVOL_MAX_GENERATIONS", "20")),
            "elite_count": 2,
            "analyzer_every_n_gens": int(os.environ.get("EVOL_ANALYZER_EVERY", "5")),
            "analyzer_model": os.environ.get("EVOL_ANALYZER_MODEL", "claude-code:sonnet"),
            "stagnation_limit": 10,
            "meta_optimizer_type": "heuristic",
            "max_concurrent_mutations": 1,
            "max_concurrent_evals": 1,
            "eval_timeout_s": 120.0,  # BLIS runs take time
        },
    )

    use_hybrid = os.environ.get("EVOL_HYBRID", "1") == "1"
    sampler = os.environ.get("EVOL_SAMPLER", "fast")
    tuning_trials = int(os.environ.get("EVOL_TUNING_TRIALS", "10"))

    if use_hybrid:
        evaluator = HybridEvaluator(
            fitness_fn=evaluate_routing_policy,
            function_name="configure_routing",
            function_spec=FUNCTION_SPEC,
            direction=OptimizationDirection.MAXIMIZE,
            timeout_s=120.0,
            tuning_trials=tuning_trials,
            tuning_sampler=sampler,
            tuning_timeout_s=300.0,
            tune_threshold=0.01,
            embedding_key="workload_scores",
        )
        print(f"Hybrid mode: LLM structure + {sampler.upper()} parameter tuning")
    else:
        evaluator = FunctionEvaluator(
            fitness_fn=evaluate_routing_policy,
            function_name="configure_routing",
            function_spec=FUNCTION_SPEC,
            direction=OptimizationDirection.MAXIMIZE,
            timeout_s=120.0,
        )
        print("LLM-only mode")

    engine = EvolutionEngine(
        config=config,
        evaluator=evaluator,
        seeds=SEEDS,
    )

    run_dir = str(engine._run_dir)
    summary = await engine.run()

    print("\n" + "=" * 60)
    print("BLIS ROUTING EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Generations: {summary['total_generations']}")
    print(f"Best fitness: {summary['best_fitness']:.4f}")
    print(f"Elapsed: {summary['elapsed_s']:.1f}s")
    print("\nBest routing policy:")
    print("-" * 40)
    print(summary.get("best_code", "N/A"))


if __name__ == "__main__":
    asyncio.run(main())
