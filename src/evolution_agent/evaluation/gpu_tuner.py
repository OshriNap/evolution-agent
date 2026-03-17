"""GPU-accelerated parameter tuning via batch evaluation.

Uses PyTorch CUDA to parallelize fitness evaluation across many parameter
combinations simultaneously. Works in two modes:

1. **GPU tour-length mode** (for TSP-like problems): The evolved function
   generates tours, and tour lengths are computed on GPU in batch.

2. **CPU multiprocess + GPU scoring**: Evolved functions run on CPU workers,
   GPU handles batch scoring/aggregation.

The key speedup: instead of evaluating parameter combos one at a time
(golden-section: ~50 evals sequential, Optuna: ~15 evals sequential),
we evaluate 200-500 combos in parallel.
"""

from __future__ import annotations

import functools
import inspect
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from evolution_agent.evaluation.parameter_tuner import (
    ParamSpec,
    TuningResult,
    _inject_params,
    extract_params,
)

logger = logging.getLogger(__name__)

try:
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False


def _sobol_samples(specs: list[ParamSpec], n: int) -> list[dict[str, float]]:
    """Generate quasi-random parameter samples using Sobol sequences.

    Falls back to stratified random if torch.quasirandom unavailable.
    """
    d = len(specs)
    try:
        engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True)
        raw = engine.draw(n)  # shape: (n, d), values in [0, 1]
    except Exception:
        raw = torch.rand(n, d)

    samples = []
    for i in range(n):
        params = {}
        for j, spec in enumerate(specs):
            u = raw[i, j].item()
            if spec.log_scale and spec.low > 0:
                # Log-uniform
                log_low = math.log(spec.low)
                log_high = math.log(spec.high)
                val = math.exp(log_low + u * (log_high - log_low))
            else:
                val = spec.low + u * (spec.high - spec.low)
            if spec.is_int:
                val = int(round(val))
            params[spec.name] = val
        samples.append(params)

    return samples


def _eval_single(args: tuple) -> tuple[int, float, dict]:
    """Worker function for multiprocessing evaluation."""
    idx, code, function_name, params, fitness_fn_pickle_args = args
    try:
        # Re-import and reconstruct in worker process
        from evolution_agent.evaluation.sandbox import CodeSandbox

        sandbox = CodeSandbox()
        fn = sandbox.compile_function(code, function_name)
        if fn is None:
            return (idx, float("-inf"), params)

        sig = inspect.signature(fn)
        if "p" in sig.parameters:
            fn = functools.partial(fn, p=params)

        fitness, _ = fitness_fn_pickle_args(fn)
        return (idx, fitness, params)
    except Exception:
        return (idx, float("-inf"), params)


@dataclass
class GPUTunerConfig:
    """Configuration for GPU-accelerated parameter tuning."""

    n_samples: int = 200  # Number of parameter combos to evaluate
    max_workers: int = 4  # CPU workers for function execution
    refine_top_k: int = 5  # Refine top-K with local search
    refine_samples: int = 25  # Samples around each top-K
    timeout_s: float = 30.0


def tune_parameters_gpu(
    code: str,
    compile_fn: Callable[[str, str], Any],
    fitness_fn: Callable[..., tuple[float, dict[str, float]]],
    function_name: str,
    direction: str = "maximize",
    param_specs: list[ParamSpec] | None = None,
    config: GPUTunerConfig | None = None,
) -> TuningResult | None:
    """Tune parameters using GPU-accelerated batch evaluation.

    Strategy:
    1. Generate N quasi-random parameter combos (Sobol sequence)
    2. Evaluate all combos in parallel using process pool
    3. Refine top-K with local perturbation search
    4. Return best
    """
    cfg = config or GPUTunerConfig()
    specs = param_specs or extract_params(code)
    if not specs:
        return None

    # Compile and evaluate with defaults
    compiled_fn = compile_fn(code, function_name)
    if compiled_fn is None:
        return None

    sig = inspect.signature(compiled_fn)
    fn_accepts_p = "p" in sig.parameters

    default_params = {s.name: s.default for s in specs}
    try:
        if fn_accepts_p:
            default_fitness, _ = fitness_fn(functools.partial(compiled_fn, p=default_params))
        else:
            default_fitness, _ = fitness_fn(compiled_fn)
    except Exception:
        return None

    maximize = direction == "maximize"
    t0 = time.monotonic()

    # Phase 1: Broad Sobol sweep
    samples = _sobol_samples(specs, cfg.n_samples)
    # Include defaults in the sweep
    samples[0] = dict(default_params)

    # Evaluate all samples — use process pool for CPU-bound function calls
    results: list[tuple[float, dict[str, float]]] = []

    # For cheap evals, just run sequentially (process overhead > eval cost)
    eval_t0 = time.monotonic()
    if fn_accepts_p:
        test_fitness, _ = fitness_fn(functools.partial(compiled_fn, p=default_params))
    else:
        test_fitness, _ = fitness_fn(compiled_fn)
    eval_cost = time.monotonic() - eval_t0

    if eval_cost < 0.005:
        # Cheap eval (<5ms): run all sequentially, no process overhead
        for params in samples:
            if time.monotonic() - t0 > cfg.timeout_s:
                break
            try:
                if fn_accepts_p:
                    fn = functools.partial(compiled_fn, p=params)
                else:
                    fn = compiled_fn
                fitness, _ = fitness_fn(fn)
                results.append((fitness, params))
            except Exception:
                worst = float("-inf") if maximize else float("inf")
                results.append((worst, params))
    else:
        # Expensive eval: reduce samples and parallelize with fork
        import multiprocessing
        ctx = multiprocessing.get_context("fork")
        # Fewer samples for expensive evals — quality over quantity
        samples = samples[:min(len(samples), 50)]

        def _worker(params_chunk, result_list):
            """Evaluate a chunk of params in a forked process."""
            for params in params_chunk:
                try:
                    fn_local = compile_fn(code, function_name)
                    if fn_local is None:
                        continue
                    sig_local = inspect.signature(fn_local)
                    if "p" in sig_local.parameters:
                        fn_local = functools.partial(fn_local, p=params)
                    fitness, _ = fitness_fn(fn_local)
                    result_list.append((fitness, params))
                except Exception:
                    pass

        # Split samples across workers
        n_workers = min(cfg.max_workers, len(samples))
        chunk_size = max(1, len(samples) // n_workers)
        chunks = [samples[i:i + chunk_size] for i in range(0, len(samples), chunk_size)]

        manager = ctx.Manager()
        shared_results = manager.list()

        processes = []
        for chunk in chunks:
            p = ctx.Process(target=_worker, args=(chunk, shared_results))
            p.start()
            processes.append(p)

        for p in processes:
            p.join(timeout=cfg.timeout_s)
            if p.is_alive():
                p.kill()
                p.join(timeout=2)

        results = list(shared_results)

    if not results:
        return None

    # Sort by fitness
    results.sort(key=lambda x: x[0], reverse=maximize)

    # Phase 2: Refine top-K with local perturbation
    best_fitness, best_params = results[0]
    top_k = results[:cfg.refine_top_k]

    for top_fitness, top_params in top_k:
        if time.monotonic() - t0 > cfg.timeout_s:
            break

        # Generate local perturbations around this point
        for _ in range(cfg.refine_samples // cfg.refine_top_k):
            if time.monotonic() - t0 > cfg.timeout_s:
                break

            perturbed = {}
            for spec in specs:
                base = top_params[spec.name]
                # Perturb by ±10% of range
                range_size = (spec.high - spec.low) * 0.1
                import random
                val = base + random.uniform(-range_size, range_size)
                val = max(spec.low, min(spec.high, val))
                if spec.is_int:
                    val = int(round(val))
                perturbed[spec.name] = val

            try:
                if fn_accepts_p:
                    fn = functools.partial(compiled_fn, p=perturbed)
                else:
                    fn = compiled_fn
                fitness, _ = fitness_fn(fn)
                if (fitness > best_fitness) == maximize:
                    best_fitness = fitness
                    best_params = perturbed
            except Exception:
                pass

    total_evals = len(results) + cfg.refine_samples

    return TuningResult(
        best_fitness=best_fitness,
        best_params=best_params,
        n_trials=total_evals,
        default_fitness=default_fitness,
        improvement=best_fitness - default_fitness,
    )


# ---------------------------------------------------------------------------
# GPU batch tour-length computation (TSP-specific optimization)
# ---------------------------------------------------------------------------

def batch_tour_lengths_gpu(
    points_list: list[list[tuple[float, float]]],
    tours_list: list[list[int]],
) -> list[float]:
    """Compute tour lengths for multiple tours on GPU.

    Args:
        points_list: List of point sets (one per instance)
        tours_list: List of tours (one per instance, same ordering as points_list)

    Returns:
        List of tour lengths
    """
    if not GPU_AVAILABLE:
        # CPU fallback
        lengths = []
        for points, tour in zip(points_list, tours_list):
            n = len(tour)
            total = 0.0
            for i in range(n):
                x1, y1 = points[tour[i]]
                x2, y2 = points[tour[(i + 1) % n]]
                total += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            lengths.append(total)
        return lengths

    device = torch.device("cuda")
    lengths = []

    for points, tour in zip(points_list, tours_list):
        n = len(tour)
        # Build coordinate tensors on GPU
        pts = torch.tensor(points, dtype=torch.float32, device=device)
        idx = torch.tensor(tour, dtype=torch.long, device=device)
        idx_next = torch.roll(idx, -1)

        # Vectorized distance computation
        p1 = pts[idx]      # (n, 2)
        p2 = pts[idx_next]  # (n, 2)
        diffs = p1 - p2
        dists = torch.sqrt((diffs * diffs).sum(dim=1))
        lengths.append(dists.sum().item())

    return lengths


def batch_eval_tours_gpu(
    points_batch: list[list[tuple[float, float]]],
    tours_batch: list[list[list[int]]],
    baseline_lengths: list[float],
) -> list[float]:
    """Evaluate multiple sets of tours against baseline on GPU.

    Args:
        points_batch: Test instances (list of point sets)
        tours_batch: For each candidate, a list of tours (one per instance)
        baseline_lengths: Baseline tour length per instance

    Returns:
        Fitness score per candidate (avg baseline/tour_length ratio)
    """
    if not GPU_AVAILABLE:
        # CPU fallback
        fitnesses = []
        for tours in tours_batch:
            ratios = []
            for points, tour, baseline in zip(points_batch, tours, baseline_lengths):
                n = len(tour)
                length = sum(
                    math.sqrt(
                        (points[tour[i]][0] - points[tour[(i + 1) % n]][0]) ** 2
                        + (points[tour[i]][1] - points[tour[(i + 1) % n]][1]) ** 2
                    )
                    for i in range(n)
                )
                ratios.append(baseline / length if length > 0 else 0)
            fitnesses.append(sum(ratios) / len(ratios) if ratios else 0)
        return fitnesses

    device = torch.device("cuda")

    # Pre-compute point tensors on GPU
    gpu_points = []
    for points in points_batch:
        gpu_points.append(torch.tensor(points, dtype=torch.float32, device=device))
    baselines_t = torch.tensor(baseline_lengths, dtype=torch.float32, device=device)

    fitnesses = []
    for tours in tours_batch:
        ratios = []
        for inst_idx, (pts_gpu, tour) in enumerate(zip(gpu_points, tours)):
            n = len(tour)
            idx = torch.tensor(tour, dtype=torch.long, device=device)
            idx_next = torch.roll(idx, -1)
            diffs = pts_gpu[idx] - pts_gpu[idx_next]
            length = torch.sqrt((diffs * diffs).sum(dim=1)).sum()
            ratio = baselines_t[inst_idx] / length if length > 0 else 0
            ratios.append(ratio.item() if isinstance(ratio, torch.Tensor) else ratio)
        fitnesses.append(sum(ratios) / len(ratios) if ratios else 0)

    return fitnesses
