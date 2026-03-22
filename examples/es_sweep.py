"""ES-based SVG evolution: population-level ES + CMA-ES parameter tuning.

Two-level evolutionary strategy:
  1. Population level: tournament selection, structural mutation, crossover
  2. Parameter level: CMA-ES tunes continuous params per genome

No LLM needed. Logs experience tuples for meta-learning memory bank.

Usage:
    python examples/es_sweep.py                              # default: glow
    EVOL_TARGET=face python examples/es_sweep.py             # face target
    EVOL_TARGET=all ES_GPU=1 python examples/es_sweep.py     # all targets, GPU
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from bo_sweep import (
    Genome, Shape, SHAPE_TYPES, SHAPE_WEIGHTS, SHAPE_PARAM_DEFS,
    random_shape, load_target, svg_to_pixels, compute_fitness,
    _eval_genome, _collect_tunable, _apply_trial_params,
    ALL_TARGETS, _genome_from_dict, hindsight_relabel,
)

# ── Population-level operations ────────────────────────────────────


def tournament_select(population: list[dict], k: int = 3) -> dict:
    """Select best individual from k random candidates."""
    candidates = random.sample(population, min(k, len(population)))
    return max(candidates, key=lambda ind: ind["fitness"])


def crossover(parent_a: Genome, parent_b: Genome) -> Genome:
    """Combine shapes from two parents."""
    child = Genome(
        bg_r=(parent_a.bg_r + parent_b.bg_r) / 2,
        bg_g=(parent_a.bg_g + parent_b.bg_g) / 2,
        bg_b=(parent_a.bg_b + parent_b.bg_b) / 2,
    )
    # Take shapes from both parents randomly
    all_shapes = parent_a.shapes + parent_b.shapes
    if all_shapes:
        n = random.randint(1, min(len(all_shapes), 15))
        child.shapes = [copy.deepcopy(s) for s in random.sample(all_shapes, n)]
        # Release all params for CMA-ES tuning
        for s in child.shapes:
            s.fixed = {k: False for k in s.fixed}
    return child


def mutate_structure(genome: Genome) -> tuple[Genome, dict]:
    """Apply a random structural mutation. Returns (new_genome, mutation_info)."""
    g = genome.clone()
    mutation_type = random.choices(
        ["add", "remove", "replace", "duplicate", "reorder"],
        weights=[0.30, 0.15, 0.20, 0.20, 0.15],
        k=1,
    )[0]

    if mutation_type == "add" and len(g.shapes) < 15:
        shape = random_shape()
        g.shapes.append(shape)
        return g, {"mutation": "add", "shape_type": shape.type}

    elif mutation_type == "remove" and len(g.shapes) > 0:
        idx = random.randint(0, len(g.shapes) - 1)
        removed = g.shapes.pop(idx)
        return g, {"mutation": "remove", "shape_type": removed.type, "idx": idx}

    elif mutation_type == "replace" and len(g.shapes) > 0:
        idx = random.randint(0, len(g.shapes) - 1)
        old_type = g.shapes[idx].type
        g.shapes[idx] = random_shape()
        return g, {"mutation": "replace", "old": old_type, "new": g.shapes[idx].type}

    elif mutation_type == "duplicate" and len(g.shapes) > 0 and len(g.shapes) < 15:
        idx = random.randint(0, len(g.shapes) - 1)
        new_shape = copy.deepcopy(g.shapes[idx])
        # Perturb position
        for pname in ["cx", "cy", "x", "y", "x1", "y1"]:
            if pname in new_shape.params:
                new_shape.params[pname] += random.gauss(0, 5)
        new_shape.fixed = {k: False for k in new_shape.fixed}
        g.shapes.append(new_shape)
        return g, {"mutation": "duplicate", "source": idx}

    elif mutation_type == "reorder" and len(g.shapes) >= 2:
        i = random.randint(0, len(g.shapes) - 1)
        j = random.randint(0, len(g.shapes) - 1)
        g.shapes[i], g.shapes[j] = g.shapes[j], g.shapes[i]
        return g, {"mutation": "reorder", "i": i, "j": j}

    # Fallback: add a shape
    shape = random_shape()
    g.shapes.append(shape)
    return g, {"mutation": "add_fallback", "shape_type": shape.type}


# ── CMA-ES parameter optimization ─────────────────────────────────


def cmaes_optimize(genome: Genome, target, w: int, h: int,
                   n_trials: int = 30, gpu_rast=None,
                   target_gpu=None) -> tuple[Genome, float]:
    """Optimize released parameters using CMA-ES."""
    g = genome.clone()
    tunable = _collect_tunable(g)

    if not tunable:
        fit, _, _, _ = _eval_genome(g, target, w, h, gpu_rast, target_gpu)
        return g, fit

    def objective(trial):
        _apply_trial_params(g, tunable, trial, source="trial")
        fitness, _, _, _ = _eval_genome(g, target, w, h, gpu_rast, target_gpu)
        return fitness

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.CmaEsSampler(seed=random.randint(0, 2**31)),
    )
    study.optimize(objective, n_trials=n_trials, timeout=30)

    _apply_trial_params(g, tunable, study.best_params, source="dict")
    return g, study.best_value


# ── Main ES loop ──────────────────────────────────────────────────


def es_sweep(
    target_name: str,
    n_generations: int = 50,
    population_size: int = 20,
    elite_count: int = 3,
    cmaes_trials: int = 20,
    crossover_rate: float = 0.3,
    use_gpu: bool = False,
    out_dir: str | None = None,
) -> list[dict]:
    """Run ES sweep on a target. Returns experience log."""
    target, w, h = load_target(target_name)

    gpu_rast = None
    target_gpu = None
    if use_gpu:
        from gpu_rasterizer import GPURasterizer
        gpu_rast = GPURasterizer(w, h, "cuda")
        target_gpu = gpu_rast.target_from_pixels(target)

    if out_dir is None:
        gpu_tag = "_gpu" if use_gpu else ""
        out_dir = f"runs/es_sweep_{target_name}{gpu_tag}_{int(time.time())}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(out_dir) / "experiences.jsonl"

    # Initialize population with diverse seeds
    population = []
    for _ in range(population_size):
        g = Genome(
            bg_r=random.uniform(0, 255),
            bg_g=random.uniform(0, 255),
            bg_b=random.uniform(0, 255),
        )
        n_shapes = random.randint(1, 5)
        for _ in range(n_shapes):
            g.shapes.append(random_shape())

        # Quick CMA-ES tune
        g, fit = cmaes_optimize(g, target, w, h, n_trials=cmaes_trials,
                                gpu_rast=gpu_rast, target_gpu=target_gpu)
        _, sim, mse, regions = _eval_genome(g, target, w, h, gpu_rast, target_gpu)
        population.append({
            "genome": g,
            "fitness": fit,
            "similarity": sim,
            "mse": mse,
            "regions": regions,
        })

    population.sort(key=lambda x: x["fitness"], reverse=True)
    best_ever = population[0]["fitness"]
    best_genome = population[0]["genome"].clone()

    mode = "GPU" if use_gpu else "CPU"
    print(f"[{target_name}] ES sweep ({mode}): pop={population_size}, "
          f"gens={n_generations}, cmaes={cmaes_trials} trials",
          flush=True)
    print(f"[{target_name}] Initial best: {best_ever:.4f}, "
          f"avg: {sum(p['fitness'] for p in population)/len(population):.4f}",
          flush=True)

    experiences = []

    for gen in range(n_generations):
        new_population = []

        # Keep elites
        elites = population[:elite_count]
        new_population.extend(copy.deepcopy(elites))

        # Generate offspring
        while len(new_population) < population_size:
            # Select parents
            parent_a = tournament_select(population)

            if random.random() < crossover_rate:
                # Crossover
                parent_b = tournament_select(population)
                child_genome = crossover(parent_a["genome"], parent_b["genome"])
                action_info = {
                    "action": "crossover",
                    "parent_a_fitness": parent_a["fitness"],
                    "parent_b_fitness": parent_b["fitness"],
                    "parent_a_shapes": parent_a["genome"].shape_count(),
                    "parent_b_shapes": parent_b["genome"].shape_count(),
                }
            else:
                # Structural mutation
                child_genome, mut_info = mutate_structure(parent_a["genome"])
                action_info = {"action": "mutation", **mut_info,
                               "parent_fitness": parent_a["fitness"]}

            # Snapshot pre-optimization state
            pre_fit, pre_sim, pre_mse, pre_regions = _eval_genome(
                child_genome, target, w, h, gpu_rast, target_gpu)

            o_t = {
                "fitness": parent_a["fitness"],
                "similarity": parent_a["similarity"],
                "mse": parent_a["mse"],
                "region_errors": parent_a["regions"],
                "shape_count": parent_a["genome"].shape_count(),
                "genome": parent_a["genome"].to_dict(),
            }

            # CMA-ES parameter optimization
            child_genome, child_fit = cmaes_optimize(
                child_genome, target, w, h, n_trials=cmaes_trials,
                gpu_rast=gpu_rast, target_gpu=target_gpu,
            )
            child_fit, child_sim, child_mse, child_regions = _eval_genome(
                child_genome, target, w, h, gpu_rast, target_gpu)

            o_t1 = {
                "fitness": child_fit,
                "similarity": child_sim,
                "mse": child_mse,
                "region_errors": child_regions,
                "shape_count": child_genome.shape_count(),
                "genome": child_genome.to_dict(),
            }

            reward = child_fit - parent_a["fitness"]

            experience = {
                "o_t": o_t, "action": action_info, "o_t1": o_t1,
                "reward": reward, "target": target_name,
                "generation": gen, "timestamp": time.time(),
            }
            experiences.append(experience)
            with open(log_path, "a") as f:
                f.write(json.dumps(experience) + "\n")

            new_population.append({
                "genome": child_genome,
                "fitness": child_fit,
                "similarity": child_sim,
                "mse": child_mse,
                "regions": child_regions,
            })

        # Sort and truncate
        new_population.sort(key=lambda x: x["fitness"], reverse=True)
        population = new_population[:population_size]

        if population[0]["fitness"] > best_ever:
            best_ever = population[0]["fitness"]
            best_genome = population[0]["genome"].clone()
            # Checkpoint
            try:
                import cairosvg
                best_svg = best_genome.to_svg(w, h)
                png = cairosvg.svg2png(bytestring=best_svg.encode(),
                                       output_width=w, output_height=h)
                with open(Path(out_dir) / "best.png", "wb") as f:
                    f.write(png)
                with open(Path(out_dir) / "best.svg", "w") as f:
                    f.write(best_svg)
            except Exception:
                pass

        avg_fit = sum(p["fitness"] for p in population) / len(population)
        diversity = len(set(p["genome"].shape_count() for p in population))

        if (gen + 1) % 5 == 0 or gen == 0:
            print(f"[{target_name}] Gen {gen+1}/{n_generations}: "
                  f"best={best_ever:.4f} avg={avg_fit:.4f} "
                  f"diversity={diversity} "
                  f"shapes={population[0]['genome'].shape_count()}",
                  flush=True)

    # Final save
    try:
        import cairosvg
        best_svg = best_genome.to_svg(w, h)
        with open(Path(out_dir) / "best.svg", "w") as f:
            f.write(best_svg)
        png = cairosvg.svg2png(bytestring=best_svg.encode(),
                               output_width=w, output_height=h)
        with open(Path(out_dir) / "best.png", "wb") as f:
            f.write(png)
    except Exception:
        pass

    print(f"[{target_name}] Done! Best: {best_ever:.4f}, "
          f"shapes: {best_genome.shape_count()}, "
          f"experiences: {len(experiences)}", flush=True)
    print(f"[{target_name}] Saved to {out_dir}/", flush=True)

    return experiences


if __name__ == "__main__":
    target_arg = os.environ.get("EVOL_TARGET", "glow")
    n_gens = int(os.environ.get("ES_GENS", "50"))
    pop_size = int(os.environ.get("ES_POP", "20"))
    elite = int(os.environ.get("ES_ELITE", "3"))
    cmaes_trials = int(os.environ.get("ES_CMAES_TRIALS", "20"))
    use_gpu = os.environ.get("ES_GPU", "0") == "1"
    do_hindsight = os.environ.get("ES_HINDSIGHT", "1") == "1"

    if target_arg == "all":
        targets = ALL_TARGETS
    else:
        targets = [target_arg]

    all_experiences = {}
    for t in targets:
        exps = es_sweep(t, n_generations=n_gens, population_size=pop_size,
                        elite_count=elite, cmaes_trials=cmaes_trials,
                        use_gpu=use_gpu)
        all_experiences[t] = exps

    # Hindsight relabeling
    if do_hindsight and len(targets) > 1:
        print("\n=== Hindsight Experience Replay ===", flush=True)
        for t, exps in all_experiences.items():
            hindsight = hindsight_relabel(exps, t, use_gpu=use_gpu)
            if hindsight:
                run_dirs = sorted(Path("runs").glob(f"es_sweep_{t}*"),
                                  key=lambda p: p.stat().st_mtime, reverse=True)
                if run_dirs:
                    hp = run_dirs[0] / "hindsight.jsonl"
                    with open(hp, "w") as f:
                        for h in hindsight:
                            f.write(json.dumps(h) + "\n")
                    positive = sum(1 for h in hindsight if h["reward"] > 0)
                    print(f"[{t}] {len(hindsight)} hindsight ({positive} positive) -> {hp}",
                          flush=True)

        total_orig = sum(len(e) for e in all_experiences.values())
        total_hs = 0
        for t in targets:
            dirs = sorted(Path("runs").glob(f"es_sweep_{t}*"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
            if dirs:
                hp = dirs[0] / "hindsight.jsonl"
                if hp.exists():
                    with open(hp) as f:
                        total_hs += sum(1 for _ in f)
        print(f"\n=== Total: {total_orig} original + {total_hs} hindsight "
              f"= {total_orig + total_hs} experiences ===", flush=True)
