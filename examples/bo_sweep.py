"""BO-only SVG sweep: generate experience data without LLM.

Explores SVG parameter space using Optuna + structural mutations (add/remove
shapes, fix/release parameters). Logs every trial as an experience tuple
for the meta-learning memory bank.

Actions:
  - add_shape: add a new SVG element (circle, rect, ellipse, line)
  - remove_shape: remove an existing element
  - fix_param: lock a parameter (BO won't touch it)
  - release_param: unlock a parameter for BO optimization

Usage:
    python examples/bo_sweep.py                          # default: glow target
    EVOL_TARGET=face python examples/bo_sweep.py         # face target
    EVOL_TARGET=all python examples/bo_sweep.py          # sweep all targets
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import cairosvg
import optuna
from PIL import Image

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Shape definitions ──────────────────────────────────────────────

SHAPE_TYPES = ["circle", "rect", "ellipse", "line"]

SHAPE_PARAM_DEFS = {
    "circle": {
        "cx": (0.0, 64.0), "cy": (0.0, 64.0), "r": (1.0, 40.0),
        "fill_r": (0, 255), "fill_g": (0, 255), "fill_b": (0, 255),
        "opacity": (0.1, 1.0),
    },
    "rect": {
        "x": (0.0, 64.0), "y": (0.0, 64.0),
        "w": (1.0, 64.0), "h": (1.0, 64.0),
        "fill_r": (0, 255), "fill_g": (0, 255), "fill_b": (0, 255),
        "opacity": (0.1, 1.0),
    },
    "ellipse": {
        "cx": (0.0, 64.0), "cy": (0.0, 64.0),
        "rx": (1.0, 40.0), "ry": (1.0, 40.0),
        "fill_r": (0, 255), "fill_g": (0, 255), "fill_b": (0, 255),
        "opacity": (0.1, 1.0),
    },
    "line": {
        "x1": (0.0, 64.0), "y1": (0.0, 64.0),
        "x2": (0.0, 64.0), "y2": (0.0, 64.0),
        "stroke_r": (0, 255), "stroke_g": (0, 255), "stroke_b": (0, 255),
        "stroke_width": (1.0, 10.0), "opacity": (0.1, 1.0),
    },
}


@dataclass
class Shape:
    type: str
    params: dict[str, float]
    fixed: dict[str, bool] = field(default_factory=dict)

    def to_svg(self) -> str:
        p = self.params
        op = p.get("opacity", 1.0)
        if self.type == "circle":
            return (f'<circle cx="{p["cx"]:.1f}" cy="{p["cy"]:.1f}" '
                    f'r="{p["r"]:.1f}" '
                    f'fill="rgb({int(p["fill_r"])},{int(p["fill_g"])},{int(p["fill_b"])})" '
                    f'opacity="{op:.2f}"/>')
        elif self.type == "rect":
            return (f'<rect x="{p["x"]:.1f}" y="{p["y"]:.1f}" '
                    f'width="{p["w"]:.1f}" height="{p["h"]:.1f}" '
                    f'fill="rgb({int(p["fill_r"])},{int(p["fill_g"])},{int(p["fill_b"])})" '
                    f'opacity="{op:.2f}"/>')
        elif self.type == "ellipse":
            return (f'<ellipse cx="{p["cx"]:.1f}" cy="{p["cy"]:.1f}" '
                    f'rx="{p["rx"]:.1f}" ry="{p["ry"]:.1f}" '
                    f'fill="rgb({int(p["fill_r"])},{int(p["fill_g"])},{int(p["fill_b"])})" '
                    f'opacity="{op:.2f}"/>')
        elif self.type == "line":
            return (f'<line x1="{p["x1"]:.1f}" y1="{p["y1"]:.1f}" '
                    f'x2="{p["x2"]:.1f}" y2="{p["y2"]:.1f}" '
                    f'stroke="rgb({int(p["stroke_r"])},{int(p["stroke_g"])},{int(p["stroke_b"])})" '
                    f'stroke-width="{p["stroke_width"]:.1f}" opacity="{op:.2f}"/>')
        return ""


@dataclass
class Genome:
    shapes: list[Shape] = field(default_factory=list)
    bg_r: float = 128.0
    bg_g: float = 128.0
    bg_b: float = 128.0
    bg_fixed: bool = False

    def to_svg(self, w: int = 64, h: int = 64) -> str:
        bg = f'rgb({int(self.bg_r)},{int(self.bg_g)},{int(self.bg_b)})'
        parts = [f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">',
                 f'<rect width="{w}" height="{h}" fill="{bg}"/>']
        for s in self.shapes:
            parts.append(s.to_svg())
        parts.append("</svg>")
        return "".join(parts)

    def clone(self) -> Genome:
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return {
            "bg": [self.bg_r, self.bg_g, self.bg_b],
            "bg_fixed": self.bg_fixed,
            "shapes": [
                {"type": s.type, "params": s.params, "fixed": s.fixed}
                for s in self.shapes
            ],
        }

    def shape_count(self) -> int:
        return len(self.shapes)


# ── Rendering & fitness ────────────────────────────────────────────

def svg_to_pixels(svg: str, w: int = 64, h: int = 64) -> list[tuple[int, int, int]] | None:
    try:
        png = cairosvg.svg2png(bytestring=svg.encode(), output_width=w, output_height=h)
        img = Image.open(io.BytesIO(png)).convert("RGB")
        return [img.getpixel((x, y)) for y in range(h) for x in range(w)]
    except Exception:
        return None


def compute_fitness(pixels: list, target: list, w: int, h: int, shape_count: int):
    n = w * h
    total_mse = 0.0
    half_w, half_h = w // 2, h // 2
    regions = [0.0, 0.0, 0.0, 0.0]
    region_counts = [0, 0, 0, 0]

    for i in range(n):
        r, g, b = pixels[i]
        tr, tg, tb = target[i]
        err = (r - tr) ** 2 + (g - tg) ** 2 + (b - tb) ** 2
        total_mse += err
        x, y = i % w, i // w
        qi = (0 if y < half_h else 2) + (0 if x < half_w else 1)
        regions[qi] += err
        region_counts[qi] += 1

    norm_mse = total_mse / (n * 3 * 255 * 255)
    similarity = 1.0 / (1.0 + norm_mse * 1000)
    fitness = similarity / (1.0 + 0.002 * shape_count)

    max_err = 3 * 255 * 255
    region_errors = [
        regions[i] / (region_counts[i] * max_err) if region_counts[i] > 0 else 1.0
        for i in range(4)
    ]
    return fitness, similarity, total_mse / n, region_errors


# ── Actions ────────────────────────────────────────────────────────

def random_shape(shape_type: str | None = None) -> Shape:
    if shape_type is None:
        shape_type = random.choice(SHAPE_TYPES)
    param_defs = SHAPE_PARAM_DEFS[shape_type]
    params = {}
    for name, (lo, hi) in param_defs.items():
        if isinstance(lo, int):
            params[name] = float(random.randint(lo, hi))
        else:
            params[name] = random.uniform(lo, hi)
    fixed = {name: False for name in param_defs}
    return Shape(type=shape_type, params=params, fixed=fixed)


def action_add_shape(genome: Genome) -> tuple[Genome, dict]:
    g = genome.clone()
    if len(g.shapes) >= 15:
        return g, {"action": "add_shape", "success": False, "reason": "max_shapes"}
    shape = random_shape()
    g.shapes.append(shape)
    return g, {"action": "add_shape", "success": True, "shape_type": shape.type,
               "shape_idx": len(g.shapes) - 1}


def action_remove_shape(genome: Genome) -> tuple[Genome, dict]:
    g = genome.clone()
    if len(g.shapes) == 0:
        return g, {"action": "remove_shape", "success": False, "reason": "no_shapes"}
    idx = random.randint(0, len(g.shapes) - 1)
    removed = g.shapes.pop(idx)
    return g, {"action": "remove_shape", "success": True, "shape_type": removed.type,
               "shape_idx": idx}


def action_fix_param(genome: Genome) -> tuple[Genome, dict]:
    g = genome.clone()
    released = []
    for si, s in enumerate(g.shapes):
        for pname, is_fixed in s.fixed.items():
            if not is_fixed:
                released.append((si, pname))
    if not released:
        return g, {"action": "fix_param", "success": False, "reason": "all_fixed"}
    si, pname = random.choice(released)
    g.shapes[si].fixed[pname] = True
    return g, {"action": "fix_param", "success": True, "shape_idx": si, "param": pname}


def action_release_param(genome: Genome) -> tuple[Genome, dict]:
    g = genome.clone()
    fixed = []
    for si, s in enumerate(g.shapes):
        for pname, is_fixed in s.fixed.items():
            if is_fixed:
                fixed.append((si, pname))
    if not fixed:
        return g, {"action": "release_param", "success": False, "reason": "none_fixed"}
    si, pname = random.choice(fixed)
    g.shapes[si].fixed[pname] = False
    return g, {"action": "release_param", "success": True, "shape_idx": si, "param": pname}


ACTIONS = [action_add_shape, action_remove_shape, action_fix_param, action_release_param]
ACTION_WEIGHTS = [0.35, 0.15, 0.25, 0.25]


# ── BO optimization ───────────────────────────────────────────────

def optimize_params(genome: Genome, target: list, w: int, h: int,
                    n_trials: int = 30) -> tuple[Genome, float, list[dict]]:
    """Run Optuna on released parameters. Returns optimized genome + fitness + trial log."""
    g = genome.clone()

    # Collect tunable parameters
    tunable = []
    # Background
    if not g.bg_fixed:
        tunable.append(("bg", "r", 0, 255))
        tunable.append(("bg", "g", 0, 255))
        tunable.append(("bg", "b", 0, 255))
    # Shapes
    for si, s in enumerate(g.shapes):
        param_defs = SHAPE_PARAM_DEFS[s.type]
        for pname, is_fixed in s.fixed.items():
            if not is_fixed and pname in param_defs:
                lo, hi = param_defs[pname]
                tunable.append((f"s{si}", pname, lo, hi))

    if not tunable:
        # Nothing to tune, just evaluate
        svg = g.to_svg(w, h)
        pixels = svg_to_pixels(svg, w, h)
        if pixels is None:
            return g, 0.0, []
        fitness, _, _, _ = compute_fitness(pixels, target, w, h, g.shape_count())
        return g, fitness, []

    trial_log = []

    def objective(trial):
        for key, pname, lo, hi in tunable:
            if isinstance(lo, int):
                val = float(trial.suggest_int(f"{key}_{pname}", lo, hi))
            else:
                val = trial.suggest_float(f"{key}_{pname}", lo, hi)
            if key == "bg":
                if pname == "r": g.bg_r = val
                elif pname == "g": g.bg_g = val
                elif pname == "b": g.bg_b = val
            else:
                si = int(key[1:])
                g.shapes[si].params[pname] = val

        svg = g.to_svg(w, h)
        pixels = svg_to_pixels(svg, w, h)
        if pixels is None:
            return 0.0
        fitness, sim, mse, regions = compute_fitness(pixels, target, w, h, g.shape_count())

        trial_log.append({
            "trial": trial.number,
            "fitness": fitness,
            "similarity": sim,
            "mse": mse,
        })
        return fitness

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random.randint(0, 2**31)))
    study.optimize(objective, n_trials=n_trials, timeout=30)

    # Apply best params
    for key, pname, lo, hi in tunable:
        param_key = f"{key}_{pname}"
        if param_key in study.best_params:
            val = study.best_params[param_key]
            if key == "bg":
                if pname == "r": g.bg_r = val
                elif pname == "g": g.bg_g = val
                elif pname == "b": g.bg_b = val
            else:
                si = int(key[1:])
                g.shapes[si].params[pname] = float(val)

    return g, study.best_value, trial_log


# ── Main sweep loop ───────────────────────────────────────────────

def load_target(name: str):
    import importlib
    m = importlib.import_module(f"examples.target_{name}")
    return m.TARGET, m.W, m.H


def sweep(target_name: str, n_steps: int = 100, bo_trials: int = 30,
          out_dir: str | None = None):
    """Run BO sweep on a target. Returns experience log."""
    target, w, h = load_target(target_name)

    if out_dir is None:
        out_dir = f"runs/bo_sweep_{target_name}_{int(time.time())}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(out_dir) / "experiences.jsonl"

    # Start with a background-only genome
    genome = Genome(bg_r=128, bg_g=128, bg_b=128)

    # Initial fitness
    svg = genome.to_svg(w, h)
    pixels = svg_to_pixels(svg, w, h)
    if pixels:
        fitness, sim, mse, regions = compute_fitness(pixels, target, w, h, 0)
    else:
        fitness, sim, mse, regions = 0.0, 0.0, 255**2, [1.0]*4

    best_fitness = fitness
    best_genome = genome.clone()
    experiences = []

    print(f"[{target_name}] Starting sweep: {n_steps} steps, {bo_trials} BO trials each")
    print(f"[{target_name}] Initial fitness: {fitness:.4f}")

    for step in range(n_steps):
        # Snapshot state before action
        o_t = {
            "fitness": fitness,
            "similarity": sim,
            "mse": mse,
            "region_errors": regions,
            "shape_count": genome.shape_count(),
            "genome": genome.to_dict(),
        }

        # Pick and apply a random action
        action_fn = random.choices(ACTIONS, weights=ACTION_WEIGHTS, k=1)[0]
        new_genome, action_info = action_fn(genome)

        if not action_info.get("success", False):
            continue

        # Optimize released parameters
        new_genome, new_fitness, trial_log = optimize_params(
            new_genome, target, w, h, n_trials=bo_trials,
        )

        # Compute new state
        new_svg = new_genome.to_svg(w, h)
        new_pixels = svg_to_pixels(new_svg, w, h)
        if new_pixels:
            new_fitness, new_sim, new_mse, new_regions = compute_fitness(
                new_pixels, target, w, h, new_genome.shape_count(),
            )
        else:
            new_fitness, new_sim, new_mse, new_regions = 0.0, 0.0, 255**2, [1.0]*4

        reward = new_fitness - fitness

        # Snapshot state after action
        o_t1 = {
            "fitness": new_fitness,
            "similarity": new_sim,
            "mse": new_mse,
            "region_errors": new_regions,
            "shape_count": new_genome.shape_count(),
            "genome": new_genome.to_dict(),
        }

        # Log experience tuple: (O_t, A, O_t+1, R, T)
        experience = {
            "o_t": o_t,
            "action": action_info,
            "o_t1": o_t1,
            "reward": reward,
            "target": target_name,
            "step": step,
            "bo_trials": len(trial_log),
            "timestamp": time.time(),
        }
        experiences.append(experience)

        # Write immediately
        with open(log_path, "a") as f:
            f.write(json.dumps(experience) + "\n")

        # Accept if improved (greedy), or occasionally accept worse (exploration)
        accept = reward > 0 or random.random() < 0.1
        if accept:
            genome = new_genome
            fitness, sim, mse, regions = new_fitness, new_sim, new_mse, new_regions

        if new_fitness > best_fitness:
            best_fitness = new_fitness
            best_genome = new_genome.clone()

        if (step + 1) % 10 == 0:
            print(f"[{target_name}] Step {step+1}/{n_steps}: "
                  f"fitness={fitness:.4f} best={best_fitness:.4f} "
                  f"shapes={genome.shape_count()} action={action_info['action']}")

    # Save best SVG
    best_svg = best_genome.to_svg(w, h)
    with open(Path(out_dir) / "best.svg", "w") as f:
        f.write(best_svg)
    png = cairosvg.svg2png(bytestring=best_svg.encode(), output_width=w, output_height=h)
    with open(Path(out_dir) / "best.png", "wb") as f:
        f.write(png)

    print(f"[{target_name}] Done! Best fitness: {best_fitness:.4f}, "
          f"shapes: {best_genome.shape_count()}, "
          f"experiences: {len(experiences)}")
    print(f"[{target_name}] Saved to {out_dir}/")

    return experiences


if __name__ == "__main__":
    target_arg = os.environ.get("EVOL_TARGET", "glow")
    n_steps = int(os.environ.get("BO_STEPS", "100"))
    bo_trials = int(os.environ.get("BO_TRIALS", "30"))

    if target_arg == "all":
        targets = ["glow", "blocks", "face", "sunset", "rings"]
    else:
        targets = [target_arg]

    for t in targets:
        sweep(t, n_steps=n_steps, bo_trials=bo_trials)
