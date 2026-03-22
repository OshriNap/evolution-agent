"""Example: evolve SVG art to match a target image.

Evolves a Python function that returns an SVG string. The SVG is rendered to
pixels and compared against a 64x64 target image. Fitness rewards pixel
similarity and penalizes shape count (fewer shapes = more elegant).

Usage:
    python examples/optimize_image.py                          # default target (glow)
    EVOL_TARGET=face python examples/optimize_image.py         # smiley face target
    EVOL_TARGET=blocks python examples/optimize_image.py       # color blocks target
    EVOL_TARGET=sunset python examples/optimize_image.py       # sunset target
    EVOL_TARGET=rings python examples/optimize_image.py        # concentric rings target
"""

from __future__ import annotations

import asyncio
import importlib
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import cairosvg
from PIL import Image

# Load target based on EVOL_TARGET env var (default: glow)
_target_name = os.environ.get("EVOL_TARGET", "glow")
_target_module = importlib.import_module(f"examples.target_{_target_name}")
TARGET = _target_module.TARGET
W = _target_module.W
H = _target_module.H
TARGET_DESC = getattr(_target_module, "DESCRIPTION", "")

TARGET_PIXELS = TARGET

# Error threshold: pixel within this squared-error is "ok"
ERROR_THRESHOLD = 50 * 50 * 3  # ~50 per channel

# Shape count penalty weight (higher = stronger pressure for fewer shapes)
SHAPE_PENALTY_WEIGHT = 0.002


def _svg_to_pixels(svg_string: str) -> list[tuple[int, int, int]] | None:
    """Render SVG string to a list of (r, g, b) pixel tuples."""
    try:
        png_data = cairosvg.svg2png(
            bytestring=svg_string.encode(),
            output_width=W,
            output_height=H,
        )
        img = Image.open(io.BytesIO(png_data)).convert("RGB")
        pixels = []
        for y in range(H):
            for x in range(W):
                pixels.append(img.getpixel((x, y)))
        return pixels
    except Exception:
        return None


def _count_shapes(svg_string: str) -> int:
    """Count SVG shape elements (rough heuristic)."""
    shape_tags = ["<circle", "<rect", "<ellipse", "<line", "<polygon",
                  "<polyline", "<path", "<text"]
    count = 0
    s = svg_string.lower()
    for tag in shape_tags:
        count += s.count(tag)
    return max(1, count)  # at least 1


def evaluate_svg(fn) -> tuple[float, dict]:
    """Evaluate an SVG generator function against the target.

    Calls fn(w, h) to get an SVG string, renders it, compares pixel-by-pixel.
    Fitness = similarity / (1 + shape_penalty).
    """
    try:
        svg_string = fn(W, H)
    except Exception:
        return 0.0, {
            "mse": 255.0 ** 2, "r_error": 255.0 ** 2,
            "g_error": 255.0 ** 2, "b_error": 255.0 ** 2,
            "pixel_coverage": 0.0, "region_errors": [1.0] * 4,
            "shape_count": 0, "svg_valid": False,
        }

    if not isinstance(svg_string, str) or "<svg" not in svg_string.lower():
        return 0.0, {
            "mse": 255.0 ** 2, "r_error": 255.0 ** 2,
            "g_error": 255.0 ** 2, "b_error": 255.0 ** 2,
            "pixel_coverage": 0.0, "region_errors": [1.0] * 4,
            "shape_count": 0, "svg_valid": False,
        }

    pixels = _svg_to_pixels(svg_string)
    if pixels is None or len(pixels) != W * H:
        return 0.0, {
            "mse": 255.0 ** 2, "r_error": 255.0 ** 2,
            "g_error": 255.0 ** 2, "b_error": 255.0 ** 2,
            "pixel_coverage": 0.0, "region_errors": [1.0] * 4,
            "shape_count": 0, "svg_valid": False,
        }

    # Single-pass pixel comparison with per-quadrant tracking
    total_mse = 0.0
    r_err = g_err = b_err = 0.0
    pixels_ok = 0
    n = W * H
    half_w, half_h = W // 2, H // 2
    regions = [0.0, 0.0, 0.0, 0.0]
    region_counts = [0, 0, 0, 0]

    for y in range(H):
        for x in range(W):
            idx = y * W + x
            r, g, b = pixels[idx]
            tr, tg, tb = TARGET_PIXELS[idx]
            dr, dg, db = r - tr, g - tg, b - tb
            pixel_err = dr * dr + dg * dg + db * db
            total_mse += pixel_err
            r_err += dr * dr
            g_err += dg * dg
            b_err += db * db

            if pixel_err < ERROR_THRESHOLD:
                pixels_ok += 1

            qi = (0 if y < half_h else 2) + (0 if x < half_w else 1)
            regions[qi] += pixel_err
            region_counts[qi] += 1

    normalized_mse = total_mse / (n * 3 * 255 * 255)
    similarity = 1.0 / (1.0 + normalized_mse * 1000)

    shape_count = _count_shapes(svg_string)
    fitness = similarity / (1.0 + SHAPE_PENALTY_WEIGHT * shape_count)

    max_region_err = 3 * 255 * 255
    region_errors = [
        regions[i] / (region_counts[i] * max_region_err) if region_counts[i] > 0 else 1.0
        for i in range(4)
    ]

    metrics = {
        "mse": total_mse / n,
        "r_error": r_err / n,
        "g_error": g_err / n,
        "b_error": b_err / n,
        "pixel_coverage": pixels_ok / n,
        "region_errors": region_errors,
        "shape_count": shape_count,
        "similarity": similarity,
        "svg_valid": True,
    }
    return fitness, metrics


FUNCTION_SPEC = f"""\
def generate_svg(w, h, p=None):
    # Generate an SVG image as a string that matches the target.
    #
    # Args:
    #     w, h: image dimensions (64x64)
    #     p: optional dict of tunable numeric parameters with defaults.
    #        Put all tunable constants in p so they can be auto-optimized.
    #        Example: if p is None: p = {{"cx": 32.0, "r": 20.0}}
    #
    # Returns:
    #     A string containing valid SVG markup.
    #     Must start with <svg> and end with </svg>.
    #
    # Available globals (already in scope, do NOT import):
    #     math (math.sin, math.cos, math.sqrt, math.exp, math.pi, etc.)
    #     range, len, min, max, int, float, abs, sum, sorted, round, pow, str
    #
{TARGET_DESC}
    #
    # SVG tips:
    #   - Use <radialGradient> for glow/circular effects
    #   - Use <linearGradient> for color transitions
    #   - Use <circle>, <rect>, <ellipse>, <path> for shapes
    #   - Use opacity and layering to blend shapes
    #   - Use <defs> for gradient definitions
    #   - Fewer shapes with gradients is better than many solid shapes
    #
    # FITNESS: pixel similarity to target MINUS penalty for shape count.
    # Fewer shapes = higher fitness at same similarity. Use gradients!
"""

SEED_FLAT = """\
def generate_svg(w, h, p=None):
    # Flat background color (average of target)
    if p is None:
        p = {"r": 83.0, "g": 50.0, "b": 135.0}
    r, g, b = int(p["r"]), int(p["g"]), int(p["b"])
    return f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg"><rect width="{w}" height="{h}" fill="rgb({r},{g},{b})"/></svg>'
"""

SEED_RADIAL = """\
def generate_svg(w, h, p=None):
    # Simple radial glow: dark background + bright center circle
    if p is None:
        p = {"bg_r": 67.0, "bg_g": 33.0, "bg_b": 132.0,
             "c_r": 200.0, "c_g": 170.0, "c_b": 180.0,
             "cx": 32.0, "cy": 32.0, "radius": 22.0}
    bg = f'rgb({int(p["bg_r"])},{int(p["bg_g"])},{int(p["bg_b"])})'
    cc = f'rgb({int(p["c_r"])},{int(p["c_g"])},{int(p["c_b"])})'
    cx, cy, r = p["cx"], p["cy"], p["radius"]
    return (f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
            f'<rect width="{w}" height="{h}" fill="{bg}"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{cc}" opacity="0.7"/>'
            f'</svg>')
"""

SEED_GRADIENT = """\
def generate_svg(w, h, p=None):
    # Radial gradient glow from bright center to dark edges
    if p is None:
        p = {"center_r": 220.0, "center_g": 200.0, "center_b": 190.0,
             "edge_r": 60.0, "edge_g": 30.0, "edge_b": 130.0,
             "cx": 0.5, "cy": 0.5, "fr": 0.0, "outer_r": 0.75}
    cr = int(p["center_r"])
    cg = int(p["center_g"])
    cb = int(p["center_b"])
    er = int(p["edge_r"])
    eg = int(p["edge_g"])
    eb = int(p["edge_b"])
    cx_pct = str(round(p["cx"] * 100)) + "%"
    cy_pct = str(round(p["cy"] * 100)) + "%"
    fr_pct = str(round(p["fr"] * 100)) + "%"
    or_pct = str(round(p["outer_r"] * 100)) + "%"
    return (f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
            f'<defs><radialGradient id="g" cx="{cx_pct}" cy="{cy_pct}" '
            f'r="{or_pct}" fx="{cx_pct}" fy="{cy_pct}">'
            f'<stop offset="0%" stop-color="rgb({cr},{cg},{cb})"/>'
            f'<stop offset="100%" stop-color="rgb({er},{eg},{eb})"/>'
            f'</radialGradient></defs>'
            f'<rect width="{w}" height="{h}" fill="url(#g)"/>'
            f'</svg>')
"""

SEEDS = [SEED_FLAT, SEED_RADIAL, SEED_GRADIENT]


def save_svg(fn, path, p=None):
    """Save the SVG output of a generator function to a file."""
    try:
        if p is not None:
            svg = fn(W, H, p)
        else:
            svg = fn(W, H)
        with open(path, "w") as f:
            f.write(svg)
    except Exception:
        pass


def save_png(fn, path, p=None):
    """Render SVG from a generator function and save as PNG."""
    try:
        if p is not None:
            svg = fn(W, H, p)
        else:
            svg = fn(W, H)
        png_data = cairosvg.svg2png(
            bytestring=svg.encode(), output_width=W, output_height=H,
        )
        with open(path, "wb") as f:
            f.write(png_data)
    except Exception:
        pass


async def main() -> None:
    from evolution_agent.core.config import load_config
    from evolution_agent.core.engine import EvolutionEngine
    from evolution_agent.core.types import OptimizationDirection
    from evolution_agent.evaluation.function_eval import FunctionEvaluator
    from evolution_agent.evaluation.hybrid_eval import HybridEvaluator

    config = load_config(
        str(Path(__file__).parent.parent / "config" / "default.yaml"),
        overrides={
            "population_size": int(os.environ.get("EVOL_POPULATION_SIZE", "12")),
            "max_generations": int(os.environ.get("EVOL_MAX_GENERATIONS", "30")),
            "elite_count": 2,
            "analyzer_every_n_gens": int(os.environ.get("EVOL_ANALYZER_EVERY", "5")),
            "analyzer_model": os.environ.get("EVOL_ANALYZER_MODEL", "claude-code:sonnet"),
            "stagnation_limit": 15,
            "ollama_base_url": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            "meta_optimizer_type": os.environ.get("EVOL_META_OPTIMIZER_TYPE", "heuristic"),
            "max_concurrent_mutations": 1,
            "max_concurrent_evals": 1,
            "eval_timeout_s": 15.0,
        },
    )

    use_hybrid = os.environ.get("EVOL_HYBRID", "1") == "1"
    sampler = os.environ.get("EVOL_SAMPLER", "fast")
    tuning_trials = int(os.environ.get("EVOL_TUNING_TRIALS", "15"))
    curiosity_weight = float(os.environ.get("EVOL_CURIOSITY", "0.0"))

    if use_hybrid:
        evaluator = HybridEvaluator(
            fitness_fn=evaluate_svg,
            function_name="generate_svg",
            function_spec=FUNCTION_SPEC,
            direction=OptimizationDirection.MAXIMIZE,
            timeout_s=15.0,
            tuning_trials=tuning_trials,
            tuning_sampler=sampler,
            tuning_timeout_s=20.0,
            tune_threshold=0.02,
            curiosity_weight=curiosity_weight,
            embedding_key="region_errors",
        )
        mode = f"Hybrid mode: LLM structure + {sampler.upper()} parameter tuning ({tuning_trials} trials)"
        if curiosity_weight > 0:
            mode += f" + curiosity (lambda={curiosity_weight})"
        print(mode)
    else:
        evaluator = FunctionEvaluator(
            fitness_fn=evaluate_svg,
            function_name="generate_svg",
            function_spec=FUNCTION_SPEC,
            direction=OptimizationDirection.MAXIMIZE,
            timeout_s=15.0,
        )
        print("LLM-only mode (no parameter tuning)")

    engine = EvolutionEngine(
        config=config,
        evaluator=evaluator,
        seeds=SEEDS,
    )

    run_dir = str(engine._run_dir)

    summary = await engine.run()

    print("\n" + "=" * 60)
    print("SVG EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Generations: {summary['total_generations']}")
    print(f"Best fitness: {summary['best_fitness']:.6f}")
    print(f"  (higher = better match with fewer shapes)")
    print(f"Elapsed: {summary['elapsed_s']:.1f}s")

    # Render best result
    best_code = summary.get("best_code", "")
    if best_code:
        from evolution_agent.evaluation.sandbox import CodeSandbox
        sandbox = CodeSandbox()
        best_fn = sandbox.compile_function(best_code, "generate_svg")
        if best_fn:
            save_svg(best_fn, os.path.join(run_dir, "best.svg"))
            save_png(best_fn, os.path.join(run_dir, "best_render.png"))
            print(f"\nBest SVG saved to: {run_dir}/best.svg")
            print(f"Best render saved to: {run_dir}/best_render.png")

            # Save target for comparison
            target_path = os.path.join(run_dir, "target.png")
            pixels = TARGET_PIXELS
            img = Image.new("RGB", (W, H))
            for y in range(H):
                for x in range(W):
                    img.putpixel((x, y), pixels[y * W + x])
            img.save(target_path)
            print(f"Target saved to: {target_path}")

    print("\nBest SVG generator code:")
    print("-" * 40)
    print(best_code)


if __name__ == "__main__":
    asyncio.run(main())
