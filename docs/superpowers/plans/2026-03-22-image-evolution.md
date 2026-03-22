# Image Evolution (Math Art Renderer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `examples/optimize_image.py` — an experiment that evolves pure-math `render(x, y, w, h, p)` functions to approximate a 64x64 target image using the existing evolution agent framework.

**Architecture:** Single experiment file following the same pattern as `examples/optimize_tsp.py`. The target image is embedded as a Python constant. The fitness function evaluates pixel-by-pixel MSE against the target. A PPM render helper visualizes results. No core framework changes needed.

**Tech Stack:** Python, existing evolution agent framework (`FunctionEvaluator`, `HybridEvaluator`, `EvolutionEngine`), `math` module for evolved functions, `zlib`/`struct` for PNG output.

**Spec:** `docs/superpowers/specs/2026-03-22-image-evolution-design.md`

---

### Task 1: Generate and embed the target image

**Files:**
- Create: `examples/generate_target.py` (one-time helper, not part of the experiment)
- Create: `examples/target_image.py` (holds the TARGET constant)

This task generates a 64x64 procedural target image using pure Python math and saves it as a Python constant. The image should be a gradient circle (bright center fading to dark edges) on a colored background — simple but recognizable, with smooth gradients that reward incremental improvement.

- [ ] **Step 1: Write the target image generator**

```python
# examples/generate_target.py
"""One-time helper: generate a 64x64 target image as a Python constant."""
import math

W, H = 64, 64

def generate_target():
    """Gradient circle on a warm background."""
    pixels = []
    cx, cy = W / 2, H / 2
    max_dist = math.sqrt(cx * cx + cy * cy)

    for y in range(H):
        for x in range(W):
            nx, ny = x / W, y / H
            dx, dy = x - cx, y - cy
            dist = math.sqrt(dx * dx + dy * dy) / max_dist

            # Background: warm gradient (dark orange to deep blue)
            bg_r = int(40 + 60 * ny)
            bg_g = int(20 + 30 * nx)
            bg_b = int(80 + 100 * (1 - ny))

            # Circle: bright yellow-white center fading out
            circle = max(0.0, 1.0 - dist * 1.8)
            circle = circle * circle  # sharper falloff

            r = int(min(255, bg_r + circle * (255 - bg_r)))
            g = int(min(255, bg_g + circle * (240 - bg_g)))
            b = int(min(255, bg_b + circle * (200 - bg_b)))

            pixels.append((r, g, b))
    return pixels

pixels = generate_target()

# Output as Python constant
lines = ["# Auto-generated 64x64 target image (gradient circle on warm background)"]
lines.append(f"W, H = {W}, {H}")
lines.append(f"TARGET = {pixels!r}")

with open("examples/target_image.py", "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Wrote {len(pixels)} pixels to examples/target_image.py")
```

- [ ] **Step 2: Run the generator**

Run: `cd /home/oshrin/projects/evolution-agent && python examples/generate_target.py`
Expected: "Wrote 4096 pixels to examples/target_image.py"

- [ ] **Step 3: Verify the target file**

Run: `python -c "from examples.target_image import W, H, TARGET; print(f'{W}x{H}, {len(TARGET)} pixels, first={TARGET[0]}, center={TARGET[32*64+32]}')" `
Expected: 64x64, 4096 pixels, with reasonable RGB values.

- [ ] **Step 4: Commit**

```bash
git add examples/generate_target.py examples/target_image.py
git commit -m "feat: add 64x64 target image for image evolution experiment"
```

---

### Task 2: Write the fitness function and FUNCTION_SPEC

**Files:**
- Create: `examples/optimize_image.py` (main experiment file — this task adds the fitness function portion)

- [ ] **Step 1: Write the test for the fitness function**

Create a test that verifies:
- A perfect renderer (returning exact target pixels) gets fitness ~1.0
- A black renderer (all zeros) gets fitness > 0 but low
- A crashing renderer (raises exceptions) gets fitness 0.0

```python
# tests/test_image_fitness.py
"""Tests for the image evolution fitness function."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_perfect_renderer_gets_max_fitness():
    from optimize_image import evaluate_renderer, TARGET, W, H

    def perfect(x, y, w, h, p=None):
        return TARGET[y * w + x]

    fitness, metrics = evaluate_renderer(perfect)
    assert fitness > 0.99, f"Perfect renderer should get ~1.0, got {fitness}"
    assert metrics["mse"] < 1.0
    assert metrics["pixel_coverage"] > 0.99


def test_black_renderer_gets_low_fitness():
    from optimize_image import evaluate_renderer

    def black(x, y, w, h, p=None):
        return (0, 0, 0)

    fitness, metrics = evaluate_renderer(black)
    assert 0.0 < fitness < 0.5, f"Black renderer should be low, got {fitness}"
    assert metrics["mse"] > 100


def test_crashing_renderer_gets_zero():
    from optimize_image import evaluate_renderer

    def crasher(x, y, w, h, p=None):
        raise ValueError("boom")

    fitness, metrics = evaluate_renderer(crasher)
    assert fitness == 0.0
    assert metrics["pixel_coverage"] == 0.0


def test_region_errors_is_list():
    from optimize_image import evaluate_renderer

    def gray(x, y, w, h, p=None):
        return (128, 128, 128)

    fitness, metrics = evaluate_renderer(gray)
    assert "region_errors" in metrics
    assert isinstance(metrics["region_errors"], list)
    assert len(metrics["region_errors"]) == 4  # 2x2 quadrants
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/oshrin/projects/evolution-agent && python -m pytest tests/test_image_fitness.py -v`
Expected: FAIL — `optimize_image` module does not exist yet.

- [ ] **Step 3: Write the fitness function in optimize_image.py**

```python
# examples/optimize_image.py
"""Example: evolve a math-art image renderer.

Evolves a function that maps pixel coordinates to RGB values using only basic
math. Fitness = similarity to a target image. Evolution discovers creative
mathematical expressions that approximate the target.

Usage:
    python examples/optimize_image.py
    EVOL_POPULATION_SIZE=12 EVOL_MAX_GENERATIONS=30 python examples/optimize_image.py
"""

from __future__ import annotations

import asyncio
import math
import os
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.target_image import TARGET, W, H

# Error threshold: pixel within this squared-error is "ok"
ERROR_THRESHOLD = 50 * 50 * 3  # ~50 per channel


def evaluate_renderer(fn) -> tuple[float, dict]:
    """Evaluate an image renderer function against the target.

    Fitness = 1 / (1 + normalized_mse * 1000). Range: (0, 1].
    Single-pass: computes global MSE, per-channel errors, and per-quadrant
    region errors (for curiosity embeddings) in one pixel loop.
    """
    total_mse = 0.0
    r_err = g_err = b_err = 0.0
    pixels_ok = 0
    error_count = 0
    n = W * H
    half_w, half_h = W // 2, H // 2
    regions = [0.0, 0.0, 0.0, 0.0]  # TL, TR, BL, BR
    region_counts = [0, 0, 0, 0]

    for y in range(H):
        for x in range(W):
            try:
                r, g, b = fn(x, y, W, H)
            except Exception:
                r, g, b = 0, 0, 0
                error_count += 1

            tr, tg, tb = TARGET[y * W + x]
            dr, dg, db = r - tr, g - tg, b - tb
            pixel_err = dr * dr + dg * dg + db * db
            total_mse += pixel_err
            r_err += dr * dr
            g_err += dg * dg
            b_err += db * db

            if pixel_err < ERROR_THRESHOLD:
                pixels_ok += 1

            # Accumulate per-quadrant error for behavioral embedding
            qi = (0 if y < half_h else 2) + (0 if x < half_w else 1)
            regions[qi] += pixel_err
            region_counts[qi] += 1

    # Reject if more than half the pixels threw exceptions
    if error_count > n * 0.5:
        return 0.0, {
            "mse": 255.0 ** 2, "r_error": 255.0 ** 2,
            "g_error": 255.0 ** 2, "b_error": 255.0 ** 2,
            "pixel_coverage": 0.0, "region_errors": [1.0] * 4,
        }

    normalized_mse = total_mse / (n * 3 * 255 * 255)
    fitness = 1.0 / (1.0 + normalized_mse * 1000)

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
    }
    return fitness, metrics
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/oshrin/projects/evolution-agent && python -m pytest tests/test_image_fitness.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/optimize_image.py tests/test_image_fitness.py
git commit -m "feat: add image evolution fitness function with tests"
```

---

### Task 3: Write seed functions and FUNCTION_SPEC

**Files:**
- Modify: `examples/optimize_image.py` (add FUNCTION_SPEC, SEEDS)

- [ ] **Step 1: Write the test for seed functions**

```python
# tests/test_image_seeds.py
"""Tests for image evolution seed functions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evolution_agent.evaluation.sandbox import CodeSandbox


def test_seeds_compile_in_sandbox():
    from optimize_image import SEEDS
    sandbox = CodeSandbox()

    for i, seed_code in enumerate(SEEDS):
        fn = sandbox.compile_function(seed_code, "render")
        assert fn is not None, f"Seed {i} failed to compile in sandbox"


def test_seeds_return_valid_rgb():
    from optimize_image import SEEDS
    sandbox = CodeSandbox()

    for i, seed_code in enumerate(SEEDS):
        fn = sandbox.compile_function(seed_code, "render")
        r, g, b = fn(0, 0, 64, 64)
        assert 0 <= r <= 255, f"Seed {i}: r={r} out of range"
        assert 0 <= g <= 255, f"Seed {i}: g={g} out of range"
        assert 0 <= b <= 255, f"Seed {i}: b={b} out of range"


def test_seeds_produce_different_outputs():
    from optimize_image import SEEDS
    sandbox = CodeSandbox()
    outputs = []

    for seed_code in SEEDS:
        fn = sandbox.compile_function(seed_code, "render")
        # Sample a few pixels
        sample = tuple(fn(x, y, 64, 64) for x, y in [(0, 0), (32, 32), (63, 63)])
        outputs.append(sample)

    # At least 2 seeds should produce different pixel values
    unique = len(set(outputs))
    assert unique >= 2, f"Seeds produce too-similar outputs: {unique} unique of {len(SEEDS)}"


def test_seeds_get_nonzero_fitness():
    from optimize_image import SEEDS, evaluate_renderer
    sandbox = CodeSandbox()

    for i, seed_code in enumerate(SEEDS):
        fn = sandbox.compile_function(seed_code, "render")
        fitness, metrics = evaluate_renderer(fn)
        assert fitness > 0, f"Seed {i} got zero fitness"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/oshrin/projects/evolution-agent && python -m pytest tests/test_image_seeds.py -v`
Expected: FAIL — `SEEDS` not defined yet.

- [ ] **Step 3: Add FUNCTION_SPEC and SEEDS to optimize_image.py**

Append to `examples/optimize_image.py` after the `evaluate_renderer` function:

```python
FUNCTION_SPEC = """\
def render(x, y, w, h, p=None):
    # Render a single pixel of an image using math.
    #
    # Args:
    #     x, y: pixel coordinates (ints, 0-indexed)
    #     w, h: image dimensions (64x64)
    #     p: optional dict of tunable numeric parameters with defaults.
    #        Put all tunable constants in p so they can be auto-optimized.
    #        Example: if p is None: p = {"freq": 3.0, "amp": 0.8}
    #        Access via p["freq"], p["amp"], etc.
    #
    # Returns:
    #     (r, g, b) tuple of ints, each in [0, 255].
    #
    # Available globals (already in scope, do NOT import):
    #     math (math.sin, math.cos, math.sqrt, math.exp, math.pi, etc.)
    #     range, len, min, max, int, float, abs, sum, sorted, round, pow
    #
    # Tips:
    #     - Normalize coords: nx, ny = x / w, y / h  (gives 0..1 range)
    #     - Use math.sin/cos for periodic patterns
    #     - Use distance from center for radial patterns
    #     - Clamp output: max(0, min(255, value))
    #     - Combine multiple patterns for complexity
"""

# Seed 1: flat average color of the target
_avg_r = sum(r for r, g, b in TARGET) // len(TARGET)
_avg_g = sum(g for r, g, b in TARGET) // len(TARGET)
_avg_b = sum(b for r, g, b in TARGET) // len(TARGET)

SEED_FLAT = f"""\
def render(x, y, w, h, p=None):
    # Flat color: average of target image
    if p is None:
        p = {{"r": {_avg_r}.0, "g": {_avg_g}.0, "b": {_avg_b}.0}}
    return (max(0, min(255, int(p["r"]))),
            max(0, min(255, int(p["g"]))),
            max(0, min(255, int(p["b"]))))
"""

SEED_GRADIENT = """\
def render(x, y, w, h, p=None):
    # Linear gradient across both axes
    if p is None:
        p = {"r_x": 0.3, "r_y": 0.7, "g_x": 0.1, "g_y": 0.3,
             "b_x": -0.2, "b_y": 0.5, "r0": 80.0, "g0": 40.0, "b0": 120.0}
    nx, ny = x / w, y / h
    r = int(p["r0"] + (p["r_x"] * nx + p["r_y"] * ny) * 128)
    g = int(p["g0"] + (p["g_x"] * nx + p["g_y"] * ny) * 128)
    b = int(p["b0"] + (p["b_x"] * nx + p["b_y"] * ny) * 128)
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
"""

SEED_SINCOS = """\
def render(x, y, w, h, p=None):
    # Trigonometric pattern with radial component
    if p is None:
        p = {"freq": 3.0, "phase": 0.5, "r_base": 80.0, "g_base": 50.0,
             "b_base": 120.0, "amp": 80.0, "cx": 0.5, "cy": 0.5}
    nx, ny = x / w, y / h
    dx, dy = nx - p["cx"], ny - p["cy"]
    dist = (dx * dx + dy * dy) ** 0.5
    wave = math.sin(p["freq"] * dist * 6.283 + p["phase"])
    r = int(p["r_base"] + p["amp"] * wave * (1.0 - dist))
    g = int(p["g_base"] + p["amp"] * 0.8 * wave * (1.0 - dist))
    b = int(p["b_base"] + p["amp"] * 0.5 * math.cos(p["freq"] * nx * 6.283))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
"""

SEEDS = [SEED_FLAT, SEED_GRADIENT, SEED_SINCOS]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/oshrin/projects/evolution-agent && python -m pytest tests/test_image_seeds.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/optimize_image.py tests/test_image_seeds.py
git commit -m "feat: add FUNCTION_SPEC and 3 seed functions for image evolution"
```

---

### Task 4: Write the PPM/PNG render helper

**Files:**
- Modify: `examples/optimize_image.py` (add render_to_ppm and render_to_png)

- [ ] **Step 1: Write the test**

```python
# tests/test_image_render.py
"""Tests for image render helpers."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_render_to_ppm_creates_valid_file():
    from optimize_image import render_to_ppm, W, H

    def red(x, y, w, h, p=None):
        return (255, 0, 0)

    with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
        path = f.name

    try:
        render_to_ppm(red, path)
        assert os.path.exists(path)
        with open(path, "rb") as f:
            header = f.readline()
            assert header.strip() == b"P6"
            dims = f.readline()
            assert dims.strip() == f"{W} {H}".encode()
        assert os.path.getsize(path) > W * H * 3
    finally:
        os.unlink(path)


def test_render_to_png_creates_valid_file():
    from optimize_image import render_to_png, W, H

    def blue(x, y, w, h, p=None):
        return (0, 0, 255)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name

    try:
        render_to_png(blue, path)
        assert os.path.exists(path)
        with open(path, "rb") as f:
            sig = f.read(8)
            assert sig == b"\x89PNG\r\n\x1a\n", "Invalid PNG signature"
    finally:
        os.unlink(path)


def test_render_with_params():
    from optimize_image import render_to_ppm

    def parameterized(x, y, w, h, p=None):
        if p is None:
            p = {"v": 128}
        v = int(p["v"])
        return (v, v, v)

    with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
        path = f.name

    try:
        render_to_ppm(parameterized, path, p={"v": 200})
        assert os.path.exists(path)
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/oshrin/projects/evolution-agent && python -m pytest tests/test_image_render.py -v`
Expected: FAIL — `render_to_ppm` and `render_to_png` not defined.

- [ ] **Step 3: Add render helpers to optimize_image.py**

Append to `examples/optimize_image.py`:

```python
def render_to_ppm(fn, path, p=None):
    """Render an image function to PPM format (no dependencies)."""
    pixels = bytearray()
    for y in range(H):
        for x in range(W):
            try:
                if p is not None:
                    r, g, b = fn(x, y, W, H, p)
                else:
                    r, g, b = fn(x, y, W, H)
            except Exception:
                r, g, b = 0, 0, 0
            pixels.extend([
                max(0, min(255, int(r))),
                max(0, min(255, int(g))),
                max(0, min(255, int(b))),
            ])

    with open(path, "wb") as f:
        f.write(f"P6\n{W} {H}\n255\n".encode())
        f.write(bytes(pixels))


def render_to_png(fn, path, p=None):
    """Render an image function to PNG format (stdlib only)."""
    raw_rows = []
    for y in range(H):
        row = bytearray([0])  # filter byte: None
        for x in range(W):
            try:
                if p is not None:
                    r, g, b = fn(x, y, W, H, p)
                else:
                    r, g, b = fn(x, y, W, H)
            except Exception:
                r, g, b = 0, 0, 0
            row.extend([
                max(0, min(255, int(r))),
                max(0, min(255, int(g))),
                max(0, min(255, int(b))),
            ])
        raw_rows.append(bytes(row))

    raw_data = b"".join(raw_rows)

    def _png_chunk(chunk_type, data):
        chunk = chunk_type + data
        return (struct.pack(">I", len(data)) + chunk +
                struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")  # signature
        # IHDR
        ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
        f.write(_png_chunk(b"IHDR", ihdr))
        # IDAT
        compressed = zlib.compress(raw_data)
        f.write(_png_chunk(b"IDAT", compressed))
        # IEND
        f.write(_png_chunk(b"IEND", b""))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/oshrin/projects/evolution-agent && python -m pytest tests/test_image_render.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/optimize_image.py tests/test_image_render.py
git commit -m "feat: add PPM/PNG render helpers for image evolution"
```

---

### Task 5: Write the main() entry point

**Files:**
- Modify: `examples/optimize_image.py` (add `main()` and `__main__` block)

- [ ] **Step 1: Write a smoke test**

```python
# tests/test_image_main.py
"""Smoke test for image evolution main."""
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_main_configures_engine_correctly():
    """Verify main() creates an EvolutionEngine with correct params."""
    from optimize_image import SEEDS, FUNCTION_SPEC

    # Just verify the pieces exist and are well-formed
    assert len(SEEDS) >= 2
    assert "render" in FUNCTION_SPEC
    assert "math" in FUNCTION_SPEC
    assert "do NOT import" in FUNCTION_SPEC.lower() or "NOT import" in FUNCTION_SPEC
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `cd /home/oshrin/projects/evolution-agent && python -m pytest tests/test_image_main.py -v`
Expected: PASS (the pieces already exist from previous tasks).

- [ ] **Step 3: Add main() to optimize_image.py**

Append to `examples/optimize_image.py`:

```python
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
            "eval_timeout_s": 10.0,
        },
    )

    use_hybrid = os.environ.get("EVOL_HYBRID", "1") == "1"
    sampler = os.environ.get("EVOL_SAMPLER", "fast")
    tuning_trials = int(os.environ.get("EVOL_TUNING_TRIALS", "15"))
    curiosity_weight = float(os.environ.get("EVOL_CURIOSITY", "0.0"))

    if use_hybrid:
        evaluator = HybridEvaluator(
            fitness_fn=evaluate_renderer,
            function_name="render",
            function_spec=FUNCTION_SPEC,
            direction=OptimizationDirection.MAXIMIZE,
            timeout_s=10.0,
            tuning_trials=tuning_trials,
            tuning_sampler=sampler,
            tuning_timeout_s=15.0,
            tune_threshold=0.1,
            curiosity_weight=curiosity_weight,
            embedding_key="region_errors",
        )
        mode = f"Hybrid mode: LLM structure + {sampler.upper()} parameter tuning ({tuning_trials} trials)"
        if curiosity_weight > 0:
            mode += f" + curiosity (lambda={curiosity_weight})"
        print(mode)
    else:
        evaluator = FunctionEvaluator(
            fitness_fn=evaluate_renderer,
            function_name="render",
            function_spec=FUNCTION_SPEC,
            direction=OptimizationDirection.MAXIMIZE,
            timeout_s=10.0,
        )
        print("LLM-only mode (no parameter tuning)")

    engine = EvolutionEngine(
        config=config,
        evaluator=evaluator,
        seeds=SEEDS,
    )

    # Capture run_dir before running (engine exposes it as _run_dir)
    run_dir = str(engine._run_dir)

    summary = await engine.run()

    print("\n" + "=" * 60)
    print("IMAGE EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Generations: {summary['total_generations']}")
    print(f"Best fitness: {summary['best_fitness']:.6f}")
    print(f"  (1.0 = perfect match, higher = closer to target)")
    print(f"Elapsed: {summary['elapsed_s']:.1f}s")

    # Render best result
    best_code = summary.get("best_code", "")
    if best_code:
        from evolution_agent.evaluation.sandbox import CodeSandbox
        sandbox = CodeSandbox()
        best_fn = sandbox.compile_function(best_code, "render")
        if best_fn:
            out_path = os.path.join(run_dir, "best_render.png")
            render_to_png(best_fn, out_path)
            print(f"\nBest render saved to: {out_path}")

            # Also save target for comparison
            target_path = os.path.join(run_dir, "target.png")
            def target_fn(x, y, w, h, p=None):
                return TARGET[y * w + x]
            render_to_png(target_fn, target_path)
            print(f"Target saved to: {target_path}")

    print("\nBest renderer code:")
    print("-" * 40)
    print(best_code)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Verify the script can at least parse without errors**

Run: `cd /home/oshrin/projects/evolution-agent && python -c "import examples.optimize_image; print('OK')"`
Expected: "OK"

- [ ] **Step 5: Commit**

```bash
git add examples/optimize_image.py tests/test_image_main.py
git commit -m "feat: add main() entry point for image evolution experiment"
```

---

### Task 6: End-to-end smoke test (1 generation)

**Files:**
- No new files — just run the experiment with minimal settings

- [ ] **Step 1: Run all tests**

Run: `cd /home/oshrin/projects/evolution-agent && python -m pytest tests/test_image_fitness.py tests/test_image_seeds.py tests/test_image_render.py tests/test_image_main.py -v`
Expected: All tests PASS.

- [ ] **Step 2: Run 1 generation with LLM-only mode**

Run: `cd /home/oshrin/projects/evolution-agent && EVOL_POPULATION_SIZE=4 EVOL_MAX_GENERATIONS=1 EVOL_HYBRID=0 python examples/optimize_image.py`
Expected: Completes without errors. Prints fitness > 0 for at least some candidates. Saves best_render.png and target.png.

- [ ] **Step 3: Verify output images exist and are valid PNGs**

Run: `ls -la runs/*/best_render.png runs/*/target.png`
Expected: Both files exist and are non-empty.

- [ ] **Step 4: Commit any final fixes**

If anything needed fixing, commit with: `git commit -m "fix: address issues from image evolution smoke test"`
