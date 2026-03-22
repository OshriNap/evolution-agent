# Image Evolution Experiment: Math Art Renderer

## Summary

A new experiment for the evolution agent that evolves a pure-math `render(x, y, w, h, p)` function to approximate a target image. The evolved function maps pixel coordinates to RGB values using only basic math — no image libraries, no file I/O. Evolution discovers creative mathematical expressions that produce recognizable images.

This replaces TSP as the flagship demo. The visual output makes evolution tangible: you can watch an image materialize from noise across generations.

## The Evolved Function

```python
def render(x, y, w, h, p=None):
    """
    Args:
        x, y: pixel coordinates (ints, 0-indexed)
        w, h: image dimensions
        p: optional parameter dict for tunable constants
    Returns:
        (r, g, b) tuple, each 0-255
    """
    if p is None:
        p = {"freq": 3.0, "phase": 0.5}

    nx, ny = x / w, y / h

    r = int(128 + 127 * math.sin(p["freq"] * nx + p["phase"]))
    g = int(128 + 127 * math.cos(p["freq"] * ny))
    b = 100
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
```

**Constraints:**
- Pure Python only. The `math` module is pre-injected into the sandbox namespace — code must NOT contain `import math`. The FUNCTION_SPEC must explicitly state available globals: `math, range, len, min, max, int, float, abs, sum, sorted`.
- Must return `(r, g, b)` with each channel clamped to 0-255
- Parameters in `p` dict for hybrid Optuna tuning
- No docstrings (comments only, per existing convention)

## Target Image

- Resolution: 64x64 pixels
- Hardcoded as a flat list of RGB tuples in the experiment file
- Image choice: a simple but recognizable subject — gradient circle on a colored background, or a basic face shape
- Chosen so partial matches are visually interpretable (you can see progress even at low fitness)

The target data is generated once from a source image and embedded as a Python constant. No file I/O at runtime.

## Fitness Function

```python
def evaluate_renderer(fn) -> tuple[float, dict[str, float]]:
    total_mse = 0.0
    r_err = g_err = b_err = 0.0
    pixels_ok = 0

    error_count = 0

    for y in range(H):
        for x in range(W):
            try:
                r, g, b = fn(x, y, W, H)
            except Exception:
                r, g, b = 0, 0, 0
                error_count += 1

            tr, tg, tb = TARGET[y * W + x]
            dr, dg, db = r - tr, g - tg, b - tb
            total_mse += dr*dr + dg*dg + db*db
            r_err += dr*dr
            g_err += dg*dg
            b_err += db*db

            if dr*dr + dg*dg + db*db < ERROR_THRESHOLD:
                pixels_ok += 1

    n = W * H

    # If more than half the pixels threw exceptions, reject entirely
    if error_count > n * 0.5:
        return 0.0, {"mse": 255.0**2, "r_error": 255.0**2,
                      "g_error": 255.0**2, "b_error": 255.0**2,
                      "pixel_coverage": 0.0, "region_errors": [1.0]*4}

    normalized_mse = total_mse / (n * 3 * 255 * 255)
    fitness = 1.0 / (1.0 + normalized_mse * 1000)

    # Per-quadrant MSE as behavioral embedding for curiosity system
    # (computed by splitting image into 2x2 grid regions)
    region_errors = _compute_region_errors(fn, W, H, TARGET)

    metrics = {
        "mse": total_mse / n,
        "r_error": r_err / n,
        "g_error": g_err / n,
        "b_error": b_err / n,
        "pixel_coverage": pixels_ok / n,
        "region_errors": region_errors,  # list[float] for curiosity embeddings
    }
    return fitness, metrics
```

**Direction:** maximize (1.0 = perfect, approaches 0.0 for bad matches)

**Evaluation speed:** 64x64 = 4096 calls to `render()`. Pure math, well under 1 second per candidate.

**Metrics feed into:**
- Dashboard display (per-channel error breakdown)
- Curiosity system (`embedding_key="region_errors"` — per-quadrant MSE as behavioral embedding vector)
- Analyzer guidance (knows which channels/regions are lagging)

## Seed Functions

Three seeds at different complexity levels for initial population diversity:

1. **Flat color** — returns the average color of the target image. Simplest baseline (~0.3-0.4 fitness). Gives evolution a correct color palette to start from.

2. **Gradient** — linear interpolation across x/y axes using target's edge colors. Captures broad spatial color distribution (~0.4-0.5 fitness).

3. **Sin/cos pattern** — basic trigonometric patterns with tunable frequency/phase. Gives the LLM structured math to riff on.

## Expected Evolution Trajectory

- **Early generations:** better base colors, axis-aligned gradients
- **Mid generations:** frequency combinations, radial patterns, conditional regions (`if nx > 0.5`), distance fields
- **Late generations:** layered compositions, pseudo-noise, edge approximations, complex conditionals

## Mutation Dynamics

Uses framework default mutation weights. The natural fit for this problem:
- **Point mutations**: tweak constants, frequencies, phase offsets — fine-tune existing patterns
- **Structural mutations**: add math layers, switch linear to radial, introduce conditionals — explore new strategies
- **Crossover**: combine color channel logic from one parent with spatial logic from another

## Visualization

**Render helper** (outside sandbox, in experiment file):
- Renders evolved function to PPM format (no dependencies needed)
- Optional PNG output via `zlib`/`struct` (stdlib only)

**Demo outputs:**
- Side-by-side: target vs. best evolved rendering
- Generation timelapse: best image per generation, showing the image materializing
- The evolved code itself: a compact math function that produces a recognizable image

## File Structure

Single file: `examples/optimize_image.py`

Contents:
- `TARGET`: hardcoded 64x64 RGB data
- `evaluate_renderer(fn)`: fitness function
- `FUNCTION_SPEC`: describes render signature for LLM
- `SEEDS`: list of 2-3 seed function strings
- `render_to_ppm(fn, path, p=None)`: visualization helper (accepts tuned params from hybrid mode)
- `main()`: configures and launches EvolutionEngine

No changes to the core engine or framework. This is purely a new experiment using existing infrastructure.

## Configuration

Uses `config/default.yaml` with these experiment-specific choices:
- `mutator_model`: claude-code:haiku (or ollama)
- `population_size`: 12-20
- `direction`: maximize
- `eval_timeout_s`: 10.0 (pure math should be fast)
- Hybrid mode recommended for auto-tuning frequency/phase constants
