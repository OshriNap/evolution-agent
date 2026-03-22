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
