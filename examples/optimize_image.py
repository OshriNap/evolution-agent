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
