"""BO-only SVG sweep: generate experience data without LLM.

Explores SVG parameter space using Optuna + structural mutations (add/remove
shapes, fix/release parameters, reorder, duplicate). Supports rich SVG
primitives including gradients, filters, and composite shapes.

Actions:
  - add_shape: add a new SVG element (with gradients, filters, etc.)
  - remove_shape: remove an existing element
  - duplicate_shape: copy a shape with slight perturbation
  - reorder_shape: move a shape forward/backward in z-order
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

SHAPE_TYPES = [
    "circle", "rect", "ellipse", "line", "polygon",
    "radial_gradient_circle", "radial_gradient_rect",
    "linear_gradient_rect",
    "blurred_circle", "blurred_ellipse",
    "ring",
]

# Weights for random shape selection (favor gradient shapes)
SHAPE_WEIGHTS = [
    8, 10, 6, 3, 4,    # basic shapes
    15, 12, 12,         # gradient shapes (heavily favored)
    8, 6,               # blurred shapes
    5,                  # ring
]

SHAPE_PARAM_DEFS = {
    "circle": {
        "cx": (0.0, 64.0), "cy": (0.0, 64.0), "r": (1.0, 40.0),
        "fill_r": (0, 255), "fill_g": (0, 255), "fill_b": (0, 255),
        "opacity": (0.1, 1.0),
    },
    "rect": {
        "x": (-10.0, 64.0), "y": (-10.0, 64.0),
        "w": (1.0, 80.0), "h": (1.0, 80.0),
        "fill_r": (0, 255), "fill_g": (0, 255), "fill_b": (0, 255),
        "opacity": (0.1, 1.0),
        "rx": (0.0, 20.0),  # rounded corners
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
        "stroke_width": (1.0, 15.0), "opacity": (0.1, 1.0),
    },
    "polygon": {
        "cx": (0.0, 64.0), "cy": (0.0, 64.0),
        "r": (5.0, 35.0), "sides": (3, 8), "rotation": (0.0, 360.0),
        "fill_r": (0, 255), "fill_g": (0, 255), "fill_b": (0, 255),
        "opacity": (0.1, 1.0),
    },
    "radial_gradient_circle": {
        "cx": (0.0, 64.0), "cy": (0.0, 64.0), "r": (5.0, 50.0),
        "inner_r": (0, 255), "inner_g": (0, 255), "inner_b": (0, 255),
        "outer_r": (0, 255), "outer_g": (0, 255), "outer_b": (0, 255),
        "inner_opacity": (0.0, 1.0), "outer_opacity": (0.0, 1.0),
        "focal_x": (0.2, 0.8), "focal_y": (0.2, 0.8),
    },
    "radial_gradient_rect": {
        "gcx": (0.1, 0.9), "gcy": (0.1, 0.9), "gr": (0.2, 1.0),
        "inner_r": (0, 255), "inner_g": (0, 255), "inner_b": (0, 255),
        "outer_r": (0, 255), "outer_g": (0, 255), "outer_b": (0, 255),
        "inner_opacity": (0.0, 1.0), "outer_opacity": (0.0, 1.0),
        "mid_r": (0, 255), "mid_g": (0, 255), "mid_b": (0, 255),
        "mid_offset": (0.2, 0.8),
    },
    "linear_gradient_rect": {
        "x1_pct": (0.0, 1.0), "y1_pct": (0.0, 1.0),
        "x2_pct": (0.0, 1.0), "y2_pct": (0.0, 1.0),
        "start_r": (0, 255), "start_g": (0, 255), "start_b": (0, 255),
        "end_r": (0, 255), "end_g": (0, 255), "end_b": (0, 255),
        "start_opacity": (0.0, 1.0), "end_opacity": (0.0, 1.0),
        "rect_x": (-10.0, 64.0), "rect_y": (-10.0, 64.0),
        "rect_w": (10.0, 80.0), "rect_h": (10.0, 80.0),
    },
    "blurred_circle": {
        "cx": (0.0, 64.0), "cy": (0.0, 64.0), "r": (3.0, 40.0),
        "fill_r": (0, 255), "fill_g": (0, 255), "fill_b": (0, 255),
        "opacity": (0.1, 1.0), "blur": (1.0, 15.0),
    },
    "blurred_ellipse": {
        "cx": (0.0, 64.0), "cy": (0.0, 64.0),
        "rx": (3.0, 40.0), "ry": (3.0, 40.0),
        "fill_r": (0, 255), "fill_g": (0, 255), "fill_b": (0, 255),
        "opacity": (0.1, 1.0), "blur": (1.0, 15.0),
    },
    "ring": {
        "cx": (0.0, 64.0), "cy": (0.0, 64.0),
        "r": (5.0, 35.0), "stroke_width": (1.0, 15.0),
        "stroke_r": (0, 255), "stroke_g": (0, 255), "stroke_b": (0, 255),
        "opacity": (0.1, 1.0),
    },
}

_def_counter = 0


def _next_id() -> str:
    global _def_counter
    _def_counter += 1
    return f"d{_def_counter}"


@dataclass
class Shape:
    type: str
    params: dict[str, float]
    fixed: dict[str, bool] = field(default_factory=dict)

    def to_svg_defs_and_element(self) -> tuple[str, str]:
        """Return (defs_xml, element_xml) for this shape."""
        p = self.params
        op = p.get("opacity", 1.0)
        defs = ""
        elem = ""

        if self.type == "circle":
            elem = (f'<circle cx="{p["cx"]:.1f}" cy="{p["cy"]:.1f}" '
                    f'r="{p["r"]:.1f}" '
                    f'fill="rgb({int(p["fill_r"])},{int(p["fill_g"])},{int(p["fill_b"])})" '
                    f'opacity="{op:.2f}"/>')

        elif self.type == "rect":
            rx = p.get("rx", 0)
            elem = (f'<rect x="{p["x"]:.1f}" y="{p["y"]:.1f}" '
                    f'width="{p["w"]:.1f}" height="{p["h"]:.1f}" '
                    f'rx="{rx:.1f}" '
                    f'fill="rgb({int(p["fill_r"])},{int(p["fill_g"])},{int(p["fill_b"])})" '
                    f'opacity="{op:.2f}"/>')

        elif self.type == "ellipse":
            elem = (f'<ellipse cx="{p["cx"]:.1f}" cy="{p["cy"]:.1f}" '
                    f'rx="{p["rx"]:.1f}" ry="{p["ry"]:.1f}" '
                    f'fill="rgb({int(p["fill_r"])},{int(p["fill_g"])},{int(p["fill_b"])})" '
                    f'opacity="{op:.2f}"/>')

        elif self.type == "line":
            elem = (f'<line x1="{p["x1"]:.1f}" y1="{p["y1"]:.1f}" '
                    f'x2="{p["x2"]:.1f}" y2="{p["y2"]:.1f}" '
                    f'stroke="rgb({int(p["stroke_r"])},{int(p["stroke_g"])},{int(p["stroke_b"])})" '
                    f'stroke-width="{p["stroke_width"]:.1f}" opacity="{op:.2f}" '
                    f'stroke-linecap="round"/>')

        elif self.type == "polygon":
            cx, cy, r = p["cx"], p["cy"], p["r"]
            sides = max(3, int(p["sides"]))
            rot = p.get("rotation", 0)
            points = []
            for i in range(sides):
                angle = math.radians(rot + 360 * i / sides)
                px = cx + r * math.cos(angle)
                py = cy + r * math.sin(angle)
                points.append(f"{px:.1f},{py:.1f}")
            pts = " ".join(points)
            elem = (f'<polygon points="{pts}" '
                    f'fill="rgb({int(p["fill_r"])},{int(p["fill_g"])},{int(p["fill_b"])})" '
                    f'opacity="{op:.2f}"/>')

        elif self.type == "radial_gradient_circle":
            gid = _next_id()
            fx = p.get("focal_x", 0.5)
            fy = p.get("focal_y", 0.5)
            io_ = p.get("inner_opacity", 1.0)
            oo = p.get("outer_opacity", 1.0)
            defs = (f'<radialGradient id="{gid}" cx="50%" cy="50%" r="50%" '
                    f'fx="{fx*100:.0f}%" fy="{fy*100:.0f}%">'
                    f'<stop offset="0%" stop-color="rgb({int(p["inner_r"])},{int(p["inner_g"])},{int(p["inner_b"])})" '
                    f'stop-opacity="{io_:.2f}"/>'
                    f'<stop offset="100%" stop-color="rgb({int(p["outer_r"])},{int(p["outer_g"])},{int(p["outer_b"])})" '
                    f'stop-opacity="{oo:.2f}"/>'
                    f'</radialGradient>')
            elem = (f'<circle cx="{p["cx"]:.1f}" cy="{p["cy"]:.1f}" '
                    f'r="{p["r"]:.1f}" fill="url(#{gid})"/>')

        elif self.type == "radial_gradient_rect":
            gid = _next_id()
            gcx = p.get("gcx", 0.5)
            gcy = p.get("gcy", 0.5)
            gr = p.get("gr", 0.5)
            io_ = p.get("inner_opacity", 1.0)
            oo = p.get("outer_opacity", 1.0)
            mo = p.get("mid_offset", 0.5)
            defs = (f'<radialGradient id="{gid}" cx="{gcx*100:.0f}%" cy="{gcy*100:.0f}%" '
                    f'r="{gr*100:.0f}%">'
                    f'<stop offset="0%" stop-color="rgb({int(p["inner_r"])},{int(p["inner_g"])},{int(p["inner_b"])})" '
                    f'stop-opacity="{io_:.2f}"/>'
                    f'<stop offset="{mo*100:.0f}%" stop-color="rgb({int(p["mid_r"])},{int(p["mid_g"])},{int(p["mid_b"])})" />'
                    f'<stop offset="100%" stop-color="rgb({int(p["outer_r"])},{int(p["outer_g"])},{int(p["outer_b"])})" '
                    f'stop-opacity="{oo:.2f}"/>'
                    f'</radialGradient>')
            elem = f'<rect width="64" height="64" fill="url(#{gid})"/>'

        elif self.type == "linear_gradient_rect":
            gid = _next_id()
            so = p.get("start_opacity", 1.0)
            eo = p.get("end_opacity", 1.0)
            defs = (f'<linearGradient id="{gid}" '
                    f'x1="{p["x1_pct"]*100:.0f}%" y1="{p["y1_pct"]*100:.0f}%" '
                    f'x2="{p["x2_pct"]*100:.0f}%" y2="{p["y2_pct"]*100:.0f}%">'
                    f'<stop offset="0%" stop-color="rgb({int(p["start_r"])},{int(p["start_g"])},{int(p["start_b"])})" '
                    f'stop-opacity="{so:.2f}"/>'
                    f'<stop offset="100%" stop-color="rgb({int(p["end_r"])},{int(p["end_g"])},{int(p["end_b"])})" '
                    f'stop-opacity="{eo:.2f}"/>'
                    f'</linearGradient>')
            elem = (f'<rect x="{p["rect_x"]:.1f}" y="{p["rect_y"]:.1f}" '
                    f'width="{p["rect_w"]:.1f}" height="{p["rect_h"]:.1f}" '
                    f'fill="url(#{gid})"/>')

        elif self.type == "blurred_circle":
            fid = _next_id()
            blur = p.get("blur", 5.0)
            defs = (f'<filter id="{fid}"><feGaussianBlur stdDeviation="{blur:.1f}"/></filter>')
            elem = (f'<circle cx="{p["cx"]:.1f}" cy="{p["cy"]:.1f}" '
                    f'r="{p["r"]:.1f}" '
                    f'fill="rgb({int(p["fill_r"])},{int(p["fill_g"])},{int(p["fill_b"])})" '
                    f'opacity="{op:.2f}" filter="url(#{fid})"/>')

        elif self.type == "blurred_ellipse":
            fid = _next_id()
            blur = p.get("blur", 5.0)
            defs = (f'<filter id="{fid}"><feGaussianBlur stdDeviation="{blur:.1f}"/></filter>')
            elem = (f'<ellipse cx="{p["cx"]:.1f}" cy="{p["cy"]:.1f}" '
                    f'rx="{p["rx"]:.1f}" ry="{p["ry"]:.1f}" '
                    f'fill="rgb({int(p["fill_r"])},{int(p["fill_g"])},{int(p["fill_b"])})" '
                    f'opacity="{op:.2f}" filter="url(#{fid})"/>')

        elif self.type == "ring":
            elem = (f'<circle cx="{p["cx"]:.1f}" cy="{p["cy"]:.1f}" '
                    f'r="{p["r"]:.1f}" fill="none" '
                    f'stroke="rgb({int(p["stroke_r"])},{int(p["stroke_g"])},{int(p["stroke_b"])})" '
                    f'stroke-width="{p["stroke_width"]:.1f}" opacity="{op:.2f}"/>')

        return defs, elem


@dataclass
class Genome:
    shapes: list[Shape] = field(default_factory=list)
    bg_r: float = 128.0
    bg_g: float = 128.0
    bg_b: float = 128.0
    bg_fixed: bool = False

    def to_svg(self, w: int = 64, h: int = 64) -> str:
        global _def_counter
        _def_counter = 0  # reset for reproducible IDs
        bg = f'rgb({int(self.bg_r)},{int(self.bg_g)},{int(self.bg_b)})'
        all_defs = []
        all_elems = [f'<rect width="{w}" height="{h}" fill="{bg}"/>']
        for s in self.shapes:
            d, e = s.to_svg_defs_and_element()
            if d:
                all_defs.append(d)
            if e:
                all_elems.append(e)
        parts = [f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">']
        if all_defs:
            parts.append("<defs>" + "".join(all_defs) + "</defs>")
        parts.extend(all_elems)
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
        shape_type = random.choices(SHAPE_TYPES, weights=SHAPE_WEIGHTS, k=1)[0]
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
    if len(g.shapes) >= 20:
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


def action_duplicate_shape(genome: Genome) -> tuple[Genome, dict]:
    g = genome.clone()
    if len(g.shapes) == 0 or len(g.shapes) >= 20:
        return g, {"action": "duplicate_shape", "success": False,
                   "reason": "no_shapes" if not g.shapes else "max_shapes"}
    idx = random.randint(0, len(g.shapes) - 1)
    new_shape = copy.deepcopy(g.shapes[idx])
    # Perturb position slightly
    for pname in ["cx", "cy", "x", "y", "x1", "y1"]:
        if pname in new_shape.params:
            new_shape.params[pname] += random.uniform(-10, 10)
    new_shape.fixed = {k: False for k in new_shape.fixed}
    g.shapes.append(new_shape)
    return g, {"action": "duplicate_shape", "success": True,
               "shape_type": new_shape.type, "source_idx": idx,
               "shape_idx": len(g.shapes) - 1}


def action_reorder_shape(genome: Genome) -> tuple[Genome, dict]:
    g = genome.clone()
    if len(g.shapes) < 2:
        return g, {"action": "reorder_shape", "success": False, "reason": "too_few"}
    idx = random.randint(0, len(g.shapes) - 1)
    direction = random.choice(["forward", "backward", "to_front", "to_back"])
    shape = g.shapes.pop(idx)
    if direction == "forward" and idx < len(g.shapes):
        g.shapes.insert(idx + 1, shape)
    elif direction == "backward" and idx > 0:
        g.shapes.insert(idx - 1, shape)
    elif direction == "to_front":
        g.shapes.append(shape)
    elif direction == "to_back":
        g.shapes.insert(0, shape)
    else:
        g.shapes.insert(idx, shape)
    return g, {"action": "reorder_shape", "success": True,
               "shape_idx": idx, "direction": direction}


def action_replace_shape(genome: Genome) -> tuple[Genome, dict]:
    """Replace a shape with a random new one (keeps slot, changes type)."""
    g = genome.clone()
    if len(g.shapes) == 0:
        return g, {"action": "replace_shape", "success": False, "reason": "no_shapes"}
    idx = random.randint(0, len(g.shapes) - 1)
    old_type = g.shapes[idx].type
    new_shape = random_shape()
    g.shapes[idx] = new_shape
    return g, {"action": "replace_shape", "success": True,
               "old_type": old_type, "new_type": new_shape.type, "shape_idx": idx}


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


ACTIONS = [
    action_add_shape, action_remove_shape, action_duplicate_shape,
    action_reorder_shape, action_replace_shape,
    action_fix_param, action_release_param,
]
ACTION_WEIGHTS = [0.25, 0.10, 0.10, 0.08, 0.12, 0.17, 0.18]


# ── BO optimization ───────────────────────────────────────────────

def _apply_trial_params(g: Genome, tunable: list, trial_or_params: dict, source: str = "trial"):
    """Apply parameter values to genome. Works with both Optuna trials and dicts."""
    for key, pname, lo, hi in tunable:
        param_key = f"{key}_{pname}"
        if source == "trial":
            if isinstance(lo, int):
                val = float(trial_or_params.suggest_int(param_key, lo, hi))
            else:
                val = trial_or_params.suggest_float(param_key, lo, hi)
        else:
            val = trial_or_params.get(param_key, None)
            if val is None:
                continue
        if key == "bg":
            if pname == "r": g.bg_r = val
            elif pname == "g": g.bg_g = val
            elif pname == "b": g.bg_b = val
        else:
            si = int(key[1:])
            g.shapes[si].params[pname] = float(val)


def _collect_tunable(genome: Genome) -> list:
    """Collect all tunable (non-fixed) parameters from a genome."""
    tunable = []
    if not genome.bg_fixed:
        tunable.append(("bg", "r", 0, 255))
        tunable.append(("bg", "g", 0, 255))
        tunable.append(("bg", "b", 0, 255))
    for si, s in enumerate(genome.shapes):
        param_defs = SHAPE_PARAM_DEFS[s.type]
        for pname, is_fixed in s.fixed.items():
            if not is_fixed and pname in param_defs:
                lo, hi = param_defs[pname]
                tunable.append((f"s{si}", pname, lo, hi))
    return tunable


def _eval_genome(g: Genome, target, w, h, gpu_rast=None, target_gpu=None):
    """Evaluate a genome using GPU or CPU."""
    if gpu_rast is not None and target_gpu is not None:
        canvas = gpu_rast.render(g)
        fit, sim, regions = gpu_rast.fitness(canvas, target_gpu, g.shape_count())
        # Approximate MSE from similarity for logging
        mse = (1.0 / sim - 1.0) / 1000.0 * 3 * 255 * 255 if sim > 0 else 255**2
        return fit, sim, mse, regions
    else:
        svg = g.to_svg(w, h)
        pixels = svg_to_pixels(svg, w, h)
        if pixels is None:
            return 0.0, 0.0, 255**2, [1.0]*4
        return compute_fitness(pixels, target, w, h, g.shape_count())


def optimize_params(genome: Genome, target, w: int, h: int,
                    n_trials: int = 30, gpu_rast=None,
                    target_gpu=None) -> tuple[Genome, float, list[dict]]:
    """Run Optuna on released parameters. Uses GPU if available."""
    g = genome.clone()
    tunable = _collect_tunable(g)

    if not tunable:
        fit, _, _, _ = _eval_genome(g, target, w, h, gpu_rast, target_gpu)
        return g, fit, []

    trial_log = []

    def objective(trial):
        _apply_trial_params(g, tunable, trial, source="trial")
        fitness, sim, mse, _ = _eval_genome(g, target, w, h, gpu_rast, target_gpu)
        trial_log.append({"trial": trial.number, "fitness": fitness,
                         "similarity": sim, "mse": mse})
        return fitness

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random.randint(0, 2**31)))
    study.optimize(objective, n_trials=n_trials, timeout=60)

    # Apply best params
    _apply_trial_params(g, tunable, study.best_params, source="dict")

    return g, study.best_value, trial_log


# ── Main sweep loop ───────────────────────────────────────────────

def load_target(name: str):
    import importlib
    m = importlib.import_module(f"examples.target_{name}")
    return m.TARGET, m.W, m.H


def sweep(target_name: str, n_steps: int = 100, bo_trials: int = 30,
          out_dir: str | None = None, use_gpu: bool = False):
    """Run BO sweep on a target. Returns experience log."""
    target, w, h = load_target(target_name)

    # Initialize GPU rasterizer if requested
    gpu_rast = None
    target_gpu = None
    if use_gpu:
        from gpu_rasterizer import GPURasterizer
        gpu_rast = GPURasterizer(w, h, "cuda")
        target_gpu = gpu_rast.target_from_pixels(target)

    if out_dir is None:
        gpu_tag = "_gpu" if use_gpu else ""
        out_dir = f"runs/bo_sweep_{target_name}{gpu_tag}_{int(time.time())}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(out_dir) / "experiences.jsonl"

    genome = Genome(bg_r=128, bg_g=128, bg_b=128)
    fitness, sim, mse, regions = _eval_genome(genome, target, w, h, gpu_rast, target_gpu)

    best_fitness = fitness
    best_genome = genome.clone()
    experiences = []

    mode = "GPU" if use_gpu else "CPU"
    print(f"[{target_name}] Starting sweep ({mode}): {n_steps} steps, {bo_trials} BO trials each",
          flush=True)
    print(f"[{target_name}] Initial fitness: {fitness:.4f}", flush=True)

    for step in range(n_steps):
        o_t = {
            "fitness": fitness, "similarity": sim, "mse": mse,
            "region_errors": regions, "shape_count": genome.shape_count(),
            "genome": genome.to_dict(),
        }

        action_fn = random.choices(ACTIONS, weights=ACTION_WEIGHTS, k=1)[0]
        new_genome, action_info = action_fn(genome)

        if not action_info.get("success", False):
            continue

        new_genome, new_fitness, trial_log = optimize_params(
            new_genome, target, w, h, n_trials=bo_trials,
            gpu_rast=gpu_rast, target_gpu=target_gpu,
        )

        new_fitness, new_sim, new_mse, new_regions = _eval_genome(
            new_genome, target, w, h, gpu_rast, target_gpu,
        )

        reward = new_fitness - fitness

        o_t1 = {
            "fitness": new_fitness, "similarity": new_sim, "mse": new_mse,
            "region_errors": new_regions, "shape_count": new_genome.shape_count(),
            "genome": new_genome.to_dict(),
        }

        experience = {
            "o_t": o_t, "action": action_info, "o_t1": o_t1,
            "reward": reward, "target": target_name,
            "step": step, "bo_trials": len(trial_log),
            "timestamp": time.time(),
        }
        experiences.append(experience)

        with open(log_path, "a") as f:
            f.write(json.dumps(experience) + "\n")

        # Accept if improved, or sometimes accept worse (simulated annealing style)
        temperature = max(0.01, 1.0 - step / n_steps)  # cool down over time
        accept = reward > 0 or random.random() < 0.1 * temperature
        if accept:
            genome = new_genome
            fitness, sim, mse, regions = new_fitness, new_sim, new_mse, new_regions

        if new_fitness > best_fitness:
            best_fitness = new_fitness
            best_genome = new_genome.clone()
            # Save checkpoint
            best_svg = best_genome.to_svg(w, h)
            try:
                png = cairosvg.svg2png(bytestring=best_svg.encode(), output_width=w, output_height=h)
                with open(Path(out_dir) / "best.png", "wb") as f:
                    f.write(png)
                with open(Path(out_dir) / "best.svg", "w") as f:
                    f.write(best_svg)
            except Exception:
                pass

        if (step + 1) % 10 == 0:
            print(f"[{target_name}] Step {step+1}/{n_steps}: "
                  f"fitness={fitness:.4f} best={best_fitness:.4f} "
                  f"shapes={genome.shape_count()} action={action_info['action']}",
                  flush=True)

    # Final save
    best_svg = best_genome.to_svg(w, h)
    with open(Path(out_dir) / "best.svg", "w") as f:
        f.write(best_svg)
    try:
        png = cairosvg.svg2png(bytestring=best_svg.encode(), output_width=w, output_height=h)
        with open(Path(out_dir) / "best.png", "wb") as f:
            f.write(png)
    except Exception:
        pass

    print(f"[{target_name}] Done! Best fitness: {best_fitness:.4f}, "
          f"shapes: {best_genome.shape_count()}, "
          f"experiences: {len(experiences)}", flush=True)
    print(f"[{target_name}] Saved to {out_dir}/", flush=True)

    return experiences


ALL_TARGETS = ["glow", "blocks", "face", "sunset", "rings"]


def _genome_from_dict(d: dict) -> Genome:
    """Reconstruct a Genome from its dict representation."""
    g = Genome(bg_r=d["bg"][0], bg_g=d["bg"][1], bg_b=d["bg"][2],
               bg_fixed=d.get("bg_fixed", False))
    for sd in d["shapes"]:
        g.shapes.append(Shape(
            type=sd["type"],
            params=sd["params"],
            fixed=sd.get("fixed", {k: False for k in sd["params"]}),
        ))
    return g


def hindsight_relabel(experiences: list[dict], original_target: str,
                      use_gpu: bool = False) -> list[dict]:
    """Relabel experiences against all other targets (HER).

    For each experience from the original sweep, evaluate the before/after
    genomes against every other target. If the action improved fitness for
    a different target, store it as a positive experience for that target.

    Returns the list of new hindsight experiences.
    """
    other_targets = [t for t in ALL_TARGETS if t != original_target]
    if not other_targets:
        return []

    # Load all other targets
    target_data = {}
    for t in other_targets:
        pixels, w, h = load_target(t)
        target_data[t] = (pixels, w, h)

    # Initialize GPU rasterizers for each target
    gpu_rasts = {}
    target_gpus = {}
    if use_gpu:
        from gpu_rasterizer import GPURasterizer
        for t in other_targets:
            _, w, h = target_data[t]
            rast = GPURasterizer(w, h, "cuda")
            gpu_rasts[t] = rast
            target_gpus[t] = rast.target_from_pixels(target_data[t][0])

    hindsight_experiences = []
    evaluated = 0

    for exp in experiences:
        genome_before = _genome_from_dict(exp["o_t"]["genome"])
        genome_after = _genome_from_dict(exp["o_t1"]["genome"])

        for t in other_targets:
            pixels, w, h = target_data[t]
            gr = gpu_rasts.get(t)
            tg = target_gpus.get(t)

            fit_before, sim_before, mse_before, reg_before = _eval_genome(
                genome_before, pixels, w, h, gr, tg)
            fit_after, sim_after, mse_after, reg_after = _eval_genome(
                genome_after, pixels, w, h, gr, tg)
            evaluated += 2

            reward = fit_after - fit_before

            hindsight_exp = {
                "o_t": {
                    "fitness": fit_before, "similarity": sim_before,
                    "mse": mse_before, "region_errors": reg_before,
                    "shape_count": genome_before.shape_count(),
                    "genome": exp["o_t"]["genome"],
                },
                "action": exp["action"],
                "o_t1": {
                    "fitness": fit_after, "similarity": sim_after,
                    "mse": mse_after, "region_errors": reg_after,
                    "shape_count": genome_after.shape_count(),
                    "genome": exp["o_t1"]["genome"],
                },
                "reward": reward,
                "target": t,
                "step": exp["step"],
                "bo_trials": 0,  # no BO, just re-evaluation
                "timestamp": time.time(),
                "hindsight": True,
                "original_target": original_target,
            }
            hindsight_experiences.append(hindsight_exp)

    positive = sum(1 for e in hindsight_experiences if e["reward"] > 0)
    print(f"[hindsight] {original_target} -> {other_targets}: "
          f"{len(hindsight_experiences)} relabeled, {positive} positive "
          f"({evaluated} evals)", flush=True)

    return hindsight_experiences


if __name__ == "__main__":
    target_arg = os.environ.get("EVOL_TARGET", "glow")
    n_steps = int(os.environ.get("BO_STEPS", "200"))
    bo_trials = int(os.environ.get("BO_TRIALS", "30"))
    use_gpu = os.environ.get("BO_GPU", "0") == "1"
    do_hindsight = os.environ.get("BO_HINDSIGHT", "1") == "1"

    if target_arg == "all":
        targets = ALL_TARGETS
    else:
        targets = [target_arg]

    all_experiences = {}
    for t in targets:
        exps = sweep(t, n_steps=n_steps, bo_trials=bo_trials, use_gpu=use_gpu)
        all_experiences[t] = exps

    # Hindsight relabeling pass
    if do_hindsight and len(targets) > 1:
        print("\n=== Hindsight Experience Replay ===", flush=True)
        for t, exps in all_experiences.items():
            hindsight = hindsight_relabel(exps, t, use_gpu=use_gpu)
            if hindsight:
                # Save hindsight experiences alongside originals
                # Find the run dir from the sweep
                run_dirs = sorted(Path("runs").glob(f"bo_sweep_{t}*"),
                                  key=lambda p: p.stat().st_mtime, reverse=True)
                if run_dirs:
                    hindsight_path = run_dirs[0] / "hindsight.jsonl"
                    with open(hindsight_path, "w") as f:
                        for h in hindsight:
                            f.write(json.dumps(h) + "\n")
                    print(f"[{t}] Saved {len(hindsight)} hindsight experiences to "
                          f"{hindsight_path}", flush=True)

        # Summary
        total_original = sum(len(e) for e in all_experiences.values())
        total_hindsight = 0
        for t in targets:
            run_dirs = sorted(Path("runs").glob(f"bo_sweep_{t}*"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
            if run_dirs:
                hp = run_dirs[0] / "hindsight.jsonl"
                if hp.exists():
                    with open(hp) as f:
                        total_hindsight += sum(1 for _ in f)

        print(f"\n=== Total: {total_original} original + {total_hindsight} hindsight "
              f"= {total_original + total_hindsight} experiences ===", flush=True)
