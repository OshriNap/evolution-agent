"""GPU-accelerated shape rasterizer using PyTorch.

Renders shapes directly to GPU tensors — no SVG string generation or
Cairo rasterization needed. Supports compositing with alpha blending,
gradients, blur, and all shape types from the BO sweep.

Usage:
    from gpu_rasterizer import GPURasterizer

    rasterizer = GPURasterizer(64, 64, device='cuda')
    canvas = rasterizer.render(genome, target_tensor)
    fitness = rasterizer.fitness(canvas, target_tensor)

    # Batched: render N genomes at once
    fitnesses = rasterizer.batch_fitness(genomes, target_tensor)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


class GPURasterizer:
    def __init__(self, w: int = 64, h: int = 64, device: str = "cuda"):
        self.w = w
        self.h = h
        self.device = torch.device(device)

        # Precompute coordinate grids (reused for every render)
        ys = torch.arange(h, device=self.device, dtype=torch.float32)
        xs = torch.arange(w, device=self.device, dtype=torch.float32)
        self.grid_y, self.grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # Normalized 0..1
        self.grid_nx = self.grid_x / w
        self.grid_ny = self.grid_y / h

    def target_from_pixels(self, pixels: list[tuple[int, int, int]]) -> torch.Tensor:
        """Convert pixel list to [3, H, W] float tensor on device."""
        t = torch.tensor(pixels, dtype=torch.float32, device=self.device)
        t = t.reshape(self.h, self.w, 3).permute(2, 0, 1)  # [3, H, W]
        return t / 255.0

    def render(self, genome) -> torch.Tensor:
        """Render a Genome to a [3, H, W] float tensor in [0, 1]."""
        # Background
        canvas = torch.zeros(3, self.h, self.w, device=self.device)
        canvas[0] = genome.bg_r / 255.0
        canvas[1] = genome.bg_g / 255.0
        canvas[2] = genome.bg_b / 255.0

        for shape in genome.shapes:
            canvas = self._composite_shape(canvas, shape)

        return canvas.clamp(0, 1)

    def fitness(self, canvas: torch.Tensor, target: torch.Tensor,
                shape_count: int = 0) -> tuple[float, float, list[float]]:
        """Compute fitness from rendered canvas vs target.

        Returns (fitness, similarity, region_errors).
        """
        diff = canvas - target
        mse_per_pixel = (diff * diff).sum(dim=0)  # [H, W]

        total_mse = mse_per_pixel.mean().item() * 3  # scale to match CPU version
        norm_mse = total_mse / 3.0  # already in [0,1] range since tensors are 0..1
        similarity = 1.0 / (1.0 + norm_mse * 1000)
        fit = similarity / (1.0 + 0.002 * shape_count)

        # Per-quadrant errors
        hh, hw = self.h // 2, self.w // 2
        regions = [
            mse_per_pixel[:hh, :hw].mean().item(),
            mse_per_pixel[:hh, hw:].mean().item(),
            mse_per_pixel[hh:, :hw].mean().item(),
            mse_per_pixel[hh:, hw:].mean().item(),
        ]

        return fit, similarity, regions

    def batch_render_and_fitness(
        self, genomes: list, target: torch.Tensor,
    ) -> list[tuple[float, float, list[float]]]:
        """Render and evaluate multiple genomes. Could be parallelized further."""
        results = []
        for g in genomes:
            canvas = self.render(g)
            results.append(self.fitness(canvas, target, g.shape_count()))
        return results

    # ── Shape compositing ──────────────────────────────────────────

    def _composite_shape(self, canvas: torch.Tensor, shape) -> torch.Tensor:
        """Alpha-composite a shape onto the canvas."""
        p = shape.params
        stype = shape.type

        if stype == "circle":
            return self._draw_circle(canvas, p)
        elif stype == "rect":
            return self._draw_rect(canvas, p)
        elif stype == "ellipse":
            return self._draw_ellipse(canvas, p)
        elif stype == "line":
            return self._draw_line(canvas, p)
        elif stype == "polygon":
            return self._draw_polygon(canvas, p)
        elif stype == "radial_gradient_circle":
            return self._draw_radial_gradient_circle(canvas, p)
        elif stype == "radial_gradient_rect":
            return self._draw_radial_gradient_rect(canvas, p)
        elif stype == "linear_gradient_rect":
            return self._draw_linear_gradient_rect(canvas, p)
        elif stype == "blurred_circle":
            return self._draw_blurred_circle(canvas, p)
        elif stype == "blurred_ellipse":
            return self._draw_blurred_ellipse(canvas, p)
        elif stype == "ring":
            return self._draw_ring(canvas, p)
        return canvas

    def _alpha_blend(self, canvas: torch.Tensor, color: torch.Tensor,
                     mask: torch.Tensor, opacity: float) -> torch.Tensor:
        """Blend color onto canvas using mask and opacity. All [3,H,W] or [H,W]."""
        alpha = mask * opacity  # [H, W]
        alpha = alpha.unsqueeze(0)  # [1, H, W]
        return canvas * (1 - alpha) + color * alpha

    def _draw_circle(self, canvas, p):
        cx, cy, r = p["cx"], p["cy"], p["r"]
        dist = torch.sqrt((self.grid_x - cx) ** 2 + (self.grid_y - cy) ** 2)
        mask = (dist < r).float()
        # Anti-alias edge
        mask = torch.clamp(r - dist, 0, 1).clamp(0, 1)

        color = torch.zeros(3, self.h, self.w, device=self.device)
        color[0] = p["fill_r"] / 255.0
        color[1] = p["fill_g"] / 255.0
        color[2] = p["fill_b"] / 255.0

        return self._alpha_blend(canvas, color, mask, p.get("opacity", 1.0))

    def _draw_rect(self, canvas, p):
        x, y, w, h = p["x"], p["y"], p["w"], p["h"]
        rx = p.get("rx", 0)

        if rx < 0.5:
            # Simple rect with AA
            mask_x = torch.clamp(torch.min(self.grid_x - x, x + w - self.grid_x), 0, 1)
            mask_y = torch.clamp(torch.min(self.grid_y - y, y + h - self.grid_y), 0, 1)
            mask = torch.min(mask_x, mask_y)
        else:
            # Rounded rect via SDF
            # Clamp point to inner rect, compute distance to clamped point
            px = torch.clamp(self.grid_x, x + rx, x + w - rx)
            py = torch.clamp(self.grid_y, y + rx, y + h - rx)
            dist = torch.sqrt((self.grid_x - px) ** 2 + (self.grid_y - py) ** 2)
            inside_inner = ((self.grid_x >= x) & (self.grid_x <= x + w) &
                           (self.grid_y >= y) & (self.grid_y <= y + h))
            mask = torch.where(inside_inner, torch.clamp(rx - dist, 0, 1), torch.zeros_like(dist))

        color = torch.zeros(3, self.h, self.w, device=self.device)
        color[0] = p["fill_r"] / 255.0
        color[1] = p["fill_g"] / 255.0
        color[2] = p["fill_b"] / 255.0

        return self._alpha_blend(canvas, color, mask, p.get("opacity", 1.0))

    def _draw_ellipse(self, canvas, p):
        cx, cy, rx, ry = p["cx"], p["cy"], p["rx"], p["ry"]
        dist = ((self.grid_x - cx) / max(rx, 0.1)) ** 2 + ((self.grid_y - cy) / max(ry, 0.1)) ** 2
        mask = torch.clamp(1.0 - dist, 0, 1)

        color = torch.zeros(3, self.h, self.w, device=self.device)
        color[0] = p["fill_r"] / 255.0
        color[1] = p["fill_g"] / 255.0
        color[2] = p["fill_b"] / 255.0

        return self._alpha_blend(canvas, color, mask, p.get("opacity", 1.0))

    def _draw_line(self, canvas, p):
        x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
        sw = p.get("stroke_width", 2.0) / 2.0

        # Distance from point to line segment
        dx, dy = x2 - x1, y2 - y1
        line_len_sq = dx * dx + dy * dy
        if line_len_sq < 0.01:
            return canvas

        t = ((self.grid_x - x1) * dx + (self.grid_y - y1) * dy) / line_len_sq
        t = t.clamp(0, 1)
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        dist = torch.sqrt((self.grid_x - proj_x) ** 2 + (self.grid_y - proj_y) ** 2)
        mask = torch.clamp(sw - dist, 0, 1)

        color = torch.zeros(3, self.h, self.w, device=self.device)
        color[0] = p["stroke_r"] / 255.0
        color[1] = p["stroke_g"] / 255.0
        color[2] = p["stroke_b"] / 255.0

        return self._alpha_blend(canvas, color, mask, p.get("opacity", 1.0))

    def _draw_polygon(self, canvas, p):
        cx, cy, r = p["cx"], p["cy"], p["r"]
        sides = max(3, int(p["sides"]))
        rot = p.get("rotation", 0)

        # Approximate polygon as intersection of half-planes
        # For each edge, compute signed distance
        mask = torch.ones(self.h, self.w, device=self.device)
        for i in range(sides):
            a1 = math.radians(rot + 360 * i / sides)
            a2 = math.radians(rot + 360 * (i + 1) / sides)
            px1, py1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
            px2, py2 = cx + r * math.cos(a2), cy + r * math.sin(a2)
            # Edge normal (inward)
            ex, ey = px2 - px1, py2 - py1
            nx, ny = -ey, ex  # normal pointing inward
            d = (self.grid_x - px1) * nx + (self.grid_y - py1) * ny
            mask = torch.min(mask, torch.clamp(d, 0, 1))

        color = torch.zeros(3, self.h, self.w, device=self.device)
        color[0] = p["fill_r"] / 255.0
        color[1] = p["fill_g"] / 255.0
        color[2] = p["fill_b"] / 255.0

        return self._alpha_blend(canvas, color, mask, p.get("opacity", 1.0))

    def _draw_radial_gradient_circle(self, canvas, p):
        cx, cy, r = p["cx"], p["cy"], max(p["r"], 0.1)
        dist = torch.sqrt((self.grid_x - cx) ** 2 + (self.grid_y - cy) ** 2)
        t = (dist / r).clamp(0, 1)  # 0 at center, 1 at edge
        mask = torch.clamp(1.0 - (dist - r), 0, 1)  # AA at boundary

        inner = torch.zeros(3, self.h, self.w, device=self.device)
        inner[0] = p["inner_r"] / 255.0
        inner[1] = p["inner_g"] / 255.0
        inner[2] = p["inner_b"] / 255.0

        outer = torch.zeros(3, self.h, self.w, device=self.device)
        outer[0] = p["outer_r"] / 255.0
        outer[1] = p["outer_g"] / 255.0
        outer[2] = p["outer_b"] / 255.0

        t3 = t.unsqueeze(0)  # [1, H, W]
        color = inner * (1 - t3) + outer * t3

        io = p.get("inner_opacity", 1.0)
        oo = p.get("outer_opacity", 1.0)
        opacity = io * (1 - t) + oo * t

        return self._alpha_blend(canvas, color, mask, 1.0) * (1 - opacity.unsqueeze(0) * mask.unsqueeze(0)) + \
               self._alpha_blend(canvas, color, mask * opacity, 1.0)

    def _draw_radial_gradient_rect(self, canvas, p):
        # Full-canvas radial gradient
        gcx = p.get("gcx", 0.5)
        gcy = p.get("gcy", 0.5)
        gr = max(p.get("gr", 0.5), 0.01)

        dist = torch.sqrt((self.grid_nx - gcx) ** 2 + (self.grid_ny - gcy) ** 2)
        t = (dist / gr).clamp(0, 1)

        mid_offset = p.get("mid_offset", 0.5)

        inner = torch.zeros(3, self.h, self.w, device=self.device)
        inner[0] = p["inner_r"] / 255.0
        inner[1] = p["inner_g"] / 255.0
        inner[2] = p["inner_b"] / 255.0

        mid = torch.zeros(3, self.h, self.w, device=self.device)
        mid[0] = p["mid_r"] / 255.0
        mid[1] = p["mid_g"] / 255.0
        mid[2] = p["mid_b"] / 255.0

        outer = torch.zeros(3, self.h, self.w, device=self.device)
        outer[0] = p["outer_r"] / 255.0
        outer[1] = p["outer_g"] / 255.0
        outer[2] = p["outer_b"] / 255.0

        t3 = t.unsqueeze(0)
        # Two-segment gradient: inner→mid→outer
        inner_to_mid = torch.where(t3 < mid_offset,
                                    inner * (1 - t3 / mid_offset) + mid * (t3 / mid_offset),
                                    mid)
        color = torch.where(t3 < mid_offset,
                           inner_to_mid,
                           mid * (1 - (t3 - mid_offset) / (1 - mid_offset + 1e-6)) +
                           outer * ((t3 - mid_offset) / (1 - mid_offset + 1e-6)))

        io = p.get("inner_opacity", 1.0)
        oo = p.get("outer_opacity", 1.0)
        opacity = io * (1 - t) + oo * t

        mask = torch.ones(self.h, self.w, device=self.device)
        return self._alpha_blend(canvas, color, mask * opacity, 1.0)

    def _draw_linear_gradient_rect(self, canvas, p):
        x1p, y1p = p["x1_pct"], p["y1_pct"]
        x2p, y2p = p["x2_pct"], p["y2_pct"]

        # Project each pixel onto gradient line
        dx, dy = x2p - x1p, y2p - y1p
        line_len_sq = dx * dx + dy * dy
        if line_len_sq < 1e-6:
            return canvas

        t = ((self.grid_nx - x1p) * dx + (self.grid_ny - y1p) * dy) / line_len_sq
        t = t.clamp(0, 1)

        start = torch.zeros(3, self.h, self.w, device=self.device)
        start[0] = p["start_r"] / 255.0
        start[1] = p["start_g"] / 255.0
        start[2] = p["start_b"] / 255.0

        end = torch.zeros(3, self.h, self.w, device=self.device)
        end[0] = p["end_r"] / 255.0
        end[1] = p["end_g"] / 255.0
        end[2] = p["end_b"] / 255.0

        t3 = t.unsqueeze(0)
        color = start * (1 - t3) + end * t3

        so = p.get("start_opacity", 1.0)
        eo = p.get("end_opacity", 1.0)
        opacity = so * (1 - t) + eo * t

        # Rect mask
        rx, ry = p.get("rect_x", 0), p.get("rect_y", 0)
        rw, rh = p.get("rect_w", 64), p.get("rect_h", 64)
        mask_x = torch.clamp(torch.min(self.grid_x - rx, rx + rw - self.grid_x), 0, 1)
        mask_y = torch.clamp(torch.min(self.grid_y - ry, ry + rh - self.grid_y), 0, 1)
        mask = torch.min(mask_x, mask_y)

        return self._alpha_blend(canvas, color, mask * opacity, 1.0)

    def _draw_blurred_circle(self, canvas, p):
        cx, cy, r = p["cx"], p["cy"], p["r"]
        blur = p.get("blur", 5.0)

        dist = torch.sqrt((self.grid_x - cx) ** 2 + (self.grid_y - cy) ** 2)
        mask = torch.clamp(r - dist, 0, 1)

        color = torch.zeros(3, self.h, self.w, device=self.device)
        color[0] = p["fill_r"] / 255.0
        color[1] = p["fill_g"] / 255.0
        color[2] = p["fill_b"] / 255.0

        # Apply Gaussian blur to the composited result
        blurred = color * mask.unsqueeze(0) * p.get("opacity", 1.0)
        k = max(3, int(blur * 3) | 1)  # kernel size must be odd
        blurred = self._gaussian_blur(blurred, blur, k)
        alpha = self._gaussian_blur(mask.unsqueeze(0) * p.get("opacity", 1.0), blur, k)

        return canvas * (1 - alpha) + blurred

    def _draw_blurred_ellipse(self, canvas, p):
        cx, cy, rx, ry = p["cx"], p["cy"], max(p["rx"], 0.1), max(p["ry"], 0.1)
        blur = p.get("blur", 5.0)

        dist = ((self.grid_x - cx) / rx) ** 2 + ((self.grid_y - cy) / ry) ** 2
        mask = torch.clamp(1.0 - dist, 0, 1)

        color = torch.zeros(3, self.h, self.w, device=self.device)
        color[0] = p["fill_r"] / 255.0
        color[1] = p["fill_g"] / 255.0
        color[2] = p["fill_b"] / 255.0

        blurred = color * mask.unsqueeze(0) * p.get("opacity", 1.0)
        k = max(3, int(blur * 3) | 1)
        blurred = self._gaussian_blur(blurred, blur, k)
        alpha = self._gaussian_blur(mask.unsqueeze(0) * p.get("opacity", 1.0), blur, k)

        return canvas * (1 - alpha) + blurred

    def _draw_ring(self, canvas, p):
        cx, cy, r = p["cx"], p["cy"], p["r"]
        sw = p.get("stroke_width", 3.0) / 2.0

        dist = torch.sqrt((self.grid_x - cx) ** 2 + (self.grid_y - cy) ** 2)
        ring_dist = torch.abs(dist - r)
        mask = torch.clamp(sw - ring_dist, 0, 1)

        color = torch.zeros(3, self.h, self.w, device=self.device)
        color[0] = p["stroke_r"] / 255.0
        color[1] = p["stroke_g"] / 255.0
        color[2] = p["stroke_b"] / 255.0

        return self._alpha_blend(canvas, color, mask, p.get("opacity", 1.0))

    def _gaussian_blur(self, tensor: torch.Tensor, sigma: float,
                       kernel_size: int) -> torch.Tensor:
        """Apply Gaussian blur to a [C, H, W] tensor."""
        if sigma < 0.1:
            return tensor
        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size, device=self.device, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Separable 2D blur
        c = tensor.shape[0]
        t = tensor.unsqueeze(0)  # [1, C, H, W]
        pad = kernel_size // 2

        # Horizontal
        k_h = kernel_1d.reshape(1, 1, 1, -1).expand(c, 1, 1, -1)
        t = F.pad(t, (pad, pad, 0, 0), mode='reflect')
        t = F.conv2d(t, k_h, groups=c)

        # Vertical
        k_v = kernel_1d.reshape(1, 1, -1, 1).expand(c, 1, -1, 1)
        t = F.pad(t, (0, 0, pad, pad), mode='reflect')
        t = F.conv2d(t, k_v, groups=c)

        return t.squeeze(0).clamp(0, 1)
