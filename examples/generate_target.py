"""Generate diverse 64x64 target images as Python constants.

Each target tests different SVG capabilities:
  - glow: radial gradient (current default)
  - blocks: geometric color blocks (sharp edges, no gradients)
  - face: simple smiley face (circles + arcs)
  - sunset: horizontal gradient with a circle (layered gradients)
  - rings: concentric rings (periodic radial pattern)

Usage:
    python examples/generate_target.py          # generate all targets
    python examples/generate_target.py glow     # generate one target
"""
import math
import sys

W, H = 64, 64


def generate_glow():
    """Bright radial glow on a dark background."""
    pixels = []
    cx, cy = W / 2, H / 2
    max_dist = math.sqrt(cx * cx + cy * cy)
    for y in range(H):
        for x in range(W):
            nx, ny = x / W, y / H
            dx, dy = x - cx, y - cy
            dist = math.sqrt(dx * dx + dy * dy) / max_dist
            bg_r = int(40 + 60 * ny)
            bg_g = int(20 + 30 * nx)
            bg_b = int(80 + 100 * (1 - ny))
            circle = max(0.0, 1.0 - dist * 1.8)
            circle = circle * circle
            r = int(min(255, bg_r + circle * (255 - bg_r)))
            g = int(min(255, bg_g + circle * (240 - bg_g)))
            b = int(min(255, bg_b + circle * (200 - bg_b)))
            pixels.append((r, g, b))
    return pixels


def generate_blocks():
    """4 colored blocks with sharp edges."""
    pixels = []
    colors = [
        (220, 60, 60),   # top-left: red
        (60, 180, 60),   # top-right: green
        (60, 60, 220),   # bottom-left: blue
        (220, 200, 60),  # bottom-right: yellow
    ]
    for y in range(H):
        for x in range(W):
            qi = (0 if y < H // 2 else 2) + (0 if x < W // 2 else 1)
            pixels.append(colors[qi])
    return pixels


def generate_face():
    """Simple smiley face — circles on yellow background."""
    pixels = []
    cx, cy = W / 2, H / 2
    for y in range(H):
        for x in range(W):
            dx, dy = x - cx, y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            # Yellow background
            r, g, b = 255, 220, 50
            # Face circle (radius 28)
            if dist > 28:
                r, g, b = 80, 80, 80
            # Left eye (center 22,24, radius 4)
            ex, ey = x - 22, y - 24
            if math.sqrt(ex * ex + ey * ey) < 4:
                r, g, b = 40, 40, 40
            # Right eye (center 42,24, radius 4)
            ex, ey = x - 42, y - 24
            if math.sqrt(ex * ex + ey * ey) < 4:
                r, g, b = 40, 40, 40
            # Smile (arc: center 32,34, radius 14, bottom half)
            sx, sy = x - 32, y - 34
            sdist = math.sqrt(sx * sx + sy * sy)
            if 12 < sdist < 16 and sy > 0:
                r, g, b = 40, 40, 40
            pixels.append((r, g, b))
    return pixels


def generate_sunset():
    """Horizontal gradient sky with sun circle."""
    pixels = []
    sun_cx, sun_cy = 32, 24
    sun_r = 12
    for y in range(H):
        for x in range(W):
            ny = y / H
            # Sky gradient: dark blue top → orange bottom
            sky_r = int(20 + 220 * ny)
            sky_g = int(10 + 100 * ny)
            sky_b = int(140 - 100 * ny)
            # Sun
            dx, dy = x - sun_cx, y - sun_cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < sun_r:
                t = dist / sun_r
                r = int(255 * (1 - t * 0.2))
                g = int(230 * (1 - t * 0.3))
                b = int(100 * (1 - t * 0.5))
            elif dist < sun_r + 8:
                # Glow around sun
                t = (dist - sun_r) / 8
                glow = (1 - t) ** 2
                r = int(sky_r + glow * (255 - sky_r))
                g = int(sky_g + glow * (200 - sky_g))
                b = int(sky_b + glow * (80 - sky_b))
            else:
                r, g, b = sky_r, sky_g, sky_b
            pixels.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
    return pixels


def generate_rings():
    """Concentric colored rings."""
    pixels = []
    cx, cy = W / 2, H / 2
    ring_colors = [
        (255, 50, 50),    # red
        (50, 255, 50),    # green
        (50, 50, 255),    # blue
        (255, 255, 50),   # yellow
        (255, 50, 255),   # magenta
    ]
    for y in range(H):
        for x in range(W):
            dx, dy = x - cx, y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            ring_idx = int(dist / 7) % len(ring_colors)
            pixels.append(ring_colors[ring_idx])
    return pixels


GENERATORS = {
    "glow": generate_glow,
    "blocks": generate_blocks,
    "face": generate_face,
    "sunset": generate_sunset,
    "rings": generate_rings,
}


def _describe_target(name, pixels):
    """Generate a text description of the target for the FUNCTION_SPEC."""
    rs = [r for r, g, b in pixels]
    gs = [g for r, g, b in pixels]
    bs = [b for r, g, b in pixels]

    # Per-quadrant averages
    quads = {}
    quad_names = {0: "Top-Left", 1: "Top-Right", 2: "Bottom-Left", 3: "Bottom-Right"}
    for qi in range(4):
        qr, qg, qb, n = 0, 0, 0, 0
        for y in range(H):
            for x in range(W):
                if ((y < H // 2) == (qi < 2)) and ((x < W // 2) == (qi % 2 == 0)):
                    r, g, b = pixels[y * W + x]
                    qr += r; qg += g; qb += b; n += 1
        quads[quad_names[qi]] = (qr // n, qg // n, qb // n)

    lines = [
        f"    # === TARGET IMAGE: {name.upper()} ===",
        f"    # Channel ranges: R=[{min(rs)}-{max(rs)}], G=[{min(gs)}-{max(gs)}], B=[{min(bs)}-{max(bs)}]",
        f"    # Overall average: ({sum(rs)//len(rs)}, {sum(gs)//len(gs)}, {sum(bs)//len(bs)})",
        "    # Per-quadrant average colors:",
    ]
    for qname, (qr, qg, qb) in quads.items():
        lines.append(f"    #   {qname}: ({qr}, {qg}, {qb})")

    return "\n".join(lines)


def write_target(name):
    gen_fn = GENERATORS[name]
    pixels = gen_fn()
    desc = _describe_target(name, pixels)

    out_path = f"examples/target_{name}.py"
    lines = [
        f'# Auto-generated 64x64 target image: {name}',
        f'W, H = {W}, {H}',
        f'TARGET = {pixels!r}',
        f'',
        f'DESCRIPTION = """{desc}"""',
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  {name}: wrote {len(pixels)} pixels to {out_path}")


if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else list(GENERATORS.keys())
    for name in names:
        if name not in GENERATORS:
            print(f"Unknown target: {name}. Available: {list(GENERATORS.keys())}")
            continue
        write_target(name)
