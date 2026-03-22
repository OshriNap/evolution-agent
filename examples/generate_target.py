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
