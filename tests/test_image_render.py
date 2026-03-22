"""Tests for SVG image render/save helpers."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_save_svg_creates_file():
    from optimize_image import save_svg

    def simple(w, h, p=None):
        return f'<svg width="{w}" height="{h}"><rect width="{w}" height="{h}" fill="red"/></svg>'

    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
        path = f.name

    try:
        save_svg(simple, path)
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "<svg" in content
    finally:
        os.unlink(path)


def test_save_png_creates_valid_file():
    from optimize_image import save_png

    def blue(w, h, p=None):
        return (f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
                f'<rect width="{w}" height="{h}" fill="blue"/></svg>')

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name

    try:
        save_png(blue, path)
        assert os.path.exists(path)
        with open(path, "rb") as f:
            sig = f.read(8)
            assert sig == b"\x89PNG\r\n\x1a\n", "Invalid PNG signature"
    finally:
        os.unlink(path)


def test_save_with_params():
    from optimize_image import save_svg

    def parameterized(w, h, p=None):
        if p is None:
            p = {"r": 128}
        v = int(p["r"])
        return f'<svg width="{w}" height="{h}"><rect width="{w}" height="{h}" fill="rgb({v},0,0)"/></svg>'

    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
        path = f.name

    try:
        save_svg(parameterized, path, p={"r": 200})
        assert os.path.exists(path)
        with open(path) as f:
            assert "200" in f.read()
    finally:
        os.unlink(path)
