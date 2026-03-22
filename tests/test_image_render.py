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
