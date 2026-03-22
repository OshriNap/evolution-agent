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
