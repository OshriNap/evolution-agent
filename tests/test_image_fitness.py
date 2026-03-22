"""Tests for the SVG image evolution fitness function."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_good_svg_gets_nonzero_fitness():
    from optimize_image import evaluate_svg

    def simple_svg(w, h, p=None):
        return (f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
                f'<rect width="{w}" height="{h}" fill="rgb(83,50,135)"/></svg>')

    fitness, metrics = evaluate_svg(simple_svg)
    assert fitness > 0, f"Valid SVG should get >0 fitness, got {fitness}"
    assert metrics["svg_valid"] is True
    assert metrics["shape_count"] == 1


def test_crashing_generator_gets_zero():
    from optimize_image import evaluate_svg

    def crasher(w, h, p=None):
        raise ValueError("boom")

    fitness, metrics = evaluate_svg(crasher)
    assert fitness == 0.0
    assert metrics["svg_valid"] is False


def test_invalid_svg_gets_zero():
    from optimize_image import evaluate_svg

    def bad_svg(w, h, p=None):
        return "not svg at all"

    fitness, metrics = evaluate_svg(bad_svg)
    assert fitness == 0.0
    assert metrics["svg_valid"] is False


def test_region_errors_is_list():
    from optimize_image import evaluate_svg

    def gray_svg(w, h, p=None):
        return (f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
                f'<rect width="{w}" height="{h}" fill="rgb(128,128,128)"/></svg>')

    fitness, metrics = evaluate_svg(gray_svg)
    assert "region_errors" in metrics
    assert isinstance(metrics["region_errors"], list)
    assert len(metrics["region_errors"]) == 4


def test_fewer_shapes_get_higher_fitness():
    from optimize_image import evaluate_svg

    def one_shape(w, h, p=None):
        return (f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
                f'<rect width="{w}" height="{h}" fill="rgb(83,50,135)"/></svg>')

    def many_shapes(w, h, p=None):
        rects = ""
        for i in range(20):
            rects += f'<rect x="0" y="{i*3}" width="{w}" height="4" fill="rgb(83,50,135)"/>'
        return (f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
                f'{rects}</svg>')

    f1, m1 = evaluate_svg(one_shape)
    f2, m2 = evaluate_svg(many_shapes)
    assert f1 > f2, f"1 shape ({f1}) should beat 20 shapes ({f2}) at similar similarity"
