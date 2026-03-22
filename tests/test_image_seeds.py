"""Tests for SVG image evolution seed functions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evolution_agent.evaluation.sandbox import CodeSandbox


def test_seeds_compile_in_sandbox():
    from optimize_image import SEEDS
    sandbox = CodeSandbox()

    for i, seed_code in enumerate(SEEDS):
        fn = sandbox.compile_function(seed_code, "generate_svg")
        assert fn is not None, f"Seed {i} failed to compile in sandbox"


def test_seeds_return_valid_svg():
    from optimize_image import SEEDS
    sandbox = CodeSandbox()

    for i, seed_code in enumerate(SEEDS):
        fn = sandbox.compile_function(seed_code, "generate_svg")
        svg = fn(64, 64)
        assert isinstance(svg, str), f"Seed {i}: expected string, got {type(svg)}"
        assert "<svg" in svg.lower(), f"Seed {i}: missing <svg> tag"
        assert "</svg>" in svg.lower(), f"Seed {i}: missing </svg> tag"


def test_seeds_produce_different_outputs():
    from optimize_image import SEEDS
    sandbox = CodeSandbox()
    outputs = []

    for seed_code in SEEDS:
        fn = sandbox.compile_function(seed_code, "generate_svg")
        svg = fn(64, 64)
        outputs.append(svg)

    unique = len(set(outputs))
    assert unique >= 2, f"Seeds produce too-similar outputs: {unique} unique of {len(SEEDS)}"


def test_seeds_get_nonzero_fitness():
    from optimize_image import SEEDS, evaluate_svg
    sandbox = CodeSandbox()

    for i, seed_code in enumerate(SEEDS):
        fn = sandbox.compile_function(seed_code, "generate_svg")
        fitness, metrics = evaluate_svg(fn)
        assert fitness > 0, f"Seed {i} got zero fitness"
        assert metrics["svg_valid"] is True
