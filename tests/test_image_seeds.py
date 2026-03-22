"""Tests for image evolution seed functions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evolution_agent.evaluation.sandbox import CodeSandbox


def test_seeds_compile_in_sandbox():
    from optimize_image import SEEDS
    sandbox = CodeSandbox()

    for i, seed_code in enumerate(SEEDS):
        fn = sandbox.compile_function(seed_code, "render")
        assert fn is not None, f"Seed {i} failed to compile in sandbox"


def test_seeds_return_valid_rgb():
    from optimize_image import SEEDS
    sandbox = CodeSandbox()

    for i, seed_code in enumerate(SEEDS):
        fn = sandbox.compile_function(seed_code, "render")
        r, g, b = fn(0, 0, 64, 64)
        assert 0 <= r <= 255, f"Seed {i}: r={r} out of range"
        assert 0 <= g <= 255, f"Seed {i}: g={g} out of range"
        assert 0 <= b <= 255, f"Seed {i}: b={b} out of range"


def test_seeds_produce_different_outputs():
    from optimize_image import SEEDS
    sandbox = CodeSandbox()
    outputs = []

    for seed_code in SEEDS:
        fn = sandbox.compile_function(seed_code, "render")
        sample = tuple(fn(x, y, 64, 64) for x, y in [(0, 0), (32, 32), (63, 63)])
        outputs.append(sample)

    unique = len(set(outputs))
    assert unique >= 2, f"Seeds produce too-similar outputs: {unique} unique of {len(SEEDS)}"


def test_seeds_get_nonzero_fitness():
    from optimize_image import SEEDS, evaluate_renderer
    sandbox = CodeSandbox()

    for i, seed_code in enumerate(SEEDS):
        fn = sandbox.compile_function(seed_code, "render")
        fitness, metrics = evaluate_renderer(fn)
        assert fitness > 0, f"Seed {i} got zero fitness"
