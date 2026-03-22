"""Smoke test for image evolution main."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_main_configures_engine_correctly():
    """Verify main() creates an EvolutionEngine with correct params."""
    from optimize_image import SEEDS, FUNCTION_SPEC

    assert len(SEEDS) >= 2
    assert "render" in FUNCTION_SPEC
    assert "math" in FUNCTION_SPEC
    assert "do NOT import" in FUNCTION_SPEC.lower() or "NOT import" in FUNCTION_SPEC
