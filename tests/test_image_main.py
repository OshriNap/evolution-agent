"""Smoke test for SVG image evolution main."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_main_configures_engine_correctly():
    """Verify the pieces exist and are well-formed."""
    from optimize_image import SEEDS, FUNCTION_SPEC

    assert len(SEEDS) >= 2
    assert "generate_svg" in FUNCTION_SPEC
    assert "SVG" in FUNCTION_SPEC or "svg" in FUNCTION_SPEC
    assert "do NOT import" in FUNCTION_SPEC.lower() or "NOT import" in FUNCTION_SPEC
