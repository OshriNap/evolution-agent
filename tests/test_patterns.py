"""Tests for meta-pattern library."""

import json
import os
import tempfile

from evolution_agent.analysis.patterns import MetaPattern, MetaPatternLibrary


def test_seed_defaults():
    lib = MetaPatternLibrary("/tmp/test_patterns_seed.json")
    lib.seed_defaults()
    patterns = lib.get_patterns()
    assert len(patterns) == 8
    ids = {p.pattern_id for p in patterns}
    assert "premature_convergence" in ids
    assert "fitness_plateau" in ids
    assert "bloat" in ids


def test_save_and_load():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        lib = MetaPatternLibrary(path)
        lib.seed_defaults()
        lib.save()

        lib2 = MetaPatternLibrary(path)
        lib2.load()
        assert len(lib2.get_patterns()) == 8
    finally:
        os.unlink(path)
        history = path.replace(".json", ".history.jsonl")
        if os.path.exists(history):
            os.unlink(history)


def test_add_and_remove_pattern():
    lib = MetaPatternLibrary("/tmp/test_patterns_ar.json")
    lib.seed_defaults()

    lib.add_pattern(MetaPattern(
        pattern_id="custom_1",
        detector="test detector",
        reframe="test reframe",
    ))
    assert len(lib.get_patterns()) == 9
    assert lib.get_pattern("custom_1") is not None

    lib.remove_pattern("custom_1")
    assert len(lib.get_patterns()) == 8
    assert lib.get_pattern("custom_1") is None


def test_update_pattern_stats():
    lib = MetaPatternLibrary("/tmp/test_patterns_upd.json")
    lib.seed_defaults()

    lib.update_pattern("bloat", detected=True)
    lib.update_pattern("bloat", detected=True, helped=True)

    pat = lib.get_pattern("bloat")
    assert pat.times_detected == 2
    assert pat.times_reframe_helped == 1
    assert pat.success_rate == 0.5


def test_merge_patterns():
    lib = MetaPatternLibrary("/tmp/test_patterns_merge.json")
    lib.seed_defaults()

    lib.add_pattern(MetaPattern(
        pattern_id="dup_1", detector="d1", reframe="r1",
        times_detected=3, times_reframe_helped=2,
    ))
    lib.add_pattern(MetaPattern(
        pattern_id="dup_2", detector="d2", reframe="r2",
        times_detected=2, times_reframe_helped=1,
    ))

    lib.merge_patterns("dup_1", "dup_2")
    merged = lib.get_pattern("dup_1")
    assert merged.times_detected == 5
    assert merged.times_reframe_helped == 3
    assert lib.get_pattern("dup_2") is None


def test_format_for_prompt():
    lib = MetaPatternLibrary("/tmp/test_patterns_fmt.json")
    lib.seed_defaults()
    text = lib.format_for_prompt()
    assert "premature_convergence" in text
    assert "Detector" in text
    assert "Reframe" in text
