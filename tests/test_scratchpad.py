"""Tests for scratchpad."""

from evolution_agent.analysis.scratchpad import Scratchpad, ScratchpadEntry


def test_empty_scratchpad():
    sp = Scratchpad()
    text = sp.format_for_prompt()
    assert "first generation" in text.lower()


def test_add_entries():
    sp = Scratchpad()
    sp.add(ScratchpadEntry(generation=1, category="observation", content="test obs"))
    sp.add(ScratchpadEntry(generation=1, category="suggestion", content="try X",
                           suggestion_id="S1_0"))
    assert len(sp.entries) == 2


def test_format_includes_entries():
    sp = Scratchpad()
    sp.add(ScratchpadEntry(generation=1, category="observation", content="diversity is low"))
    text = sp.format_for_prompt()
    assert "diversity is low" in text


def test_pending_suggestions():
    sp = Scratchpad()
    sp.add(ScratchpadEntry(generation=1, category="suggestion", content="try X",
                           suggestion_id="S1_0"))
    sp.add(ScratchpadEntry(generation=1, category="suggestion", content="try Y",
                           suggestion_id="S1_1", followed=True))
    pending = sp.get_pending_suggestions()
    assert len(pending) == 1
    assert pending[0].suggestion_id == "S1_0"


def test_mark_suggestion_followed():
    sp = Scratchpad()
    sp.add(ScratchpadEntry(generation=1, category="suggestion", content="try X",
                           suggestion_id="S1_0"))
    sp.mark_suggestion_followed("S1_0", True)
    assert sp.entries[0].followed is True


def test_format_full():
    sp = Scratchpad()
    sp.add(ScratchpadEntry(generation=1, category="observation", content="obs1"))
    sp.add(ScratchpadEntry(generation=2, category="conclusion", content="conc1"))
    text = sp.format_full()
    assert "obs1" in text
    assert "conc1" in text


def test_to_dict():
    sp = Scratchpad()
    sp.add(ScratchpadEntry(generation=1, category="observation", content="test"))
    d = sp.to_dict()
    assert len(d) == 1
    assert d[0]["content"] == "test"
