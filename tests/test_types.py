"""Tests for core types."""

from evolution_agent.core.types import (
    EvalResult,
    EvolutionConfig,
    Individual,
    MutationType,
    OptimizationDirection,
    SelectionStrategy,
)


def test_individual_id_is_code_hash():
    ind = Individual(code="def f(): return 1")
    assert len(ind.id) == 16
    assert ind.id == ind.code_hash


def test_individual_different_code_different_id():
    a = Individual(code="def f(): return 1")
    b = Individual(code="def f(): return 2")
    assert a.id != b.id


def test_individual_same_code_same_id():
    a = Individual(code="def f(): return 1")
    b = Individual(code="def f(): return 1")
    assert a.id == b.id


def test_individual_to_dict():
    ind = Individual(code="def f(): return 1", fitness=0.5, generation=3)
    d = ind.to_dict()
    assert d["fitness"] == 0.5
    assert d["generation"] == 3
    assert d["code"] == "def f(): return 1"
    assert d["id"] == ind.id


def test_evolution_config_defaults():
    cfg = EvolutionConfig()
    assert cfg.population_size == 20
    assert cfg.elite_count == 3
    assert cfg.direction == OptimizationDirection.MAXIMIZE


def test_evolution_config_to_dict():
    cfg = EvolutionConfig()
    d = cfg.to_dict()
    assert d["direction"] == "maximize"
    assert d["selection_strategy"] == "tournament"
