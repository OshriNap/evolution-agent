"""Tests for Optuna meta-optimizer."""

import pytest

from evolution_agent.analysis.scratchpad import Scratchpad
from evolution_agent.core.types import EvolutionConfig, Generation, Individual

try:
    from evolution_agent.meta.optuna_optimizer import OptunaMetaOptimizer, OPTUNA_AVAILABLE
except ImportError:
    OPTUNA_AVAILABLE = False

pytestmark = pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")


def _make_gen(number: int, best_fitness: float, diversity: float) -> Generation:
    return Generation(
        number=number,
        individuals=[Individual(code=f"def f(): return {number}")],
        best_fitness=best_fitness,
        avg_fitness=best_fitness * 0.8,
        diversity=diversity,
    )


def test_optuna_optimizer_basic():
    cfg = EvolutionConfig(population_size=10)
    opt = OptunaMetaOptimizer(cfg, objective_mode="composite")
    sp = Scratchpad()

    # Run a few steps
    for i in range(5):
        gen = _make_gen(i, 0.5 + i * 0.1, 0.8)
        adjustments = opt.step(gen, sp)
        assert "source" in adjustments
        assert adjustments["source"] == "optuna"
        assert "mutation_weights" in adjustments

    # Config should have been modified
    assert len(cfg.mutation_weights) == 2
    assert 0.1 <= cfg.mutation_weights[0] <= 0.9


def test_optuna_optimizer_improvement_mode():
    cfg = EvolutionConfig(population_size=10)
    opt = OptunaMetaOptimizer(cfg, objective_mode="improvement")
    sp = Scratchpad()

    for i in range(3):
        gen = _make_gen(i, 0.5 + i * 0.05, 0.7)
        opt.step(gen, sp)

    best = opt.get_best_params()
    assert "n_trials" in best


def test_optuna_optimizer_multi_objective():
    cfg = EvolutionConfig(population_size=10)
    opt = OptunaMetaOptimizer(cfg, objective_mode="multi")
    sp = Scratchpad()

    for i in range(5):
        gen = _make_gen(i, 0.5 + i * 0.1, 0.8 - i * 0.05)
        opt.step(gen, sp)

    best = opt.get_best_params()
    assert "pareto_front" in best


def test_optuna_modifies_all_params():
    cfg = EvolutionConfig(
        population_size=10,
        mutation_weights=[0.6, 0.4],
        crossover_rate=0.2,
        tournament_size=3,
        elite_count=2,
    )
    opt = OptunaMetaOptimizer(cfg)
    sp = Scratchpad()

    gen = _make_gen(0, 0.5, 0.8)
    adj = opt.step(gen, sp)

    # All parameters should be set
    assert "crossover_rate" in adj
    assert "tournament_size" in adj
    assert "elite_count" in adj


def test_optuna_with_analyzer_result():
    cfg = EvolutionConfig(population_size=10)
    opt = OptunaMetaOptimizer(cfg)
    sp = Scratchpad()

    gen = _make_gen(0, 0.5, 0.8)
    opt.step(gen, sp, analysis_result={
        "phase": "exploring",
        "detected_patterns": ["premature_convergence"],
    })

    # Should not crash with analyzer data
    gen2 = _make_gen(1, 0.6, 0.7)
    opt.step(gen2, sp)


def test_optuna_scratchpad_logging():
    cfg = EvolutionConfig(population_size=10)
    opt = OptunaMetaOptimizer(cfg)
    sp = Scratchpad()

    gen = _make_gen(0, 0.5, 0.8)
    opt.step(gen, sp)

    # Should have added an observation entry
    assert len(sp.entries) == 1
    assert "Optuna trial" in sp.entries[0].content


def test_optuna_study_accessible():
    cfg = EvolutionConfig(population_size=10)
    opt = OptunaMetaOptimizer(cfg)
    assert opt.study is not None
    assert opt.study.study_name == "evolution_meta"
