"""Tests for meta-optimizer."""

from evolution_agent.analysis.scratchpad import Scratchpad
from evolution_agent.core.types import EvolutionConfig, Generation, Individual
from evolution_agent.meta.optimizer import MetaOptimizer


def _make_gen(number: int, best_fitness: float, diversity: float) -> Generation:
    return Generation(
        number=number,
        individuals=[Individual(code=f"def f(): return {number}")],
        best_fitness=best_fitness,
        avg_fitness=best_fitness * 0.8,
        diversity=diversity,
    )


def test_stagnation_shifts_weights():
    cfg = EvolutionConfig()
    opt = MetaOptimizer(cfg)
    sp = Scratchpad()

    # Simulate stagnation (same best fitness for 6 gens)
    for i in range(6):
        gen = _make_gen(i, 0.5, 0.8)
        opt.step(gen, sp)

    # Should have shifted toward structural
    assert cfg.mutation_weights[1] > 0.5  # structural weight increased


def test_low_diversity_increases_crossover():
    cfg = EvolutionConfig(crossover_rate=0.1)
    opt = MetaOptimizer(cfg)
    sp = Scratchpad()

    gen = _make_gen(1, 0.5, 0.2)  # low diversity
    opt.step(gen, sp)

    assert cfg.crossover_rate > 0.1


def test_normal_conditions_no_changes():
    cfg = EvolutionConfig()
    original_weights = list(cfg.mutation_weights)
    original_crossover = cfg.crossover_rate
    opt = MetaOptimizer(cfg)
    sp = Scratchpad()

    # First gen always has no last_best, so step 1 sets it
    gen = _make_gen(1, 0.5, 0.8)
    opt.step(gen, sp)

    # Second gen with improvement
    gen2 = _make_gen(2, 0.6, 0.8)
    adjustments = opt.step(gen2, sp)

    assert cfg.mutation_weights == original_weights
    assert cfg.crossover_rate == original_crossover


def test_analyzer_recommendations_applied():
    cfg = EvolutionConfig()
    opt = MetaOptimizer(cfg)
    sp = Scratchpad()

    gen = _make_gen(1, 0.5, 0.8)
    opt.step(gen, sp, analysis_result={
        "recommended_mutation_weights": {"point": 0.8, "structural": 0.2},
    })

    assert abs(cfg.mutation_weights[0] - 0.8) < 0.01
    assert abs(cfg.mutation_weights[1] - 0.2) < 0.01
