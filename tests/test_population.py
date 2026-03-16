"""Tests for population management."""

from evolution_agent.core.population import PopulationManager
from evolution_agent.core.types import EvalResult, EvolutionConfig, OptimizationDirection


def test_init_from_seeds():
    pm = PopulationManager(EvolutionConfig(population_size=5))
    pop = pm.init_from_seeds(["def f(): return 1", "def f(): return 2"])
    assert len(pop) == 2
    assert pm.generation == 0


def test_cache_and_update():
    pm = PopulationManager(EvolutionConfig())
    pm.init_from_seeds(["def f(): return 1"])
    ind = pm.population[0]
    result = EvalResult(fitness=0.8, metrics={"acc": 0.9})
    pm.update_fitness(ind, result)
    assert ind.fitness == 0.8
    assert pm.get_cached_result(ind.code_hash) is result


def test_advance_generation_elitism():
    cfg = EvolutionConfig(population_size=4, elite_count=2)
    pm = PopulationManager(cfg)
    pm.init_from_seeds([f"def f(): return {i}" for i in range(4)])

    # Set fitness
    for i, ind in enumerate(pm.population):
        pm.update_fitness(ind, EvalResult(fitness=float(i)))

    # Top 2 should be ind[3] (fitness=3) and ind[2] (fitness=2)
    new_inds = []
    from evolution_agent.core.types import Individual
    for j in range(2):
        new_inds.append(Individual(code=f"def f(): return {10+j}", fitness=1.5))

    gen = pm.advance_generation(new_inds)
    assert gen.number == 1
    # Best fitness should be from elite (3.0)
    assert gen.best_fitness == 3.0
    assert len(pm.population) <= 4


def test_diversity():
    pm = PopulationManager(EvolutionConfig(population_size=3))
    pm.init_from_seeds(["def f(): return 1", "def f(): return 2", "def f(): return 3"])
    assert pm.compute_diversity() == 1.0

    # Duplicate code → lower diversity
    pm.init_from_seeds(["def f(): return 1", "def f(): return 1", "def f(): return 2"])
    assert pm.compute_diversity() < 1.0


def test_hall_of_fame():
    cfg = EvolutionConfig(population_size=3, elite_count=2)
    pm = PopulationManager(cfg)
    pm.init_from_seeds([f"def f(): return {i}" for i in range(3)])
    for i, ind in enumerate(pm.population):
        pm.update_fitness(ind, EvalResult(fitness=float(i)))

    from evolution_agent.core.types import Individual
    new = [Individual(code="def f(): return 99", fitness=0.5)]
    pm.advance_generation(new)

    assert len(pm.hall_of_fame) > 0
    assert pm.hall_of_fame[0].fitness == 2.0  # best ever


def test_minimize_direction():
    cfg = EvolutionConfig(population_size=3, elite_count=1, direction=OptimizationDirection.MINIMIZE)
    pm = PopulationManager(cfg)
    pm.init_from_seeds([f"def f(): return {i}" for i in range(3)])
    for i, ind in enumerate(pm.population):
        pm.update_fitness(ind, EvalResult(fitness=float(i + 1)))  # 1, 2, 3

    best = pm.get_best()
    assert best.fitness == 1.0  # lowest is best for minimize
