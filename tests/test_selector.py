"""Tests for selection strategies."""

from evolution_agent.core.selector import select_parents, select_pair_for_crossover
from evolution_agent.core.types import Individual, OptimizationDirection, SelectionStrategy


def _make_pop(n=10):
    pop = []
    for i in range(n):
        ind = Individual(code=f"def f(): return {i}", fitness=float(i))
        pop.append(ind)
    return pop


def test_tournament_selection():
    pop = _make_pop()
    parents = select_parents(pop, 5, SelectionStrategy.TOURNAMENT)
    assert len(parents) == 5
    # Tournament should bias toward higher fitness
    avg_fitness = sum(p.fitness for p in parents) / len(parents)
    assert avg_fitness > 3.0  # better than random (4.5)


def test_elite_selection():
    pop = _make_pop()
    parents = select_parents(pop, 3, SelectionStrategy.ELITE)
    assert len(parents) == 3
    # Should pick the top 3
    fitnesses = sorted([p.fitness for p in parents], reverse=True)
    assert fitnesses == [9.0, 8.0, 7.0]


def test_roulette_selection():
    pop = _make_pop()
    parents = select_parents(pop, 5, SelectionStrategy.ROULETTE)
    assert len(parents) == 5


def test_rank_selection():
    pop = _make_pop()
    parents = select_parents(pop, 5, SelectionStrategy.RANK)
    assert len(parents) == 5


def test_minimize_direction():
    pop = _make_pop()
    parents = select_parents(
        pop, 3, SelectionStrategy.ELITE,
        direction=OptimizationDirection.MINIMIZE,
    )
    # Should pick lowest fitness
    fitnesses = sorted([p.fitness for p in parents])
    assert fitnesses == [0.0, 1.0, 2.0]


def test_crossover_pair():
    pop = _make_pop()
    a, b = select_pair_for_crossover(pop)
    assert isinstance(a, Individual)
    assert isinstance(b, Individual)


def test_empty_population():
    parents = select_parents([], 5)
    assert parents == []
