"""Selection strategies for choosing parents."""

from __future__ import annotations

import random
from typing import Sequence

from evolution_agent.core.types import (
    Individual,
    OptimizationDirection,
    SelectionStrategy,
)


def select_parents(
    population: Sequence[Individual],
    count: int,
    strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT,
    direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
    tournament_size: int = 3,
) -> list[Individual]:
    """Select `count` parents from population using the given strategy."""
    if not population:
        return []

    selectors = {
        SelectionStrategy.TOURNAMENT: _tournament,
        SelectionStrategy.ELITE: _elite,
        SelectionStrategy.ROULETTE: _roulette,
        SelectionStrategy.RANK: _rank,
    }

    selector = selectors.get(strategy, _tournament)
    return selector(list(population), count, direction, tournament_size)


def _tournament(
    population: list[Individual],
    count: int,
    direction: OptimizationDirection,
    tournament_size: int,
) -> list[Individual]:
    """Tournament selection: pick k random, keep the best."""
    selected: list[Individual] = []
    for _ in range(count):
        contestants = random.sample(
            population, min(tournament_size, len(population))
        )
        if direction == OptimizationDirection.MAXIMIZE:
            winner = max(contestants, key=lambda x: x.fitness)
        else:
            winner = min(contestants, key=lambda x: x.fitness)
        selected.append(winner)
    return selected


def _elite(
    population: list[Individual],
    count: int,
    direction: OptimizationDirection,
    _tournament_size: int,
) -> list[Individual]:
    """Elite selection: always pick the top individuals."""
    reverse = direction == OptimizationDirection.MAXIMIZE
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=reverse)
    # Cycle through top individuals if count > population
    result: list[Individual] = []
    while len(result) < count:
        for ind in sorted_pop:
            if len(result) >= count:
                break
            result.append(ind)
    return result


def _roulette(
    population: list[Individual],
    count: int,
    direction: OptimizationDirection,
    _tournament_size: int,
) -> list[Individual]:
    """Fitness-proportionate (roulette wheel) selection."""
    fitnesses = [ind.fitness for ind in population]

    # Handle minimize by inverting
    if direction == OptimizationDirection.MINIMIZE:
        max_f = max(fitnesses)
        fitnesses = [max_f - f + 1e-10 for f in fitnesses]

    # Shift to non-negative
    min_f = min(fitnesses)
    if min_f < 0:
        fitnesses = [f - min_f + 1e-10 for f in fitnesses]

    total = sum(fitnesses)
    if total == 0:
        return random.choices(population, k=count)

    weights = [f / total for f in fitnesses]
    return random.choices(population, weights=weights, k=count)


def _rank(
    population: list[Individual],
    count: int,
    direction: OptimizationDirection,
    _tournament_size: int,
) -> list[Individual]:
    """Rank-based selection: probability proportional to rank."""
    reverse = direction == OptimizationDirection.MAXIMIZE
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=reverse)
    n = len(sorted_pop)
    # Rank weights: best=n, worst=1
    weights = [n - i for i in range(n)]
    return random.choices(sorted_pop, weights=weights, k=count)


def select_pair_for_crossover(
    population: Sequence[Individual],
    direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
    tournament_size: int = 3,
) -> tuple[Individual, Individual]:
    """Select two different parents for crossover."""
    parents = select_parents(
        population, 2,
        strategy=SelectionStrategy.TOURNAMENT,
        direction=direction,
        tournament_size=tournament_size,
    )
    if len(parents) < 2:
        parents = list(population[:2]) if len(population) >= 2 else [population[0], population[0]]
    return parents[0], parents[1]
