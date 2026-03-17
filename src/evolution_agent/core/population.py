"""Population management: init, advance, hall of fame, diversity, dedup."""

from __future__ import annotations

import logging
import random
from typing import Any

from evolution_agent.core.types import (
    EvalResult,
    EvolutionConfig,
    Generation,
    Individual,
    OptimizationDirection,
)

logger = logging.getLogger(__name__)


class PopulationManager:
    """Manages the population across generations."""

    def __init__(self, config: EvolutionConfig) -> None:
        self._config = config
        self._population: list[Individual] = []
        self._hall_of_fame: list[Individual] = []
        self._generation: int = 0
        self._generations: list[Generation] = []
        self._eval_cache: dict[str, EvalResult] = {}
        self._hof_size = max(5, config.elite_count)

    @property
    def population(self) -> list[Individual]:
        return list(self._population)

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def hall_of_fame(self) -> list[Individual]:
        return list(self._hall_of_fame)

    @property
    def generations(self) -> list[Generation]:
        return list(self._generations)

    def _is_better(self, a: float, b: float) -> bool:
        if self._config.direction == OptimizationDirection.MAXIMIZE:
            return a > b
        return a < b

    def _worst_fitness(self) -> float:
        if self._config.direction == OptimizationDirection.MAXIMIZE:
            return float("-inf")
        return float("inf")

    def init_from_seeds(self, seeds: list[str]) -> list[Individual]:
        """Initialize population from seed code strings."""
        self._population = []
        for code in seeds:
            ind = Individual(code=code, generation=0)
            self._population.append(ind)
        logger.info("Initialized population with %d seeds", len(self._population))
        return list(self._population)

    def get_cached_result(self, code_hash: str) -> EvalResult | None:
        """Check if we already evaluated this exact code."""
        return self._eval_cache.get(code_hash)

    def cache_result(self, code_hash: str, result: EvalResult) -> None:
        """Cache an evaluation result by code hash (bounded to 500 entries)."""
        if len(self._eval_cache) >= 500:
            # Remove oldest entries (first inserted)
            to_remove = list(self._eval_cache.keys())[:100]
            for k in to_remove:
                del self._eval_cache[k]
        self._eval_cache[code_hash] = result

    def update_fitness(self, individual: Individual, result: EvalResult) -> None:
        """Update an individual's fitness from eval result."""
        individual.fitness = result.fitness
        individual.eval_result = result
        self.cache_result(individual.code_hash, result)

    def advance_generation(
        self,
        new_individuals: list[Individual],
    ) -> Generation:
        """Replace population with new individuals (elitism applied).

        Returns a Generation snapshot.
        """
        self._generation += 1

        # Sort current population by fitness
        old_sorted = sorted(
            self._population,
            key=lambda x: x.fitness,
            reverse=(self._config.direction == OptimizationDirection.MAXIMIZE),
        )

        # Elites always survive
        elites = old_sorted[:self._config.elite_count]

        # Combine elites + new, deduplicate by code hash
        seen: set[str] = set()
        combined: list[Individual] = []

        for ind in elites:
            if ind.code_hash not in seen:
                seen.add(ind.code_hash)
                combined.append(ind)

        for ind in new_individuals:
            if ind.code_hash not in seen:
                seen.add(ind.code_hash)
                ind.generation = self._generation
                combined.append(ind)

        # Sort and trim to population size
        combined.sort(
            key=lambda x: x.fitness,
            reverse=(self._config.direction == OptimizationDirection.MAXIMIZE),
        )
        self._population = combined[:self._config.population_size]

        # Update hall of fame
        self._update_hall_of_fame()

        # Create generation snapshot (individuals stored only for return value,
        # _generations keeps lightweight copies without individuals to avoid OOM)
        gen = Generation(
            number=self._generation,
            individuals=list(self._population),
            best_fitness=self._population[0].fitness if self._population else self._worst_fitness(),
            avg_fitness=self._compute_avg_fitness(),
            diversity=self.compute_diversity(),
        )
        # Store lightweight copy (no individuals) for history queries
        gen_summary = Generation(
            number=gen.number,
            individuals=[],
            best_fitness=gen.best_fitness,
            avg_fitness=gen.avg_fitness,
            diversity=gen.diversity,
            timestamp=gen.timestamp,
        )
        self._generations.append(gen_summary)

        logger.info(
            "Gen %d: best=%.6f avg=%.6f diversity=%.4f pop=%d",
            self._generation, gen.best_fitness, gen.avg_fitness,
            gen.diversity, len(self._population),
        )

        return gen

    def _update_hall_of_fame(self) -> None:
        """Keep the best individuals ever seen."""
        candidates = self._hall_of_fame + self._population
        seen: set[str] = set()
        unique: list[Individual] = []
        for ind in candidates:
            if ind.code_hash not in seen:
                seen.add(ind.code_hash)
                unique.append(ind)

        unique.sort(
            key=lambda x: x.fitness,
            reverse=(self._config.direction == OptimizationDirection.MAXIMIZE),
        )
        self._hall_of_fame = unique[:self._hof_size]

    def _compute_avg_fitness(self) -> float:
        if not self._population:
            return 0.0
        valid = [ind.fitness for ind in self._population if ind.fitness != self._worst_fitness()]
        return sum(valid) / len(valid) if valid else 0.0

    def compute_diversity(self) -> float:
        """Compute population diversity as fraction of unique code hashes."""
        if len(self._population) <= 1:
            return 1.0
        hashes = {ind.code_hash for ind in self._population}
        return len(hashes) / len(self._population)

    def get_best(self) -> Individual | None:
        """Return the best individual in current population."""
        if not self._population:
            return None
        return max(
            self._population,
            key=lambda x: x.fitness if self._config.direction == OptimizationDirection.MAXIMIZE else -x.fitness,
        )

    def inject_from_hall_of_fame(self, count: int = 1) -> list[Individual]:
        """Re-inject hall of fame individuals into the population."""
        injected: list[Individual] = []
        current_hashes = {ind.code_hash for ind in self._population}
        for hof in self._hall_of_fame:
            if len(injected) >= count:
                break
            if hof.code_hash not in current_hashes:
                self._population.append(hof)
                injected.append(hof)
                current_hashes.add(hof.code_hash)
        if injected:
            logger.info("Re-injected %d individuals from hall of fame", len(injected))
        return injected

    def get_summary(self) -> dict[str, Any]:
        """Return population summary for analysis."""
        best = self.get_best()
        return {
            "generation": self._generation,
            "population_size": len(self._population),
            "best_fitness": best.fitness if best else None,
            "avg_fitness": self._compute_avg_fitness(),
            "diversity": self.compute_diversity(),
            "hall_of_fame_size": len(self._hall_of_fame),
            "cache_size": len(self._eval_cache),
        }
