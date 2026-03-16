"""Meta-optimizer: heuristic-based hyperparameter tuning during evolution."""

from __future__ import annotations

import logging
from typing import Any

from evolution_agent.analysis.scratchpad import Scratchpad, ScratchpadEntry
from evolution_agent.core.types import EvolutionConfig, Generation

logger = logging.getLogger(__name__)


class MetaOptimizer:
    """Heuristic meta-optimizer that adjusts evolution parameters based on trends.

    Monitors population dynamics and adjusts:
    - Mutation weights (point vs structural)
    - Crossover rate
    - Tournament size
    - Elite count
    """

    def __init__(self, config: EvolutionConfig) -> None:
        self._config = config
        self._stagnation_count = 0
        self._last_best_fitness: float | None = None
        self._diversity_trend: list[float] = []

    def step(
        self,
        gen: Generation,
        scratchpad: Scratchpad,
        analysis_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one meta-optimization step. Returns dict of adjustments made."""
        adjustments: dict[str, Any] = {}

        # Track stagnation
        if self._last_best_fitness is not None:
            if abs(gen.best_fitness - self._last_best_fitness) < 1e-10:
                self._stagnation_count += 1
            else:
                self._stagnation_count = 0
        self._last_best_fitness = gen.best_fitness

        # Track diversity trend
        self._diversity_trend.append(gen.diversity)
        if len(self._diversity_trend) > 20:
            self._diversity_trend = self._diversity_trend[-20:]

        # Apply analyzer recommendations if available
        if analysis_result:
            rec_weights = analysis_result.get("recommended_mutation_weights", {})
            if rec_weights:
                point_w = rec_weights.get("point", 0.6)
                struct_w = rec_weights.get("structural", 0.4)
                total = point_w + struct_w
                if total > 0:
                    self._config.mutation_weights = [point_w / total, struct_w / total]
                    adjustments["mutation_weights"] = self._config.mutation_weights

        # Heuristic: stagnation → more structural mutations
        if self._stagnation_count >= 5:
            old_weights = list(self._config.mutation_weights)
            # Shift toward structural
            self._config.mutation_weights = [0.3, 0.7]
            adjustments["mutation_weights_stagnation"] = self._config.mutation_weights
            scratchpad.add(ScratchpadEntry(
                generation=gen.number,
                category="observation",
                content=(
                    f"Stagnation detected ({self._stagnation_count} gens). "
                    f"Shifted mutation weights from {old_weights} to {self._config.mutation_weights}"
                ),
                source="meta_optimizer",
            ))

        # Heuristic: low diversity → increase crossover and tournament size
        if gen.diversity < 0.3:
            old_crossover = self._config.crossover_rate
            self._config.crossover_rate = min(0.5, self._config.crossover_rate + 0.1)
            if self._config.crossover_rate != old_crossover:
                adjustments["crossover_rate"] = self._config.crossover_rate
                scratchpad.add(ScratchpadEntry(
                    generation=gen.number,
                    category="observation",
                    content=(
                        f"Low diversity ({gen.diversity:.2f}). "
                        f"Increased crossover rate to {self._config.crossover_rate:.2f}"
                    ),
                    source="meta_optimizer",
                ))

        # Heuristic: diversity dropping fast → increase elitism
        if len(self._diversity_trend) >= 5:
            recent_div = self._diversity_trend[-5:]
            div_slope = (recent_div[-1] - recent_div[0]) / len(recent_div)
            if div_slope < -0.05:
                old_elite = self._config.elite_count
                self._config.elite_count = min(
                    self._config.population_size // 2,
                    self._config.elite_count + 1,
                )
                if self._config.elite_count != old_elite:
                    adjustments["elite_count"] = self._config.elite_count

        # Heuristic: extreme stagnation → reset mutation weights
        if self._stagnation_count >= 15:
            self._config.mutation_weights = [0.5, 0.5]
            self._config.crossover_rate = 0.3
            self._stagnation_count = 0
            adjustments["reset"] = True
            scratchpad.add(ScratchpadEntry(
                generation=gen.number,
                category="conclusion",
                content=(
                    "Extreme stagnation (15+ gens). Reset mutation weights "
                    "and crossover rate to defaults."
                ),
                source="meta_optimizer",
            ))

        if adjustments:
            logger.info("Meta-optimizer adjustments at gen %d: %s", gen.number, adjustments)

        return adjustments
