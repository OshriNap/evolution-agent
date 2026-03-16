"""Mutation orchestrator: chooses type, retries, batch mutation."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from evolution_agent.core.selector import select_pair_for_crossover
from evolution_agent.core.types import (
    EvolutionConfig,
    Individual,
    MutationType,
)
from evolution_agent.evaluation.sandbox import CodeSandbox
from evolution_agent.llm.base import BaseLLMClient
from evolution_agent.mutation.strategies import (
    apply_crossover,
    apply_guided_mutation,
    apply_point_mutation,
    apply_structural_mutation,
)

logger = logging.getLogger(__name__)


class Mutator:
    """Orchestrates mutation: type selection, retries, and batch processing."""

    def __init__(
        self,
        config: EvolutionConfig,
        sandbox: CodeSandbox,
        function_spec: str,
        function_name: str | None = None,
    ) -> None:
        self._config = config
        self._sandbox = sandbox
        self._function_spec = function_spec
        self._function_name = function_name

    def _choose_mutation_type(self) -> MutationType:
        """Weighted random selection of mutation type."""
        types = self._config.mutation_types
        weights = self._config.mutation_weights
        # Pad weights if needed
        while len(weights) < len(types):
            weights.append(1.0 / len(types))
        return random.choices(types, weights=weights[:len(types)], k=1)[0]

    async def mutate_one(
        self,
        parent: Individual,
        llm: BaseLLMClient,
        population: list[Individual] | None = None,
        guidance: str = "",
        force_type: MutationType | None = None,
    ) -> Individual:
        """Create one mutant from a parent, with retries.

        Falls back to cloning the parent if all retries fail.
        """
        mutation_type = force_type or self._choose_mutation_type()

        for attempt in range(self._config.max_mutation_retries):
            mutant: Individual | None = None

            if mutation_type == MutationType.POINT:
                mutant = await apply_point_mutation(
                    parent, llm, self._function_spec, guidance,
                )
            elif mutation_type == MutationType.STRUCTURAL:
                mutant = await apply_structural_mutation(
                    parent, llm, self._function_spec, guidance,
                )
            elif mutation_type == MutationType.CROSSOVER and population:
                _, partner = select_pair_for_crossover(
                    population, self._config.direction, self._config.tournament_size,
                )
                mutant = await apply_crossover(
                    parent, partner, llm, self._function_spec,
                )
            elif mutation_type == MutationType.GUIDED and guidance:
                mutant = await apply_guided_mutation(
                    parent, llm, self._function_spec, guidance,
                )
            else:
                # Fallback to point mutation
                mutant = await apply_point_mutation(
                    parent, llm, self._function_spec, guidance,
                )

            if mutant is None:
                logger.debug(
                    "Mutation attempt %d/%d failed (no output)",
                    attempt + 1, self._config.max_mutation_retries,
                )
                continue

            # Validate via sandbox
            errors = self._sandbox.validate(mutant.code, self._function_name)
            if errors:
                logger.debug(
                    "Mutation attempt %d/%d failed lint: %s",
                    attempt + 1, self._config.max_mutation_retries, errors,
                )
                continue

            # Skip if identical to parent
            if mutant.code_hash == parent.code_hash:
                logger.debug("Mutation produced identical code, retrying")
                continue

            return mutant

        # All retries failed — clone parent
        logger.warning(
            "All %d mutation retries failed, cloning parent %s",
            self._config.max_mutation_retries, parent.id,
        )
        return Individual(
            code=parent.code,
            parent_ids=[parent.id],
            mutation_type=mutation_type,
            metadata={"cloned": True},
        )

    async def batch_mutate(
        self,
        parents: list[Individual],
        llm: BaseLLMClient,
        population: list[Individual] | None = None,
        guidance: str = "",
        count: int | None = None,
    ) -> list[Individual]:
        """Generate multiple mutants concurrently."""
        n = count or len(parents)
        sem = asyncio.Semaphore(self._config.max_concurrent_mutations)

        async def _mutate_with_sem(parent: Individual) -> Individual:
            async with sem:
                # Decide if this should be a crossover
                if (
                    random.random() < self._config.crossover_rate
                    and population
                    and len(population) >= 2
                ):
                    return await self.mutate_one(
                        parent, llm, population, guidance,
                        force_type=MutationType.CROSSOVER,
                    )
                return await self.mutate_one(
                    parent, llm, population, guidance,
                )

        # Cycle through parents if we need more mutants than parents
        targets: list[Individual] = []
        while len(targets) < n:
            targets.extend(parents)
        targets = targets[:n]

        tasks = [_mutate_with_sem(p) for p in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        mutants: list[Individual] = []
        for r in results:
            if isinstance(r, Individual):
                mutants.append(r)
            else:
                logger.warning("Batch mutation error: %s", r)

        logger.info("Batch mutation: %d/%d succeeded", len(mutants), n)
        return mutants
