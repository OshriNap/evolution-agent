"""Main async evolution loop orchestrator."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from evolution_agent.analysis.analyzer import EvolutionAnalyzer
from evolution_agent.analysis.patterns import MetaPatternLibrary
from evolution_agent.analysis.scratchpad import Scratchpad
from evolution_agent.core.config import RuntimeConfig
from evolution_agent.core.population import PopulationManager
from evolution_agent.core.selector import select_parents
from evolution_agent.core.types import EvolutionConfig, Generation, Individual
from evolution_agent.evaluation.base import BaseEvaluator
from evolution_agent.evaluation.sandbox import CodeSandbox
from evolution_agent.llm.router import LLMRouter
from evolution_agent.logging.logger import EvolutionLogger
from evolution_agent.mutation.mutator import Mutator

logger = logging.getLogger(__name__)


class EvolutionEngine:
    """Main async evolution loop.

    Flow per generation:
    1. Select parents
    2. Analyze (every Nth gen) → update scratchpad + get guidance
    3. Mutate/crossover → generate offspring
    4. Evaluate all new individuals
    5. Replace population (elitism)
    6. Meta-learn (adjust hyperparameters)
    7. Log
    8. Check termination
    """

    def __init__(
        self,
        config: EvolutionConfig,
        evaluator: BaseEvaluator,
        seeds: list[str],
        run_dir: str | None = None,
    ) -> None:
        self._config = config
        self._evaluator = evaluator
        self._seeds = seeds

        # Set up run directory
        if run_dir is None:
            run_dir = str(
                Path(config.log_dir)
                / f"run_{int(time.time())}"
            )
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Core components
        self._population_mgr = PopulationManager(config)
        self._sandbox = CodeSandbox()
        self._router = LLMRouter(config)
        self._scratchpad = Scratchpad()
        self._logger = EvolutionLogger(self._run_dir)

        # Pattern library
        lib_path = config.pattern_library_path or str(
            self._run_dir / "meta_patterns.json"
        )
        self._pattern_library = MetaPatternLibrary(lib_path)
        self._pattern_library.load()

        # Meta components (initialized lazily after router is available)
        self._analyzer: EvolutionAnalyzer | None = None
        self._meta_optimizer = self._create_meta_optimizer(config)

        # Mutation orchestrator
        self._mutator = Mutator(
            config=config,
            sandbox=self._sandbox,
            function_spec=evaluator.get_function_spec(),
        )

        # Runtime config for hot-reload
        self._runtime_config = RuntimeConfig(self._run_dir / "runtime.json")

        # Dashboard refresh
        self._dashboard_updater = self._create_dashboard_updater()

        # State
        self._start_time: float = 0.0
        self._last_analysis_result: dict[str, Any] | None = None

    def _create_dashboard_updater(self):
        """Create a callable that refreshes dashboard data.json if the dashboard dir exists."""
        from evolution_agent.logging.dashboard import _load_run_data, _json_safe
        dashboard_dir = self._run_dir.parent.parent / ".dashboard"
        api_dir = dashboard_dir / "_data"
        if not dashboard_dir.exists():
            return None
        api_dir.mkdir(exist_ok=True)

        import json as _json

        def _update():
            try:
                data = _load_run_data(str(self._run_dir))
                tmp = api_dir / "data.json.tmp"
                tmp.write_text(_json.dumps(_json_safe(data), default=str), encoding="utf-8")
                tmp.rename(api_dir / "data.json")
            except Exception as e:
                logger.debug("Dashboard update failed: %s", e)

        return _update

    @staticmethod
    def _create_meta_optimizer(config: EvolutionConfig) -> Any:
        """Factory for meta-optimizer based on config."""
        if not config.meta_optimizer_enabled or config.meta_optimizer_type == "none":
            return None

        if config.meta_optimizer_type == "optuna":
            try:
                from evolution_agent.meta.optuna_optimizer import OptunaMetaOptimizer
                return OptunaMetaOptimizer(
                    config,
                    objective_mode=config.optuna_objective_mode,
                    storage=config.optuna_storage or None,
                )
            except ImportError:
                logger.warning(
                    "Optuna not installed, falling back to heuristic optimizer. "
                    "Install with: pip install evolution-agent[optuna]"
                )

        from evolution_agent.meta.optimizer import MetaOptimizer
        return MetaOptimizer(config)

    async def run(self) -> dict[str, Any]:
        """Run the full evolution loop. Returns run summary."""
        self._start_time = time.monotonic()
        logger.info("Starting evolution run in %s", self._run_dir)

        # Initialize analyzer (only if cloud LLM is available)
        analyzer_llm = await self._router.get_analyzer()
        if await analyzer_llm.is_available():
            self._analyzer = EvolutionAnalyzer(analyzer_llm, self._pattern_library)
        else:
            logger.info("Cloud LLM not configured — running without analyzer")

        # Log config
        self._logger.log_config(self._config.to_dict())

        # Gen 0: seed → evaluate
        population = self._population_mgr.init_from_seeds(self._seeds)
        await self._evaluate_batch(population)

        # Fill population to target size via mutation
        if len(population) < self._config.population_size:
            mutator_llm = await self._router.get_mutator()
            needed = self._config.population_size - len(population)
            parents = select_parents(
                population, needed,
                strategy=self._config.selection_strategy,
                direction=self._config.direction,
                tournament_size=self._config.tournament_size,
            )
            offspring = await self._mutator.batch_mutate(
                parents, mutator_llm, population, count=needed,
            )
            await self._evaluate_batch(offspring)
            population.extend(offspring)

        # Initial generation snapshot
        gen = self._population_mgr.advance_generation(
            [ind for ind in population if ind not in self._population_mgr.population]
        )
        self._logger.log_generation(gen)
        if self._dashboard_updater:
            self._dashboard_updater()

        # Main evolution loop
        for gen_num in range(1, self._config.max_generations + 1):
            # Check runtime config for pause/stop
            if self._runtime_config.get("stop", False):
                logger.info("Stop requested via runtime config")
                break

            current_pop = self._population_mgr.population

            # 1. Select parents
            n_offspring = self._config.population_size - self._config.elite_count
            parents = select_parents(
                current_pop, n_offspring,
                strategy=self._config.selection_strategy,
                direction=self._config.direction,
                tournament_size=self._config.tournament_size,
            )

            # 2. Analyze (every Nth gen) + mid-run library updates
            guidance = ""
            if gen_num % self._config.analyzer_every_n_gens == 0 and self._analyzer:
                analyzer_llm = await self._router.get_analyzer()
                if await analyzer_llm.is_available():
                    try:
                        analysis = await self._analyzer.analyze(
                            generation=gen_num,
                            population=current_pop,
                            generation_history=[
                                g.to_dict() for g in self._population_mgr.generations
                            ],
                            scratchpad=self._scratchpad,
                            function_spec=self._evaluator.get_function_spec(),
                            direction=self._config.direction.value,
                            max_generations_remaining=self._config.max_generations - gen_num,
                        )
                        guidance = analysis.mutation_guidance
                        self._last_analysis_result = {
                            "recommended_mutation_weights": analysis.recommended_mutation_weights,
                            "phase": analysis.phase,
                            "detected_patterns": analysis.detected_patterns,
                        }
                        self._logger.log_analysis(gen_num, analysis.__dict__)

                        # Compress scratchpad
                        await self._scratchpad.update_digest(analyzer_llm, gen_num)

                        # Mid-run library update (every 2nd analysis cycle)
                        if gen_num % (self._config.analyzer_every_n_gens * 2) == 0:
                            try:
                                lib_changes = await self._analyzer.mid_run_library_update(
                                    self._scratchpad, gen_num,
                                )
                                if lib_changes:
                                    self._logger.log_event("library_update", {
                                        "generation": gen_num,
                                        "changes": len(lib_changes),
                                    })
                            except Exception as e:
                                logger.debug("Mid-run library update failed: %s", e)

                    except Exception as e:
                        logger.warning("Analysis at gen %d failed: %s", gen_num, e)
                else:
                    if gen_num == self._config.analyzer_every_n_gens:
                        logger.info("Cloud LLM not available — skipping analysis")

            # Add failed-approaches context to guidance
            failed = self._scratchpad.format_failed_approaches()
            if failed:
                guidance = (guidance + "\n\n## Previously failed approaches (avoid repeating):\n" + failed) if guidance else ""

            # 3. Mutate
            mutator_llm = await self._router.get_mutator()
            offspring = await self._mutator.batch_mutate(
                parents, mutator_llm, current_pop, guidance, count=n_offspring,
            )

            # 4. Evaluate
            await self._evaluate_batch(offspring)

            # 5. Replace (elitism built into advance_generation)
            gen = self._population_mgr.advance_generation(offspring)
            self._logger.log_generation(gen)
            if self._dashboard_updater:
                self._dashboard_updater()

            # 6. Meta-learn
            if self._meta_optimizer:
                adjustments = self._meta_optimizer.step(
                    gen, self._scratchpad, self._last_analysis_result,
                )
                if adjustments:
                    self._logger.log_event("meta_optimizer", adjustments)

                # Hall of fame re-injection on stagnation
                if gen.diversity < 0.3:
                    self._population_mgr.inject_from_hall_of_fame(2)

            # 7. Check termination
            if self._should_terminate(gen):
                logger.info("Termination condition met at gen %d", gen_num)
                break

        # End-of-run
        return await self._finalize()

    async def _evaluate_batch(self, individuals: list[Individual]) -> None:
        """Evaluate a batch of individuals concurrently with caching."""
        sem = asyncio.Semaphore(self._config.max_concurrent_evals)

        async def _eval_one(ind: Individual) -> None:
            # Check cache
            cached = self._population_mgr.get_cached_result(ind.code_hash)
            if cached:
                self._population_mgr.update_fitness(ind, cached)
                return

            async with sem:
                result = await self._evaluator.evaluate(ind.code)
                self._population_mgr.update_fitness(ind, result)
                self._logger.log_evaluation(ind)

        await asyncio.gather(*[_eval_one(ind) for ind in individuals])

    def _should_terminate(self, gen: Generation) -> bool:
        """Check if evolution should stop."""
        # Fitness target reached
        if self._config.fitness_target is not None:
            best = self._population_mgr.get_best()
            if best and self._evaluator.is_better(
                best.fitness, self._config.fitness_target
            ):
                return True

        # Stagnation
        gens = self._population_mgr.generations
        if len(gens) >= self._config.stagnation_limit:
            recent = gens[-self._config.stagnation_limit:]
            best_fitnesses = [g.best_fitness for g in recent]
            if max(best_fitnesses) - min(best_fitnesses) < 1e-10:
                return True

        return False

    async def _finalize(self) -> dict[str, Any]:
        """End-of-run: update pattern library, write summary."""
        best = self._population_mgr.get_best()
        summary = {
            "total_generations": self._population_mgr.generation,
            "best_fitness": best.fitness if best else None,
            "best_code": best.code if best else None,
            "elapsed_s": time.monotonic() - self._start_time,
            "population_summary": self._population_mgr.get_summary(),
            "llm_stats": self._router.get_stats(),
        }

        # Update pattern library
        if self._analyzer:
            try:
                updates = await self._analyzer.propose_library_updates(
                    self._scratchpad, summary,
                )
                summary["library_updates"] = len(updates)
            except Exception as e:
                logger.warning("End-of-run library update failed: %s", e)

        # Include Optuna stats if available
        if self._meta_optimizer and hasattr(self._meta_optimizer, "get_best_params"):
            summary["optuna"] = self._meta_optimizer.get_best_params()

        # Log summary
        self._logger.log_summary(summary)

        # Save hall of fame
        hof = self._population_mgr.hall_of_fame
        self._logger.log_event("hall_of_fame", {
            "individuals": [ind.to_dict() for ind in hof],
        })

        # Final dashboard update
        if self._dashboard_updater:
            self._dashboard_updater()

        # Cleanup
        await self._router.close()

        logger.info(
            "Evolution complete: %d generations, best fitness=%.6f, elapsed=%.1fs",
            summary["total_generations"],
            summary["best_fitness"] or 0,
            summary["elapsed_s"],
        )

        return summary
