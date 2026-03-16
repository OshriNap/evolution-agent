"""Optuna-based meta-optimizer using ask/tell API for online Bayesian tuning.

Replaces or wraps the heuristic MetaOptimizer with Optuna's TPE sampler
for data-driven hyperparameter adaptation during evolution.

Requires: pip install evolution-agent[optuna]
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from evolution_agent.analysis.scratchpad import Scratchpad, ScratchpadEntry
from evolution_agent.core.types import EvolutionConfig, Generation

logger = logging.getLogger(__name__)


class OptunaMetaOptimizer:
    """Online Bayesian meta-optimizer using Optuna's ask/tell API.

    Each generation is one Optuna trial. The optimizer suggests
    hyperparameters (mutation weights, crossover rate, etc.) and
    receives feedback based on fitness improvement and diversity.

    Supports three objective modes:
    - "improvement": maximize fitness improvement per generation
    - "composite": weighted combo of improvement + diversity
    - "multi": multi-objective (improvement, diversity) via NSGA-II
    """

    def __init__(
        self,
        config: EvolutionConfig,
        objective_mode: str = "composite",
        improvement_weight: float = 0.7,
        diversity_weight: float = 0.3,
        n_startup_trials: int = 5,
        study_name: str | None = None,
        storage: str | None = None,
    ) -> None:
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for OptunaMetaOptimizer. "
                "Install with: pip install evolution-agent[optuna]"
            )

        self._config = config
        self._objective_mode = objective_mode
        self._improvement_weight = improvement_weight
        self._diversity_weight = diversity_weight

        self._last_best_fitness: float | None = None
        self._pending_trial: optuna.trial.Trial | None = None

        # Suppress Optuna's default logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create study
        if objective_mode == "multi":
            self._study = optuna.create_study(
                study_name=study_name or "evolution_meta",
                storage=storage,
                directions=["maximize", "maximize"],
                sampler=optuna.samplers.NSGAIISampler(seed=42),
                load_if_exists=True,
            )
        else:
            self._study = optuna.create_study(
                study_name=study_name or "evolution_meta",
                storage=storage,
                direction="maximize",
                sampler=TPESampler(
                    n_startup_trials=n_startup_trials,
                    seed=42,
                ),
                load_if_exists=True,
            )

    @property
    def study(self) -> optuna.study.Study:
        """Access the underlying Optuna study for inspection."""
        return self._study

    def step(
        self,
        gen: Generation,
        scratchpad: Scratchpad,
        analysis_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one Optuna ask/tell cycle.

        1. Tell Optuna the result of the previous trial (if any)
        2. Ask Optuna for new hyperparameters
        3. Apply them to the config
        """
        adjustments: dict[str, Any] = {}

        # --- Tell: report result of previous trial ---
        if self._pending_trial is not None and self._last_best_fitness is not None:
            improvement = gen.best_fitness - self._last_best_fitness

            if self._objective_mode == "multi":
                self._study.tell(
                    self._pending_trial,
                    values=[improvement, gen.diversity],
                )
            elif self._objective_mode == "composite":
                score = (
                    self._improvement_weight * improvement
                    + self._diversity_weight * gen.diversity
                )
                self._study.tell(self._pending_trial, score)
            else:  # "improvement"
                self._study.tell(self._pending_trial, improvement)

        self._last_best_fitness = gen.best_fitness

        # --- Ask: get new hyperparameters ---
        trial = self._study.ask()
        self._pending_trial = trial

        # Suggest hyperparameters
        point_weight = trial.suggest_float("point_mutation_weight", 0.1, 0.9)
        struct_weight = 1.0 - point_weight
        crossover_rate = trial.suggest_float("crossover_rate", 0.0, 0.5)
        tournament_size = trial.suggest_int("tournament_size", 2, min(7, self._config.population_size))
        elite_ratio = trial.suggest_float("elite_ratio", 0.05, 0.3)
        elite_count = max(1, int(elite_ratio * self._config.population_size))

        # Apply to config
        self._config.mutation_weights = [point_weight, struct_weight]
        self._config.crossover_rate = crossover_rate
        self._config.tournament_size = tournament_size
        self._config.elite_count = elite_count

        adjustments = {
            "source": "optuna",
            "trial_number": trial.number,
            "mutation_weights": [point_weight, struct_weight],
            "crossover_rate": crossover_rate,
            "tournament_size": tournament_size,
            "elite_count": elite_count,
        }

        # Incorporate analyzer recommendations as Optuna user attributes
        if analysis_result:
            trial.set_user_attr("analyzer_phase", analysis_result.get("phase", ""))
            detected = analysis_result.get("detected_patterns", [])
            if detected:
                trial.set_user_attr("detected_patterns", detected)

        # Log to scratchpad
        scratchpad.add(ScratchpadEntry(
            generation=gen.number,
            category="observation",
            content=(
                f"Optuna trial #{trial.number}: "
                f"point={point_weight:.2f} struct={struct_weight:.2f} "
                f"crossover={crossover_rate:.2f} tournament={tournament_size} "
                f"elite={elite_count}"
            ),
            source="meta_optimizer",
        ))

        logger.info(
            "Optuna trial #%d at gen %d: weights=[%.2f, %.2f] "
            "crossover=%.2f tournament=%d elite=%d",
            trial.number, gen.number, point_weight, struct_weight,
            crossover_rate, tournament_size, elite_count,
        )

        return adjustments

    def get_best_params(self) -> dict[str, Any]:
        """Return the best hyperparameters found so far."""
        if self._objective_mode == "multi":
            # Return Pareto front
            return {
                "pareto_front": [
                    {"params": t.params, "values": t.values}
                    for t in self._study.best_trials
                ],
            }
        try:
            return {
                "best_params": self._study.best_params,
                "best_value": self._study.best_value,
                "n_trials": len(self._study.trials),
            }
        except ValueError:
            return {"n_trials": 0}

    def get_importance(self) -> dict[str, float]:
        """Return hyperparameter importance (requires enough trials)."""
        try:
            return optuna.importance.get_param_importances(self._study)
        except Exception:
            return {}
