"""Core data types for the evolution agent."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MutationType(str, Enum):
    POINT = "point"
    STRUCTURAL = "structural"
    CROSSOVER = "crossover"
    GUIDED = "guided"


class SelectionStrategy(str, Enum):
    TOURNAMENT = "tournament"
    ELITE = "elite"
    ROULETTE = "roulette"
    RANK = "rank"


class OptimizationDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class EvalResult:
    """Result of evaluating one individual."""

    fitness: float
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    eval_time_s: float = 0.0


@dataclass
class Individual:
    """One member of the population."""

    code: str
    fitness: float = float("-inf")
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    mutation_type: MutationType | None = None
    eval_result: EvalResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def id(self) -> str:
        return self.code_hash

    @property
    def code_hash(self) -> str:
        return hashlib.sha256(self.code.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "code": self.code,
            "fitness": self.fitness,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "mutation_type": self.mutation_type.value if self.mutation_type else None,
            "eval_result": {
                "fitness": self.eval_result.fitness,
                "metrics": self.eval_result.metrics,
                "error": self.eval_result.error,
                "eval_time_s": self.eval_result.eval_time_s,
            } if self.eval_result else None,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class Generation:
    """Snapshot of one generation."""

    number: int
    individuals: list[Individual]
    best_fitness: float = float("-inf")
    avg_fitness: float = 0.0
    diversity: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "diversity": self.diversity,
            "population_size": len(self.individuals),
            "timestamp": self.timestamp,
        }


@dataclass
class EvolutionConfig:
    """Configuration for an evolution run."""

    # Population
    population_size: int = 20
    elite_count: int = 3
    max_generations: int = 100

    # Selection
    selection_strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT
    tournament_size: int = 3

    # Mutation
    mutation_types: list[MutationType] = field(
        default_factory=lambda: [MutationType.POINT, MutationType.STRUCTURAL]
    )
    mutation_weights: list[float] = field(
        default_factory=lambda: [0.6, 0.4]
    )
    crossover_rate: float = 0.2
    max_mutation_retries: int = 3

    # LLM
    mutator_model: str = "qwen2.5-coder:7b"
    analyzer_model: str = "claude-opus-4-6"
    analyzer_every_n_gens: int = 5
    ollama_base_url: str = "http://localhost:11434"
    cloud_api_key: str = ""
    cloud_base_url: str = ""
    fallback_model: str = "claude-haiku-4-5-20251001"

    # Evaluation
    eval_timeout_s: float = 30.0
    direction: OptimizationDirection = OptimizationDirection.MAXIMIZE

    # Termination
    fitness_target: float | None = None
    stagnation_limit: int = 20

    # Meta-learning
    meta_optimizer_enabled: bool = True
    meta_optimizer_type: str = "heuristic"  # "heuristic" | "optuna" | "none"
    optuna_objective_mode: str = "composite"  # "improvement" | "composite" | "multi"
    optuna_storage: str = ""  # Optuna storage URL (e.g. "sqlite:///optuna.db")
    pattern_library_path: str = ""

    # Logging
    log_dir: str = "runs"
    dashboard_port: int = 8050

    # Concurrency
    max_concurrent_evals: int = 5
    max_concurrent_mutations: int = 5

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Enum):
                d[k] = v.value
            elif isinstance(v, list) and v and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            else:
                d[k] = v
        return d
