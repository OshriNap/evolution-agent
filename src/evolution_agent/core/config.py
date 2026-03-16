"""Configuration loading from YAML + environment variables."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from evolution_agent.core.types import (
    EvolutionConfig,
    MutationType,
    OptimizationDirection,
    SelectionStrategy,
)

logger = logging.getLogger(__name__)


def load_config(
    yaml_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> EvolutionConfig:
    """Load config from YAML file, env vars, and explicit overrides.

    Priority: overrides > env vars > YAML > defaults.
    """
    data: dict[str, Any] = {}

    # 1. YAML file
    if yaml_path:
        path = Path(yaml_path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            logger.info("Loaded config from %s", path)

    # 2. Environment variables (prefixed with EVOL_)
    env_map = {
        "EVOL_POPULATION_SIZE": ("population_size", int),
        "EVOL_ELITE_COUNT": ("elite_count", int),
        "EVOL_MAX_GENERATIONS": ("max_generations", int),
        "EVOL_TOURNAMENT_SIZE": ("tournament_size", int),
        "EVOL_CROSSOVER_RATE": ("crossover_rate", float),
        "EVOL_MUTATOR_MODEL": ("mutator_model", str),
        "EVOL_ANALYZER_MODEL": ("analyzer_model", str),
        "EVOL_ANALYZER_EVERY_N": ("analyzer_every_n_gens", int),
        "EVOL_OLLAMA_URL": ("ollama_base_url", str),
        "EVOL_CLOUD_API_KEY": ("cloud_api_key", str),
        "EVOL_CLOUD_BASE_URL": ("cloud_base_url", str),
        "EVOL_FALLBACK_MODEL": ("fallback_model", str),
        "EVOL_EVAL_TIMEOUT": ("eval_timeout_s", float),
        "EVOL_DIRECTION": ("direction", str),
        "EVOL_FITNESS_TARGET": ("fitness_target", float),
        "EVOL_STAGNATION_LIMIT": ("stagnation_limit", int),
        "EVOL_LOG_DIR": ("log_dir", str),
        "EVOL_DASHBOARD_PORT": ("dashboard_port", int),
        "EVOL_MAX_CONCURRENT_EVALS": ("max_concurrent_evals", int),
        "EVOL_MAX_CONCURRENT_MUTATIONS": ("max_concurrent_mutations", int),
        "EVOL_PATTERN_LIBRARY_PATH": ("pattern_library_path", str),
        "EVOL_META_OPTIMIZER_TYPE": ("meta_optimizer_type", str),
        "EVOL_OPTUNA_OBJECTIVE_MODE": ("optuna_objective_mode", str),
        "EVOL_OPTUNA_STORAGE": ("optuna_storage", str),
    }

    for env_key, (config_key, converter) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                data[config_key] = converter(val)
            except (ValueError, TypeError):
                logger.warning("Invalid env var %s=%s", env_key, val)

    # 3. Explicit overrides
    if overrides:
        data.update(overrides)

    # Convert enum strings
    if "direction" in data and isinstance(data["direction"], str):
        data["direction"] = OptimizationDirection(data["direction"])
    if "selection_strategy" in data and isinstance(data["selection_strategy"], str):
        data["selection_strategy"] = SelectionStrategy(data["selection_strategy"])
    if "mutation_types" in data and isinstance(data["mutation_types"], list):
        data["mutation_types"] = [
            MutationType(m) if isinstance(m, str) else m
            for m in data["mutation_types"]
        ]

    return EvolutionConfig(**{
        k: v for k, v in data.items()
        if k in EvolutionConfig.__dataclass_fields__
    })


class RuntimeConfig:
    """Hot-reloadable runtime configuration via JSON file.

    Write to the JSON file during a run to adjust parameters on the fly.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: dict[str, Any] = {}
        self._mtime: float = 0.0

    def get(self, key: str, default: Any = None) -> Any:
        self._maybe_reload()
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._maybe_reload()
        self._data[key] = value
        self._save()

    def _maybe_reload(self) -> None:
        if not self._path.exists():
            return
        mtime = self._path.stat().st_mtime
        if mtime > self._mtime:
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
                self._mtime = mtime
            except Exception as e:
                logger.warning("Failed to reload runtime config: %s", e)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2), encoding="utf-8"
        )
        self._mtime = self._path.stat().st_mtime
