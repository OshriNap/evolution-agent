"""Tests for config loading."""

import os
import tempfile

import yaml

from evolution_agent.core.config import RuntimeConfig, load_config
from evolution_agent.core.types import OptimizationDirection


def test_load_config_defaults():
    cfg = load_config()
    assert cfg.population_size == 20
    assert cfg.direction == OptimizationDirection.MAXIMIZE


def test_load_config_overrides():
    cfg = load_config(overrides={"population_size": 10, "direction": "minimize"})
    assert cfg.population_size == 10
    assert cfg.direction == OptimizationDirection.MINIMIZE


def test_load_config_yaml():
    data = {"population_size": 50, "elite_count": 5}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()
        cfg = load_config(f.name)
    assert cfg.population_size == 50
    assert cfg.elite_count == 5
    os.unlink(f.name)


def test_load_config_env_vars():
    os.environ["EVOL_POPULATION_SIZE"] = "42"
    try:
        cfg = load_config()
        assert cfg.population_size == 42
    finally:
        del os.environ["EVOL_POPULATION_SIZE"]


def test_load_config_priority():
    """Overrides > env > yaml."""
    os.environ["EVOL_POPULATION_SIZE"] = "42"
    try:
        cfg = load_config(overrides={"population_size": 99})
        assert cfg.population_size == 99
    finally:
        del os.environ["EVOL_POPULATION_SIZE"]


def test_runtime_config():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        rc = RuntimeConfig(f.name)
        rc.set("foo", "bar")
        assert rc.get("foo") == "bar"
        assert rc.get("missing", 123) == 123

        # Reload from file
        rc2 = RuntimeConfig(f.name)
        assert rc2.get("foo") == "bar"
        os.unlink(f.name)
