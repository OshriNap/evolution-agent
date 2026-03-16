"""Tests for JSONL logger."""

import json
import os
import tempfile
from pathlib import Path

from evolution_agent.core.types import EvalResult, Generation, Individual
from evolution_agent.logging.logger import EvolutionLogger


def test_log_config():
    with tempfile.TemporaryDirectory() as d:
        logger = EvolutionLogger(d)
        logger.log_config({"population_size": 20})

        entries = logger.read_log()
        assert len(entries) == 1
        assert entries[0]["type"] == "config"
        assert entries[0]["data"]["population_size"] == 20


def test_log_generation():
    with tempfile.TemporaryDirectory() as d:
        logger = EvolutionLogger(d)
        gen = Generation(
            number=1,
            individuals=[Individual(code="def f(): return 1", fitness=0.5)],
            best_fitness=0.5,
            avg_fitness=0.5,
            diversity=1.0,
        )
        logger.log_generation(gen)

        entries = logger.read_log()
        assert entries[0]["type"] == "generation"
        assert entries[0]["data"]["best_fitness"] == 0.5


def test_log_evaluation():
    with tempfile.TemporaryDirectory() as d:
        logger = EvolutionLogger(d)
        ind = Individual(code="def f(): return 1", fitness=0.8)
        ind.eval_result = EvalResult(fitness=0.8, eval_time_s=1.5)
        logger.log_evaluation(ind)

        entries = logger.read_log()
        assert entries[0]["type"] == "evaluation"
        assert entries[0]["data"]["fitness"] == 0.8


def test_log_summary_writes_json():
    with tempfile.TemporaryDirectory() as d:
        logger = EvolutionLogger(d)
        summary = {"best_fitness": 0.95, "total_generations": 50}
        logger.log_summary(summary)

        # Should write both JSONL entry and standalone summary.json
        summary_path = Path(d) / "summary.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert data["best_fitness"] == 0.95
