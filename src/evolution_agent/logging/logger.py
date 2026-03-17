"""JSONL structured logger for evolution runs."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from evolution_agent.core.types import Generation, Individual

logger = logging.getLogger(__name__)


class EvolutionLogger:
    """Writes structured JSONL log files for evolution runs."""

    def __init__(self, run_dir: str | Path) -> None:
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._run_dir / "evolution.jsonl"
        self._gen_count = 0

    def _write(self, record: dict[str, Any]) -> None:
        record["timestamp"] = time.time()
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning("Failed to write log: %s", e)

    def log_config(self, config: dict[str, Any]) -> None:
        self._write({"type": "config", "data": config})

    def log_generation(self, gen: Generation) -> None:
        self._gen_count += 1
        self._write({
            "type": "generation",
            "data": gen.to_dict(),
        })
        # Log best individual's code each generation (population is pre-sorted)
        if gen.individuals:
            best = gen.individuals[0]
            self._write({
                "type": "best_code",
                "data": {
                    "generation": gen.number,
                    "id": best.id,
                    "fitness": best.fitness,
                    "code": best.code,
                },
            })
        # Also write full population to separate file every 10 gens
        if self._gen_count % 10 == 0:
            pop_path = self._run_dir / f"population_gen{gen.number}.json"
            try:
                pop_data = [ind.to_dict() for ind in gen.individuals]
                pop_path.write_text(
                    json.dumps(pop_data, indent=2, default=str),
                    encoding="utf-8",
                )
            except Exception as e:
                logger.warning("Failed to write population snapshot: %s", e)

    def log_evaluation(self, individual: Individual) -> None:
        self._write({
            "type": "evaluation",
            "data": {
                "id": individual.id,
                "fitness": individual.fitness,
                "generation": individual.generation,
                "mutation_type": individual.mutation_type.value if individual.mutation_type else None,
                "error": individual.eval_result.error if individual.eval_result else None,
                "eval_time_s": individual.eval_result.eval_time_s if individual.eval_result else None,
            },
        })

    def log_analysis(self, generation: int, data: dict[str, Any]) -> None:
        self._write({"type": "analysis", "generation": generation, "data": data})

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        self._write({"type": event_type, "data": data})

    def log_summary(self, summary: dict[str, Any]) -> None:
        self._write({"type": "summary", "data": summary})
        # Also write summary as standalone JSON
        summary_path = self._run_dir / "summary.json"
        try:
            summary_path.write_text(
                json.dumps(summary, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Failed to write summary: %s", e)

    def read_log(self) -> list[dict[str, Any]]:
        """Read all log entries."""
        entries: list[dict[str, Any]] = []
        if not self._log_path.exists():
            return entries
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries
