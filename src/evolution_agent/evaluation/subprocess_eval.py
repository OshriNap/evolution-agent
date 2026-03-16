"""Subprocess evaluator: run external scripts for fitness evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path

from evolution_agent.core.types import EvalResult, OptimizationDirection
from evolution_agent.evaluation.base import BaseEvaluator

logger = logging.getLogger(__name__)


class SubprocessEvaluator(BaseEvaluator):
    """Evaluate code by writing it to a temp file and running an external script.

    The script receives the code file path as an argument and must
    print a JSON object to stdout with at least a "fitness" key.

    Expected JSON output format:
    {"fitness": 0.95, "metrics": {"accuracy": 0.95, "speed": 1.2}}
    """

    def __init__(
        self,
        script_path: str,
        function_spec: str,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        timeout_s: float = 60.0,
        extra_args: list[str] | None = None,
    ) -> None:
        self._script_path = script_path
        self._function_spec = function_spec
        self._direction = direction
        self._timeout_s = timeout_s
        self._extra_args = extra_args or []

    async def evaluate(self, code: str) -> EvalResult:
        t0 = time.monotonic()

        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
        ) as f:
            f.write(code)
            code_path = f.name

        try:
            cmd = ["python", self._script_path, code_path, *self._extra_args]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._timeout_s,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return EvalResult(
                    fitness=self.worst_fitness(),
                    error=f"Script timed out after {self._timeout_s}s",
                    eval_time_s=time.monotonic() - t0,
                )

            if proc.returncode != 0:
                err = stderr.decode(errors="replace").strip()
                return EvalResult(
                    fitness=self.worst_fitness(),
                    error=f"Script exited with code {proc.returncode}: {err[:500]}",
                    eval_time_s=time.monotonic() - t0,
                )

            # Parse JSON from stdout
            output = stdout.decode(errors="replace").strip()
            try:
                data = json.loads(output)
            except json.JSONDecodeError:
                return EvalResult(
                    fitness=self.worst_fitness(),
                    error=f"Script output is not valid JSON: {output[:200]}",
                    eval_time_s=time.monotonic() - t0,
                )

            return EvalResult(
                fitness=float(data.get("fitness", self.worst_fitness())),
                metrics=data.get("metrics", {}),
                eval_time_s=time.monotonic() - t0,
            )

        except Exception as e:
            return EvalResult(
                fitness=self.worst_fitness(),
                error=str(e),
                eval_time_s=time.monotonic() - t0,
            )
        finally:
            Path(code_path).unlink(missing_ok=True)

    def get_function_spec(self) -> str:
        return self._function_spec

    def get_direction(self) -> OptimizationDirection:
        return self._direction
