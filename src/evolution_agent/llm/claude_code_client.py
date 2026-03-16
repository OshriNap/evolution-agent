"""Claude Code CLI client — uses the `claude` CLI as the LLM backend.

This avoids needing API keys since it uses your existing Claude Code auth.
Runs `claude -p` (print mode) as a subprocess for each completion.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from evolution_agent.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class ClaudeCodeClient(BaseLLMClient):
    """LLM client that shells out to the `claude` CLI.

    Uses `claude -p` (print mode) which takes a prompt on stdin and
    prints the response to stdout without interactive UI.
    """

    def __init__(
        self,
        model: str = "opus",
        claude_path: str = "claude",
        timeout_s: float = 120.0,
    ) -> None:
        super().__init__()
        self._model = model
        self._claude_path = claude_path
        self._timeout_s = timeout_s

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        t0 = time.monotonic()

        # Build prompt: system + messages concatenated
        parts: list[str] = []
        if system:
            parts.append(f"<system>\n{system}\n</system>\n")
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                parts.append(content)
            elif role == "assistant":
                parts.append(f"[Previous response]: {content}")

        prompt = "\n\n".join(parts)

        # Build command
        cmd = [
            self._claude_path,
            "-p",  # print mode (non-interactive)
            "--output-format", "text",
        ]
        if self._model:
            cmd.extend(["--model", self._model])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=prompt.encode("utf-8")),
                    timeout=self._timeout_s,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                self.stats.errors += 1
                self.stats.total_time_s += time.monotonic() - t0
                raise TimeoutError(
                    f"Claude Code timed out after {self._timeout_s}s"
                )

            elapsed = time.monotonic() - t0
            self.stats.total_calls += 1
            self.stats.total_time_s += elapsed

            if proc.returncode != 0:
                err = stderr.decode(errors="replace").strip()
                self.stats.errors += 1
                raise RuntimeError(f"Claude Code exited with code {proc.returncode}: {err[:500]}")

            text = stdout.decode(errors="replace").strip()
            logger.debug(
                "Claude Code [%s] completed in %.1fs (%d chars)",
                self._model, elapsed, len(text),
            )
            return text

        except (TimeoutError, RuntimeError):
            raise
        except Exception as e:
            self.stats.errors += 1
            self.stats.total_time_s += time.monotonic() - t0
            logger.warning("Claude Code call failed: %s", e)
            raise

    @property
    def model_name(self) -> str:
        return f"claude-code:{self._model}"

    async def is_available(self) -> bool:
        """Check if the claude CLI is installed and working."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self._claude_path, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            return proc.returncode == 0
        except Exception:
            return False
