"""Local LLM client via Ollama HTTP API."""

from __future__ import annotations

import logging
import time
from typing import Any

import aiohttp

from evolution_agent.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    """LLM client using Ollama's /api/chat endpoint."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        super().__init__()
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        session = await self._get_session()
        t0 = time.monotonic()

        ollama_messages = [{"role": "system", "content": system}]
        ollama_messages.extend(messages)

        payload = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": False,
            "keep_alive": "30m",  # keep model loaded between calls
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            async with session.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=180),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            elapsed = time.monotonic() - t0
            self.stats.total_calls += 1
            self.stats.total_time_s += elapsed

            # Ollama returns token counts in eval_count / prompt_eval_count
            self.stats.total_output_tokens += data.get("eval_count", 0)
            self.stats.total_input_tokens += data.get("prompt_eval_count", 0)

            text = data.get("message", {}).get("content", "")
            logger.debug(
                "Ollama [%s] completed in %.1fs (%d tokens)",
                self._model, elapsed, data.get("eval_count", 0),
            )
            return text

        except Exception as e:
            self.stats.errors += 1
            self.stats.total_time_s += time.monotonic() - t0
            logger.warning("Ollama call failed: %s", e)
            raise

    @property
    def model_name(self) -> str:
        return f"ollama:{self._model}"

    async def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self._base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return False
                data = await resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                # Check if our model is available (exact or prefix match)
                for m in models:
                    if m == self._model or m.startswith(self._model.split(":")[0]):
                        return True
                logger.warning(
                    "Ollama running but model '%s' not found. Available: %s",
                    self._model, models,
                )
                return False
        except Exception:
            return False

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
