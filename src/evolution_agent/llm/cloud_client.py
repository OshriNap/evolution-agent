"""Cloud LLM client via OpenAI SDK (works with Claude, OpenAI, LiteLLM)."""

from __future__ import annotations

import logging
import time
from typing import Any

from evolution_agent.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class CloudClient(BaseLLMClient):
    """LLM client using the OpenAI SDK for cloud models.

    Lazily initializes the OpenAI client to avoid errors when no API key is set.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        api_key: str = "",
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client = None  # lazy init

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        t0 = time.monotonic()
        client = self._get_client()

        full_messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            *messages,
        ]

        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            elapsed = time.monotonic() - t0
            self.stats.total_calls += 1
            self.stats.total_time_s += elapsed

            usage = response.usage
            if usage:
                self.stats.total_input_tokens += usage.prompt_tokens
                self.stats.total_output_tokens += usage.completion_tokens

            text = response.choices[0].message.content or ""
            logger.debug(
                "Cloud [%s] completed in %.1fs", self._model, elapsed,
            )
            return text

        except Exception as e:
            self.stats.errors += 1
            self.stats.total_time_s += time.monotonic() - t0
            logger.warning("Cloud LLM call failed: %s", e)
            raise

    @property
    def model_name(self) -> str:
        return f"cloud:{self._model}"

    async def is_available(self) -> bool:
        """Cloud is assumed available if we have an API key."""
        import os
        return bool(self._api_key or os.environ.get("OPENAI_API_KEY"))
