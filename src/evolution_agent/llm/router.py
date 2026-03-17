"""LLM Router: two-tier routing with fallback logic."""

from __future__ import annotations

import logging
from typing import Any

from evolution_agent.core.types import EvolutionConfig
from evolution_agent.llm.base import BaseLLMClient
from evolution_agent.llm.cloud_client import CloudClient
from evolution_agent.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


def _make_client(model_spec: str, config: EvolutionConfig) -> BaseLLMClient:
    """Create an LLM client from a model spec string.

    Supported formats:
      - "claude-code:haiku" → ClaudeCodeClient(model="haiku")
      - "claude-code"       → ClaudeCodeClient(model="opus")
      - "qwen2.5-coder:7b"  → OllamaClient
      - "claude-haiku-..."   → CloudClient
    """
    if model_spec.startswith("claude-code"):
        from evolution_agent.llm.claude_code_client import ClaudeCodeClient
        parts = model_spec.split(":", 1)
        model_hint = parts[1] if len(parts) > 1 else "opus"
        return ClaudeCodeClient(model=model_hint)
    elif ":" in model_spec and not model_spec.startswith("claude"):
        # Looks like an Ollama model (e.g. "qwen2.5-coder:7b")
        return OllamaClient(model=model_spec, base_url=config.ollama_base_url)
    else:
        return CloudClient(
            model=model_spec,
            api_key=config.cloud_api_key,
            base_url=config.cloud_base_url or None,
        )


class LLMRouter:
    """Routes LLM calls between mutator (local) and analyzer (cloud/claude-code).

    - Mutator: Ollama, Claude Code CLI, or cloud model
    - Analyzer: Cloud model or Claude Code CLI for population analysis
    - Fallback: If primary mutator unavailable, use fallback model
    """

    def __init__(self, config: EvolutionConfig) -> None:
        self._config = config
        self._mutator = _make_client(config.mutator_model, config)
        self._analyzer: BaseLLMClient = _make_client(config.analyzer_model, config)
        self._fallback = _make_client(config.fallback_model, config)
        self._mutator_available: bool | None = None  # lazy check
        self._fallback_warned = False

    async def get_mutator(self) -> BaseLLMClient:
        """Get the LLM client for mutation generation."""
        if self._mutator_available is None:
            self._mutator_available = await self._mutator.is_available()
            if self._mutator_available:
                logger.info("Using [%s] for mutations", self._mutator.model_name)
            else:
                logger.warning(
                    "%s unavailable, falling back to [%s]",
                    self._mutator.model_name, self._fallback.model_name,
                )

        if self._mutator_available:
            return self._mutator

        if not self._fallback_warned:
            logger.warning(
                "*** Using fallback model '%s' for mutations. ***",
                self._fallback.model_name,
            )
            self._fallback_warned = True
        return self._fallback

    async def get_analyzer(self) -> BaseLLMClient:
        """Get the LLM client for population analysis."""
        return self._analyzer

    async def refresh_mutator_status(self) -> bool:
        """Re-check mutator availability."""
        self._mutator_available = await self._mutator.is_available()
        return self._mutator_available

    def get_stats(self) -> dict[str, Any]:
        """Return combined stats from all clients."""
        return {
            "mutator": {
                "model": self._mutator.model_name,
                "available": self._mutator_available,
                **self._mutator.stats.__dict__,
            },
            "analyzer": {
                "model": self._analyzer.model_name,
                **self._analyzer.stats.__dict__,
            },
            "fallback": {
                "model": self._fallback.model_name,
                **self._fallback.stats.__dict__,
            },
        }

    async def close(self) -> None:
        if hasattr(self._mutator, 'close'):
            await self._mutator.close()
