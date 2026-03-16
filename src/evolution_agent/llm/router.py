"""LLM Router: two-tier routing with fallback logic."""

from __future__ import annotations

import logging
from typing import Any

from evolution_agent.core.types import EvolutionConfig
from evolution_agent.llm.base import BaseLLMClient
from evolution_agent.llm.cloud_client import CloudClient
from evolution_agent.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class LLMRouter:
    """Routes LLM calls between mutator (local) and analyzer (cloud/claude-code).

    - Mutator: Ollama local model for cheap, fast mutation generation
    - Analyzer: Cloud model or Claude Code CLI for population analysis
    - Fallback: If Ollama unavailable, use cheap cloud model (haiku)
    """

    def __init__(self, config: EvolutionConfig) -> None:
        self._config = config
        self._ollama = OllamaClient(
            model=config.mutator_model,
            base_url=config.ollama_base_url,
        )

        # Analyzer: use Claude Code CLI if model starts with "claude-code:"
        # or if analyzer_model is "claude-code"
        if config.analyzer_model.startswith("claude-code"):
            from evolution_agent.llm.claude_code_client import ClaudeCodeClient
            # Extract model hint: "claude-code:opus" → "opus", "claude-code" → "opus"
            parts = config.analyzer_model.split(":", 1)
            model_hint = parts[1] if len(parts) > 1 else "opus"
            self._analyzer: BaseLLMClient = ClaudeCodeClient(model=model_hint)
        else:
            self._analyzer = CloudClient(
                model=config.analyzer_model,
                api_key=config.cloud_api_key,
                base_url=config.cloud_base_url or None,
            )

        self._fallback = CloudClient(
            model=config.fallback_model,
            api_key=config.cloud_api_key,
            base_url=config.cloud_base_url or None,
        )
        self._ollama_available: bool | None = None  # lazy check
        self._fallback_warned = False

    async def get_mutator(self) -> BaseLLMClient:
        """Get the LLM client for mutation generation."""
        if self._ollama_available is None:
            self._ollama_available = await self._ollama.is_available()
            if self._ollama_available:
                logger.info("Using Ollama [%s] for mutations", self._config.mutator_model)
            else:
                logger.warning(
                    "Ollama unavailable, falling back to cloud model [%s]",
                    self._config.fallback_model,
                )

        if self._ollama_available:
            return self._ollama

        if not self._fallback_warned:
            logger.warning(
                "*** COST WARNING: Using cloud model '%s' for mutations. "
                "This will be significantly more expensive than local Ollama. ***",
                self._config.fallback_model,
            )
            self._fallback_warned = True
        return self._fallback

    async def get_analyzer(self) -> BaseLLMClient:
        """Get the LLM client for population analysis."""
        return self._analyzer

    async def refresh_ollama_status(self) -> bool:
        """Re-check Ollama availability (e.g. if it was started mid-run)."""
        self._ollama_available = await self._ollama.is_available()
        return self._ollama_available

    def get_stats(self) -> dict[str, Any]:
        """Return combined stats from all clients."""
        return {
            "ollama": {
                "model": self._ollama.model_name,
                "available": self._ollama_available,
                **self._ollama.stats.__dict__,
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
        await self._ollama.close()
