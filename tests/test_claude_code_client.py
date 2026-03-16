"""Tests for Claude Code CLI client."""

import asyncio
import shutil

import pytest

from evolution_agent.llm.claude_code_client import ClaudeCodeClient


@pytest.fixture
def client():
    return ClaudeCodeClient(model="sonnet")


@pytest.mark.skipif(
    not shutil.which("claude"),
    reason="claude CLI not installed",
)
class TestClaudeCodeClient:
    @pytest.mark.asyncio
    async def test_is_available(self, client):
        assert await client.is_available()

    @pytest.mark.asyncio
    async def test_model_name(self, client):
        assert client.model_name == "claude-code:sonnet"

    @pytest.mark.asyncio
    async def test_complete_simple(self, client):
        result = await client.complete(
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Reply with only the word 'hello'"}],
            temperature=0.0,
            max_tokens=10,
        )
        assert "hello" in result.lower()
        assert client.stats.total_calls == 1

    @pytest.mark.asyncio
    async def test_complete_json(self, client):
        result = await client.complete_json(
            system="You respond only with JSON objects.",
            messages=[{"role": "user", "content": 'Reply with: {"status": "ok"}'}],
            temperature=0.0,
        )
        assert result.get("status") == "ok"
