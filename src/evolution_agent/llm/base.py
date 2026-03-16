"""Abstract base LLM client interface."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMStats:
    """Accumulated LLM usage statistics."""

    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    errors: int = 0
    total_time_s: float = 0.0


class BaseLLMClient(ABC):
    """Interface for LLM clients (local or cloud)."""

    def __init__(self) -> None:
        self.stats = LLMStats()

    @abstractmethod
    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Plain text completion."""

    async def complete_json(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Completion with JSON extraction from response."""
        raw = await self.complete(system, messages, temperature, max_tokens)
        return extract_json(raw)

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this LLM backend is reachable."""


def extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from LLM output, handling ```json fences and truncation.

    Ported from reference client.py.
    """
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence (closed)
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try extracting from unclosed fence (truncated output)
    match = re.search(r"```(?:json)?\s*\n(.*)", text, re.DOTALL)
    if match:
        fragment = match.group(1).rstrip("`\n\r ")
        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            repaired = _repair_truncated_json(fragment)
            if repaired is not None:
                return repaired

    # Try finding the last { ... } block in the text
    depth = 0
    start = -1
    end = -1
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "}":
            if depth == 0:
                end = i
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                start = i
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    depth = 0
                    start = -1

    # Last resort: try to extract mutated_code from malformed JSON
    # Local models often break JSON with triple-quoted Python strings
    # Try triple-quote pattern first (LLMs sometimes use Python-style triple quotes)
    code_match = re.search(
        r'"mutated_code"\s*:\s*"{3}\s*\n(.*?)\n\s*"{3}',
        text, re.DOTALL,
    )
    if not code_match:
        # Try regular quoted string
        code_match = re.search(
            r'"mutated_code"\s*:\s*"((?:[^"\\]|\\.)+)"',
            text, re.DOTALL,
        )
    if code_match:
        code = code_match.group(1)
        if code:
            # Unescape JSON string escapes
            code = code.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
            desc_match = re.search(
                r'"change_description"\s*:\s*"((?:[^"\\]|\\.)*)"', text,
            )
            desc = desc_match.group(1) if desc_match else ""
            return {"mutated_code": code, "change_description": desc}

    raise ValueError(f"Could not extract JSON from LLM output: {text[:200]}...")


def _repair_truncated_json(text: str) -> dict[str, Any] | None:
    """Try to repair truncated JSON by closing open braces/brackets and strings."""
    text = text.rstrip()

    # Close any open string
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
    if in_string:
        text += '"'

    # Remove trailing comma or colon (incomplete entry)
    text = re.sub(r"[,:\s]+$", "", text)

    # Count open braces/brackets and close them
    opens: list[str] = []
    in_str = False
    esc = False
    for ch in text:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            opens.append(ch)
        elif ch in "}]":
            if opens:
                opens.pop()

    for opener in reversed(opens):
        text += "}" if opener == "{" else "]"

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
