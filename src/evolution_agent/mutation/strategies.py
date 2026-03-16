"""Mutation strategy implementations.

Parses LLM output in markdown format:
    DESCRIPTION: <change description>
    ```python
    <code>
    ```
"""

from __future__ import annotations

import logging
import re

from evolution_agent.core.types import Individual, MutationType
from evolution_agent.llm.base import BaseLLMClient
from evolution_agent.mutation.prompts import (
    crossover_prompt,
    guided_mutation_prompt,
    point_mutation_prompt,
    structural_mutation_prompt,
)

logger = logging.getLogger(__name__)


def _parse_mutation_response(raw: str) -> tuple[str, str]:
    """Parse LLM mutation response into (code, description).

    Extracts code from ```python ... ``` blocks and description from
    DESCRIPTION: lines. Robust to various LLM output quirks.

    Returns ("", "") if parsing fails.
    """
    code = ""
    description = ""

    # Extract description
    desc_match = re.search(r"DESCRIPTION:\s*(.+)", raw)
    if desc_match:
        description = desc_match.group(1).strip()

    # Extract code from fenced block (prefer ```python, fall back to ```)
    code_match = re.search(r"```python\s*\n(.*?)```", raw, re.DOTALL)
    if not code_match:
        code_match = re.search(r"```\s*\n(.*?)```", raw, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()

    # If no code block found, try to find a bare function definition
    if not code:
        fn_match = re.search(r"(def \w+\(.*?\).*)", raw, re.DOTALL)
        if fn_match:
            # Take everything from the def to the end, stripping trailing prose
            lines = fn_match.group(1).split("\n")
            code_lines: list[str] = []
            for line in lines:
                # Stop at blank lines followed by non-indented non-code
                if code_lines and not line.strip():
                    code_lines.append(line)
                elif code_lines and line and not line[0].isspace() and not line.startswith("def "):
                    break
                else:
                    code_lines.append(line)
            code = "\n".join(code_lines).rstrip()

    return code, description


async def apply_point_mutation(
    parent: Individual,
    llm: BaseLLMClient,
    function_spec: str,
    guidance: str = "",
) -> Individual | None:
    """Apply a small, targeted mutation."""
    system, user = point_mutation_prompt(parent.code, function_spec, guidance)
    try:
        raw = await llm.complete(system, [{"role": "user", "content": user}])
        code, desc = _parse_mutation_response(raw)
        if not code:
            logger.debug("Point mutation: no code extracted from response")
            return None
        return Individual(
            code=code,
            parent_ids=[parent.id],
            mutation_type=MutationType.POINT,
            metadata={"change": desc},
        )
    except Exception as e:
        logger.warning("Point mutation failed: %s", e)
        return None


async def apply_structural_mutation(
    parent: Individual,
    llm: BaseLLMClient,
    function_spec: str,
    guidance: str = "",
) -> Individual | None:
    """Apply a structural/algorithmic change."""
    system, user = structural_mutation_prompt(parent.code, function_spec, guidance)
    try:
        raw = await llm.complete(system, [{"role": "user", "content": user}])
        code, desc = _parse_mutation_response(raw)
        if not code:
            logger.debug("Structural mutation: no code extracted from response")
            return None
        return Individual(
            code=code,
            parent_ids=[parent.id],
            mutation_type=MutationType.STRUCTURAL,
            metadata={"change": desc},
        )
    except Exception as e:
        logger.warning("Structural mutation failed: %s", e)
        return None


async def apply_crossover(
    parent_a: Individual,
    parent_b: Individual,
    llm: BaseLLMClient,
    function_spec: str,
) -> Individual | None:
    """Combine two parents into an offspring."""
    system, user = crossover_prompt(
        parent_a.code, parent_b.code, function_spec,
        parent_a.fitness, parent_b.fitness,
    )
    try:
        raw = await llm.complete(system, [{"role": "user", "content": user}])
        code, desc = _parse_mutation_response(raw)
        if not code:
            logger.debug("Crossover: no code extracted from response")
            return None
        return Individual(
            code=code,
            parent_ids=[parent_a.id, parent_b.id],
            mutation_type=MutationType.CROSSOVER,
            metadata={"change": desc},
        )
    except Exception as e:
        logger.warning("Crossover failed: %s", e)
        return None


async def apply_guided_mutation(
    parent: Individual,
    llm: BaseLLMClient,
    function_spec: str,
    guidance: str,
    population_context: str = "",
) -> Individual | None:
    """Apply analyzer-guided mutation."""
    system, user = guided_mutation_prompt(
        parent.code, function_spec, guidance, population_context,
    )
    try:
        raw = await llm.complete(system, [{"role": "user", "content": user}])
        code, desc = _parse_mutation_response(raw)
        if not code:
            logger.debug("Guided mutation: no code extracted from response")
            return None
        return Individual(
            code=code,
            parent_ids=[parent.id],
            mutation_type=MutationType.GUIDED,
            metadata={
                "change": desc,
                "guidance": guidance[:200],
            },
        )
    except Exception as e:
        logger.warning("Guided mutation failed: %s", e)
        return None
