"""Prompt templates for mutation operations.

Uses a markdown code-fence output format instead of JSON-embedded code.
Local models (qwen, llama, etc.) are much more reliable at producing
code in ```python blocks than properly JSON-escaping multi-line strings.
"""

from __future__ import annotations

# Shared output format instructions appended to all system prompts.
_OUTPUT_FORMAT = (
    "## Output format\n"
    "You MUST respond in EXACTLY this format, nothing else:\n\n"
    "DESCRIPTION: <one sentence describing the change>\n\n"
    "```python\n"
    "<the complete modified function — NO docstrings, use # comments only>\n"
    "```\n\n"
    "RULES:\n"
    "- Output the COMPLETE function, not a diff\n"
    "- Do NOT use triple-quoted docstrings (\"\"\"...\"\"\") — use # comments instead\n"
    "- Do NOT add any text before DESCRIPTION or after the code block\n"
    "- Do NOT wrap in JSON\n"
    "- Put tunable numeric constants in a `p` dict with defaults, like:\n"
    "    if p is None:\n"
    "        p = {\"alpha\": 1.0, \"threshold\": 0.5}\n"
    "  This allows automatic parameter optimization after your code is evaluated.\n"
    "  Use `p[\"name\"]` to access parameters in the code body.\n"
)


def point_mutation_prompt(
    code: str,
    function_spec: str,
    guidance: str = "",
) -> tuple[str, str]:
    """Prompt for small, targeted code changes."""
    system = (
        "You are a code mutation engine. You make small, targeted improvements "
        "to Python functions.\n\n"
        + _OUTPUT_FORMAT
    )

    user = (
        f"## Function specification\n{function_spec}\n\n"
        f"## Current code\n```python\n{code}\n```\n\n"
        "## Task\n"
        "Make a SMALL, targeted improvement. Examples:\n"
        "- Change a constant or threshold\n"
        "- Modify a comparison operator\n"
        "- Add or remove a simple condition\n"
        "- Tweak arithmetic expressions\n\n"
        "Keep the function signature identical.\n"
    )
    if guidance:
        user += f"\n## Guidance\n{guidance}\n"
    return system, user


def structural_mutation_prompt(
    code: str,
    function_spec: str,
    guidance: str = "",
) -> tuple[str, str]:
    """Prompt for larger structural changes."""
    system = (
        "You are a code evolution engine. You make structural improvements "
        "to Python functions — changing algorithms, data structures, or control flow.\n\n"
        + _OUTPUT_FORMAT
    )

    user = (
        f"## Function specification\n{function_spec}\n\n"
        f"## Current code\n```python\n{code}\n```\n\n"
        "## Task\n"
        "Make a STRUCTURAL change. Examples:\n"
        "- Replace the algorithm entirely\n"
        "- Change the data structure used\n"
        "- Restructure control flow (loops, recursion, etc.)\n"
        "- Add caching or memoization\n\n"
        "Keep the function signature identical. The code should still be correct.\n"
    )
    if guidance:
        user += f"\n## Guidance\n{guidance}\n"
    return system, user


def crossover_prompt(
    code_a: str,
    code_b: str,
    function_spec: str,
    fitness_a: float,
    fitness_b: float,
) -> tuple[str, str]:
    """Prompt for combining two parent solutions."""
    system = (
        "You are a code crossover engine. You combine the best aspects of "
        "two parent solutions into a new offspring.\n\n"
        + _OUTPUT_FORMAT
    )

    user = (
        f"## Function specification\n{function_spec}\n\n"
        f"## Parent A (fitness: {fitness_a:.6f})\n```python\n{code_a}\n```\n\n"
        f"## Parent B (fitness: {fitness_b:.6f})\n```python\n{code_b}\n```\n\n"
        "## Task\n"
        "Create an offspring that combines the best ideas from both parents.\n"
        "- Take the strongest aspects from the higher-fitness parent\n"
        "- Incorporate useful innovations from the other parent\n"
        "- Keep the function signature identical\n"
    )
    return system, user


def guided_mutation_prompt(
    code: str,
    function_spec: str,
    guidance: str,
    population_context: str = "",
) -> tuple[str, str]:
    """Prompt for analyzer-guided mutation."""
    system = (
        "You are a code evolution engine guided by population analysis. "
        "You implement specific suggestions from the analyzer.\n\n"
        + _OUTPUT_FORMAT
    )

    user = (
        f"## Function specification\n{function_spec}\n\n"
        f"## Current code\n```python\n{code}\n```\n\n"
        f"## Analyzer guidance\n{guidance}\n\n"
    )
    if population_context:
        user += f"## Population context\n{population_context}\n\n"
    user += (
        "## Task\n"
        "Implement the analyzer's guidance. Make targeted changes.\n"
        "Keep the function signature identical.\n"
    )
    return system, user
