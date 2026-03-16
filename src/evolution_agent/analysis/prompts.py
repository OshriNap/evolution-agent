"""Analyzer prompt templates for population analysis."""

from __future__ import annotations

from typing import Any

from evolution_agent.core.types import Individual


def build_analyzer_system_prompt(pattern_library_text: str) -> str:
    return (
        "You are an expert evolutionary optimization analyst. You observe the "
        "state of an evolving code population and provide structured analysis.\n\n"
        "Your role:\n"
        "1. Identify patterns in the population's evolution\n"
        "2. Suggest specific improvements for the next generations\n"
        "3. Detect known anti-patterns (premature convergence, bloat, etc.)\n"
        "4. Guide mutation strategy (more point vs structural, crossover, etc.)\n\n"
        f"{pattern_library_text}\n\n"
        "Respond with ONLY a JSON object:\n"
        "{\n"
        '  "observations": ["..."],\n'
        '  "conclusions": ["..."],\n'
        '  "suggestions": [\n'
        '    {"content": "...", "category": "mutation|structural|diagnostic", '
        '"priority": "high|medium|low", "rationale": "..."}\n'
        "  ],\n"
        '  "detected_patterns": ["pattern_id", ...],\n'
        '  "mutation_guidance": "specific guidance for the mutator LLM",\n'
        '  "recommended_mutation_weights": {"point": 0.6, "structural": 0.4},\n'
        '  "phase": "exploring|exploiting|stuck"\n'
        "}"
    )


def build_analyzer_user_prompt(
    generation: int,
    population: list[Individual],
    generation_history: list[dict[str, Any]],
    scratchpad_text: str,
    function_spec: str,
    direction: str,
    max_generations_remaining: int,
) -> str:
    parts: list[str] = []

    parts.append(f"## Generation {generation}")
    parts.append(f"**Direction**: {direction} fitness")
    parts.append(f"**Remaining generations**: {max_generations_remaining}")
    parts.append(f"**Function spec**: {function_spec}")
    parts.append("")

    # Top 5 individuals with full code
    sorted_pop = sorted(
        population, key=lambda x: x.fitness,
        reverse=(direction == "maximize"),
    )
    parts.append("## Top 5 Individuals (full code)")
    for i, ind in enumerate(sorted_pop[:5]):
        parts.append(
            f"### #{i+1} (fitness={ind.fitness:.6f}, "
            f"mutation={ind.mutation_type.value if ind.mutation_type else 'seed'}, "
            f"gen={ind.generation})"
        )
        parts.append(f"```python\n{ind.code}\n```")
        parts.append("")

    # Rest as summary
    if len(sorted_pop) > 5:
        parts.append("## Remaining Population (summary)")
        for ind in sorted_pop[5:]:
            first_line = ind.code.split("\n")[0][:80]
            parts.append(
                f"- fitness={ind.fitness:.6f} | {first_line}"
            )
        parts.append("")

    # Generation history
    if generation_history:
        parts.append("## Generation History")
        for gh in generation_history[-10:]:
            parts.append(
                f"- Gen {gh['number']}: best={gh['best_fitness']:.6f} "
                f"avg={gh['avg_fitness']:.6f} diversity={gh['diversity']:.2f}"
            )
        parts.append("")

    # Scratchpad
    parts.append("## Evolution Scratchpad")
    parts.append(scratchpad_text)

    return "\n".join(parts)


def build_library_update_prompt(
    scratchpad_text: str,
    pattern_library_text: str,
    run_summary: dict[str, Any] | None,
) -> str:
    parts: list[str] = []
    parts.append("## Pattern Library")
    parts.append(pattern_library_text)
    parts.append("")
    parts.append("## Full Scratchpad")
    parts.append(scratchpad_text)
    parts.append("")
    if run_summary:
        parts.append("## Run Summary")
        for k, v in run_summary.items():
            parts.append(f"- {k}: {v}")
    parts.append("")
    parts.append(
        "For each known pattern, report if it was detected and if the "
        "reframe helped. Propose any NEW patterns you discovered."
    )
    return "\n".join(parts)
