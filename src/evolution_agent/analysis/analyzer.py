"""Evolution analyzer: population analysis via cloud LLM."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from evolution_agent.analysis.patterns import MetaPattern, MetaPatternLibrary
from evolution_agent.analysis.prompts import (
    build_analyzer_system_prompt,
    build_analyzer_user_prompt,
    build_library_update_prompt,
)
from evolution_agent.analysis.scratchpad import Scratchpad, ScratchpadEntry
from evolution_agent.core.types import Individual
from evolution_agent.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Output of one analyzer call."""

    observations: list[str] = field(default_factory=list)
    conclusions: list[str] = field(default_factory=list)
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    detected_patterns: list[str] = field(default_factory=list)
    mutation_guidance: str = ""
    recommended_mutation_weights: dict[str, float] = field(default_factory=dict)
    phase: str = "exploring"


class EvolutionAnalyzer:
    """Analyzes population state and provides guidance via cloud LLM."""

    def __init__(
        self,
        llm: BaseLLMClient,
        pattern_library: MetaPatternLibrary,
    ) -> None:
        self._llm = llm
        self._library = pattern_library

    async def analyze(
        self,
        generation: int,
        population: list[Individual],
        generation_history: list[dict[str, Any]],
        scratchpad: Scratchpad,
        function_spec: str,
        direction: str,
        max_generations_remaining: int,
    ) -> AnalysisResult:
        """Run one analysis pass after evaluation."""
        system_prompt = build_analyzer_system_prompt(
            self._library.format_for_prompt(),
        )
        user_prompt = build_analyzer_user_prompt(
            generation=generation,
            population=population,
            generation_history=generation_history,
            scratchpad_text=scratchpad.format_for_prompt(),
            function_spec=function_spec,
            direction=direction,
            max_generations_remaining=max_generations_remaining,
        )

        try:
            data = await self._llm.complete_json(
                system_prompt,
                [{"role": "user", "content": user_prompt}],
                temperature=0.3,
                max_tokens=4096,
            )
        except Exception as e:
            logger.warning("Analyzer LLM call failed: %s", e)
            return AnalysisResult()

        result = AnalysisResult(
            observations=data.get("observations", []),
            conclusions=data.get("conclusions", []),
            suggestions=data.get("suggestions", []),
            detected_patterns=data.get("detected_patterns", []),
            mutation_guidance=data.get("mutation_guidance", ""),
            recommended_mutation_weights=data.get("recommended_mutation_weights", {}),
            phase=data.get("phase", "exploring"),
        )

        # Update scratchpad
        for obs in result.observations:
            scratchpad.add(ScratchpadEntry(
                generation=generation,
                category="observation",
                content=obs,
                source="analyzer",
            ))

        for conc in result.conclusions:
            scratchpad.add(ScratchpadEntry(
                generation=generation,
                category="conclusion",
                content=conc,
                source="analyzer",
            ))

        for i, sug in enumerate(result.suggestions):
            if isinstance(sug, dict) and "content" in sug:
                scratchpad.add(ScratchpadEntry(
                    generation=generation,
                    category="suggestion",
                    content=sug["content"],
                    source="analyzer",
                    suggestion_id=f"S{generation}_{i}",
                ))

        # Update pattern library
        for pid in result.detected_patterns:
            self._library.update_pattern(pid, detected=True)

        logger.info(
            "Analysis gen %d: phase=%s, %d observations, %d suggestions, "
            "%d patterns detected",
            generation, result.phase, len(result.observations),
            len(result.suggestions), len(result.detected_patterns),
        )

        return result

    async def propose_library_updates(
        self,
        scratchpad: Scratchpad,
        run_summary: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """End-of-run library update."""
        system_prompt = (
            "You are a meta-learning analyst. Given the scratchpad from a "
            "completed evolution run, propose updates to the meta-pattern library.\n\n"
            "Respond with ONLY a JSON object:\n"
            "{\n"
            '  "pattern_updates": [\n'
            '    {"pattern_id": "...", "detected": true/false, '
            '"helped": true/false, "example_context": "...", '
            '"example_outcome": "..."}\n'
            "  ],\n"
            '  "new_patterns": [\n'
            '    {"pattern_id": "...", "detector": "...", "reframe": "..."}\n'
            "  ]\n"
            "}"
        )
        user_prompt = build_library_update_prompt(
            scratchpad_text=scratchpad.format_full(),
            pattern_library_text=self._library.format_for_prompt(),
            run_summary=run_summary,
        )

        try:
            data = await self._llm.complete_json(
                system_prompt,
                [{"role": "user", "content": user_prompt}],
                temperature=0.1,
            )
        except Exception as e:
            logger.warning("Library update LLM call failed: %s", e)
            return []

        changes: list[dict[str, Any]] = []

        for upd in data.get("pattern_updates", []):
            if not isinstance(upd, dict) or "pattern_id" not in upd:
                continue
            self._library.update_pattern(
                upd["pattern_id"],
                detected=upd.get("detected", False),
                helped=upd.get("helped", False),
                example={
                    "context": upd.get("example_context", ""),
                    "outcome": upd.get("example_outcome", ""),
                },
            )
            changes.append(upd)

        for np_entry in data.get("new_patterns", []):
            if not isinstance(np_entry, dict):
                continue
            pid = np_entry.get("pattern_id", "")
            detector = np_entry.get("detector", "")
            reframe = np_entry.get("reframe", "")
            if pid and detector and reframe:
                self._library.add_pattern(MetaPattern(
                    pattern_id=pid,
                    detector=detector,
                    reframe=reframe,
                    times_detected=1,
                ))
                changes.append({"new": True, **np_entry})

        if changes:
            self._library.save()

        return changes

    async def mid_run_library_update(
        self,
        scratchpad: Scratchpad,
        generation: int,
    ) -> list[dict[str, Any]]:
        """Incremental library update called every N*2 generations mid-run.

        Checks which known patterns were detected and whether reframes helped.
        Also consolidates (prune/merge) if library has grown large.
        """
        recent = [
            e for e in scratchpad.entries
            if e.generation >= max(0, generation - 5)
        ]
        if not recent:
            return []

        recent_text = "\n".join(
            f"[gen {e.generation}] ({e.category}) {e.content}"
            for e in recent
        )

        system_prompt = (
            "You are a meta-learning analyst performing a mid-run check-in.\n"
            "Analyze recent scratchpad entries and update the pattern library.\n"
            "Be conservative — only report patterns you are confident about.\n\n"
            "Respond with ONLY a JSON object:\n"
            "{\n"
            '  "pattern_updates": [\n'
            '    {"pattern_id": "...", "detected": true/false, '
            '"helped": true/false}\n'
            "  ],\n"
            '  "new_patterns": [\n'
            '    {"pattern_id": "...", "detector": "...", "reframe": "..."}\n'
            "  ]\n"
            "}"
        )
        user_prompt = (
            f"## Recent scratchpad (gens {max(0, generation-5)}-{generation})\n"
            f"{recent_text}\n\n"
            f"## Pattern library\n{self._library.format_for_prompt()}\n\n"
            "Which patterns were detected? Did the reframes help?"
        )

        try:
            data = await self._llm.complete_json(
                system_prompt,
                [{"role": "user", "content": user_prompt}],
                temperature=0.1,
            )
        except Exception as e:
            logger.warning("Mid-run library update failed: %s", e)
            return []

        changes: list[dict[str, Any]] = []

        for upd in data.get("pattern_updates", []):
            if isinstance(upd, dict) and "pattern_id" in upd:
                self._library.update_pattern(
                    upd["pattern_id"],
                    detected=upd.get("detected", False),
                    helped=upd.get("helped", False),
                )
                changes.append(upd)

        for np_entry in data.get("new_patterns", []):
            if not isinstance(np_entry, dict):
                continue
            pid = np_entry.get("pattern_id", "")
            detector = np_entry.get("detector", "")
            reframe = np_entry.get("reframe", "")
            if pid and detector and reframe:
                self._library.add_pattern(MetaPattern(
                    pattern_id=pid, detector=detector, reframe=reframe,
                    times_detected=1,
                ))
                changes.append({"new": True, **np_entry})

        # Consolidate if library is large
        if len(self._library.get_patterns()) >= 20:
            self._prune_weak_patterns()

        if changes:
            self._library.save()
            logger.info(
                "Mid-run library update at gen %d: %d changes, %d patterns total",
                generation, len(changes), len(self._library.get_patterns()),
            )

        return changes

    def _prune_weak_patterns(self) -> None:
        """Remove auto-discovered patterns that were never detected or never helped."""
        to_remove = []
        for p in self._library.get_patterns():
            if p.source == "human":
                continue
            # Never detected and not a seed (seeds have short IDs)
            if p.times_detected == 0 and len(p.pattern_id) > 25:
                to_remove.append(p.pattern_id)
            # Well-tested but almost never helpful
            elif p.times_detected >= 5 and p.success_rate < 0.1:
                to_remove.append(p.pattern_id)

        for pid in to_remove:
            self._library.remove_pattern(pid)
            logger.info("Pruned weak pattern: %s", pid)
