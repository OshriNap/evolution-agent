"""Per-run scratchpad with rolling digest compression.

Ported from reference meta_analysis.py, adapted for evolution context.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ScratchpadEntry:
    """One entry in the persistent scratchpad."""

    generation: int
    category: str  # "observation" | "conclusion" | "suggestion" | "counterfactual"
    content: str
    source: str = "analyzer"  # "analyzer" | "system" | "meta_optimizer"
    suggestion_id: str | None = None
    followed: bool | None = None
    outcome_note: str | None = None


class Scratchpad:
    """Persistent memory across evolution generations (cleared per run).

    Has two layers:
    - entries: full list of all entries
    - digest: a rolling LLM-compressed summary of older entries
    """

    RECENT_WINDOW = 5  # generations kept verbatim

    def __init__(self) -> None:
        self.entries: list[ScratchpadEntry] = []
        self._digest: str = ""
        self._digest_up_to_gen: int = -1

    @property
    def digest(self) -> str:
        return self._digest

    def set_digest(self, text: str, up_to_gen: int) -> None:
        self._digest = text
        self._digest_up_to_gen = up_to_gen

    def add(self, entry: ScratchpadEntry) -> None:
        self.entries.append(entry)

    async def update_digest(self, llm: Any, current_generation: int) -> None:
        """Compress older entries into a rolling digest via LLM."""
        cutoff = current_generation - self.RECENT_WINDOW
        if cutoff <= self._digest_up_to_gen:
            return

        to_compress = [
            e for e in self.entries
            if self._digest_up_to_gen < e.generation <= cutoff
        ]
        if not to_compress:
            return

        new_text_lines = []
        for e in to_compress:
            new_text_lines.append(f"[gen {e.generation}] ({e.category}) {e.content}")
        new_text = "\n".join(new_text_lines)

        prompt_text = (
            "You are compressing an evolution optimization scratchpad to save tokens.\n"
            "Below is the CURRENT DIGEST (rolling summary of earlier generations),\n"
            "followed by NEW ENTRIES that need to be incorporated.\n\n"
            "Produce an UPDATED DIGEST that:\n"
            "1. Preserves all key insights, patterns, and lessons learned\n"
            "2. Keeps specific fitness numbers/metrics when they matter\n"
            "3. Removes redundancy and repetition\n"
            "4. Stays under 800 words\n"
            "5. Organizes by theme (not chronologically)\n\n"
            f"## Current Digest\n{self._digest or '(empty -- first compression)'}\n\n"
            f"## New Entries (generations {self._digest_up_to_gen + 1}-{cutoff})\n{new_text}\n\n"
            "Write ONLY the updated digest, no preamble."
        )
        try:
            result = await llm.complete(
                "You compress optimization scratchpads into concise digests.",
                [{"role": "user", "content": prompt_text}],
                temperature=0.1,
            )
            self._digest = result.strip()
            self._digest_up_to_gen = cutoff
            logger.info(
                "Scratchpad digest updated (up to gen %d, %d chars)",
                cutoff, len(self._digest),
            )
        except Exception as e:
            logger.warning("Scratchpad digest compression failed: %s", e)

    def get_pending_suggestions(self) -> list[ScratchpadEntry]:
        return [
            e for e in self.entries
            if e.category == "suggestion" and e.followed is None
        ]

    def mark_suggestion_followed(self, suggestion_id: str, followed: bool) -> None:
        for e in self.entries:
            if e.suggestion_id == suggestion_id:
                e.followed = followed

    def format_for_prompt(self) -> str:
        """Render scratchpad as digest + recent entries for LLM prompts."""
        if not self.entries:
            return "(No entries yet -- this is the first generation.)"

        parts: list[str] = []

        if self._digest:
            parts.append("## Summary of Earlier Generations")
            parts.append(self._digest)
            parts.append("")

        cutoff = self._digest_up_to_gen
        recent = [e for e in self.entries if e.generation > cutoff]

        if recent:
            if self._digest:
                parts.append("## Recent Generations (verbatim)")

            for cat_label, cat_key in [
                ("Observations", "observation"),
                ("Conclusions", "conclusion"),
                ("Suggestions", "suggestion"),
                ("Counterfactuals", "counterfactual"),
            ]:
                items = [e for e in recent if e.category == cat_key]
                if not items:
                    continue
                lines = [f"### {cat_label}"]
                for e in items:
                    prefix = f"[gen {e.generation}]"
                    if e.category == "suggestion":
                        status = ""
                        if e.followed is True:
                            status = " [FOLLOWED]"
                        elif e.followed is False:
                            status = " [NOT FOLLOWED]"
                        else:
                            status = " [PENDING]"
                        lines.append(
                            f"- {prefix} (id={e.suggestion_id}){status}: {e.content}"
                        )
                    elif e.category == "counterfactual" and e.outcome_note:
                        lines.append(
                            f"- {prefix}: {e.content} --> Outcome: {e.outcome_note}"
                        )
                    else:
                        lines.append(f"- {prefix}: {e.content}")
                parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def format_full(self) -> str:
        """Render ALL entries (uncompressed). For end-of-run analysis only."""
        if not self.entries:
            return "(No entries.)"
        sections: list[str] = []
        for cat_label, cat_key in [
            ("Observations", "observation"),
            ("Conclusions", "conclusion"),
            ("Suggestions", "suggestion"),
            ("Counterfactuals", "counterfactual"),
        ]:
            items = [e for e in self.entries if e.category == cat_key]
            if not items:
                continue
            lines = [f"### {cat_label}"]
            for e in items:
                prefix = f"[gen {e.generation}]"
                lines.append(f"- {prefix}: {e.content}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def format_failed_approaches(self) -> str | None:
        """Summarize approaches that led to regressions.

        Provided as context so the LLM knows what was tried and failed.
        """
        failed = []
        for e in self.entries:
            if e.category == "conclusion" and any(
                kw in e.content.lower()
                for kw in ("regression", "worse", "catastroph", "broken", "failed", "harmful")
            ):
                failed.append(f"- gen {e.generation}: {e.content[:200]}")
        if not failed:
            return None
        return "\n".join(failed[-6:])

    def to_dict(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self.entries]
