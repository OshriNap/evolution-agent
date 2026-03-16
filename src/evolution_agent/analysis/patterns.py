"""Meta-pattern library: cross-run persistent patterns.

Ported from reference meta_analysis.py, with 8 evolution-specific seed patterns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetaPattern:
    """One validated detector/reframe pair."""

    pattern_id: str
    detector: str
    reframe: str
    source: str = ""  # "human" = protected
    times_detected: int = 0
    times_reframe_helped: int = 0
    success_rate: float = 0.0
    examples: list[dict[str, Any]] = field(default_factory=list)


# 8 evolution-specific seed patterns
_SEED_PATTERNS: list[dict[str, Any]] = [
    {
        "pattern_id": "premature_convergence",
        "detector": (
            "Population diversity drops below 30% while fitness has not improved "
            "for 3+ generations. The population is converging prematurely on a "
            "local optimum."
        ),
        "reframe": (
            "Inject random individuals, increase mutation rate, and prefer "
            "structural over point mutations to escape the local optimum. "
            "Consider re-injecting hall-of-fame individuals with high diversity."
        ),
    },
    {
        "pattern_id": "fitness_plateau",
        "detector": (
            "Best fitness has not improved for N generations (N >= 5). "
            "Point mutations are not finding better solutions."
        ),
        "reframe": (
            "Switch to structural mutations. Try fundamentally different "
            "approaches: different algorithms, data structures, or control flow. "
            "Point mutations are exhausted in this region."
        ),
    },
    {
        "pattern_id": "crossover_ineffective",
        "detector": (
            "Crossover offspring consistently have lower fitness than both parents. "
            "The parents may be too different for meaningful recombination."
        ),
        "reframe": (
            "Reduce crossover rate or restrict crossover to parents with similar "
            "fitness levels. The population may have diverged into incompatible "
            "lineages."
        ),
    },
    {
        "pattern_id": "mutation_too_aggressive",
        "detector": (
            "Most mutations (>70%) produce individuals with worse fitness than "
            "their parents. The mutations are too destructive."
        ),
        "reframe": (
            "Shift mutation weights toward point mutations and reduce structural "
            "mutation rate. The current code is close to a good solution and "
            "needs fine-tuning, not restructuring."
        ),
    },
    {
        "pattern_id": "mutation_too_conservative",
        "detector": (
            "Most mutations (<10% difference from parent fitness) don't meaningfully "
            "change fitness. The mutations are too timid to explore."
        ),
        "reframe": (
            "Increase structural mutation rate and add more aggressive prompting. "
            "The search space near the current solutions is flat and needs larger "
            "jumps to find improvement."
        ),
    },
    {
        "pattern_id": "overfitting_test_cases",
        "detector": (
            "Fitness is high but the evolved code uses hardcoded values, lookup "
            "tables, or special-cases specific test inputs rather than solving "
            "the general problem."
        ),
        "reframe": (
            "Add diversity to test cases, penalize code length (parsimony), and "
            "explicitly instruct mutations to avoid hardcoding. Consider adding "
            "generalization metrics to the fitness function."
        ),
    },
    {
        "pattern_id": "bloat",
        "detector": (
            "Code length grows across generations without corresponding fitness "
            "improvement. Individuals accumulate dead code or unnecessary complexity."
        ),
        "reframe": (
            "Apply parsimony pressure: penalize code length in fitness, or prefer "
            "shorter code at equal fitness. Explicitly prompt mutations to simplify "
            "and remove dead code."
        ),
    },
    {
        "pattern_id": "lost_good_traits",
        "detector": (
            "A beneficial trait (algorithm, technique) that appeared in earlier "
            "generations has disappeared from the current population. Good genes "
            "were lost through drift."
        ),
        "reframe": (
            "Increase elitism count to preserve more good individuals. Re-inject "
            "hall-of-fame individuals that had the lost trait. Consider increasing "
            "population size to maintain diversity."
        ),
    },
]


class MetaPatternLibrary:
    """Persistent library of validated meta-patterns. Stored as JSON on disk."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = Path("meta_patterns.json")
        self._path = Path(path)
        self._history_path = self._path.with_suffix(".history.jsonl")
        self._version = 0
        self._patterns: list[MetaPattern] = []

    def load(self) -> None:
        """Load patterns from disk, or seed defaults if file doesn't exist."""
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._patterns = [
                    MetaPattern(**p) for p in data.get("patterns", [])
                ]
                self._version = data.get("save_version", 0)
                logger.info(
                    "Loaded %d meta-patterns (v%d) from %s",
                    len(self._patterns), self._version, self._path,
                )
                return
            except Exception as e:
                logger.warning("Failed to load meta-patterns: %s", e)

        self.seed_defaults()
        self.save()

    def save(self) -> None:
        """Write patterns to disk with version bump."""
        self._version += 1
        data = {
            "version": "1.0",
            "save_version": self._version,
            "patterns": [asdict(p) for p in self._patterns],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        self._log_history("save", {"pattern_count": len(self._patterns)})
        logger.info(
            "Saved %d meta-patterns (v%d) to %s",
            len(self._patterns), self._version, self._path,
        )

    def _log_history(self, action: str, details: dict[str, Any]) -> None:
        import time as _time
        record = {
            "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": self._version,
            "action": action,
            "pattern_count": len(self._patterns),
            **details,
        }
        try:
            with open(self._history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("History log write failed: %s", e)

    def seed_defaults(self) -> None:
        self._patterns = [MetaPattern(**p) for p in _SEED_PATTERNS]

    def get_patterns(self) -> list[MetaPattern]:
        return list(self._patterns)

    def get_pattern(self, pattern_id: str) -> MetaPattern | None:
        for p in self._patterns:
            if p.pattern_id == pattern_id:
                return p
        return None

    def update_pattern(
        self,
        pattern_id: str,
        *,
        detected: bool = False,
        helped: bool = False,
        example: dict[str, Any] | None = None,
    ) -> None:
        pat = self.get_pattern(pattern_id)
        if pat is None:
            return
        if detected:
            pat.times_detected += 1
        if helped:
            pat.times_reframe_helped += 1
        if pat.times_detected > 0:
            pat.success_rate = pat.times_reframe_helped / pat.times_detected
        if example:
            pat.examples.append(example)
            if len(pat.examples) > 10:
                pat.examples = pat.examples[-10:]

    def add_pattern(self, pattern: MetaPattern) -> None:
        if self.get_pattern(pattern.pattern_id):
            logger.warning("Pattern %s already exists, skipping", pattern.pattern_id)
            return
        self._patterns.append(pattern)
        self._log_history("add", {
            "pattern_id": pattern.pattern_id,
            "detector": pattern.detector[:100],
        })

    def remove_pattern(self, pattern_id: str) -> bool:
        for i, p in enumerate(self._patterns):
            if p.pattern_id == pattern_id:
                self._patterns.pop(i)
                self._log_history("remove", {"pattern_id": pattern_id})
                return True
        return False

    def merge_patterns(
        self,
        keep_id: str,
        remove_id: str,
        new_detector: str | None = None,
        new_reframe: str | None = None,
    ) -> bool:
        keep = self.get_pattern(keep_id)
        remove = self.get_pattern(remove_id)
        if keep is None or remove is None:
            return False

        keep.times_detected += remove.times_detected
        keep.times_reframe_helped += remove.times_reframe_helped
        if keep.times_detected > 0:
            keep.success_rate = keep.times_reframe_helped / keep.times_detected
        combined = keep.examples + remove.examples
        keep.examples = combined[-10:]

        if new_detector:
            keep.detector = new_detector
        if new_reframe:
            keep.reframe = new_reframe

        self.remove_pattern(remove_id)
        return True

    def format_for_prompt(self) -> str:
        if not self._patterns:
            return "(Pattern library is empty.)"

        lines = ["## Known Evolution Patterns\n"]
        for p in self._patterns:
            stats = ""
            if p.times_detected > 0:
                stats = (
                    f" [detected {p.times_detected}x, "
                    f"helped {p.times_reframe_helped}x, "
                    f"rate {p.success_rate:.0%}]"
                )
            lines.append(f"### {p.pattern_id}{stats}")
            lines.append(f"**Detector**: {p.detector}")
            lines.append(f"**Reframe**: {p.reframe}")
            lines.append("")
        return "\n".join(lines)
