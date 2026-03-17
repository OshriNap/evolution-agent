"""Extract GRPO training pairs from evolution run logs.

Reconstructs (system_prompt, user_prompt, response, reward) tuples from
JSONL run logs. Designed for fine-tuning a general-purpose code mutation
model — rewards are based on general skills (format compliance, code
validity, guidance following, improvement) NOT problem-specific knowledge.

Usage:
    python scripts/extract_training_data.py [--runs-dir runs/] [--output training_data.jsonl]
    python scripts/extract_training_data.py --format grpo  # paired chosen/rejected
    python scripts/extract_training_data.py --format sft   # only positive examples
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evolution_agent.mutation.prompts import (
    crossover_prompt,
    guided_mutation_prompt,
    point_mutation_prompt,
    structural_mutation_prompt,
)


def _reconstruct_prompt(
    mutation_type: str,
    parent_code: str,
    function_spec: str,
    guidance: str = "",
    partner_code: str = "",
    parent_fitness: float = 0.0,
    partner_fitness: float = 0.0,
) -> tuple[str, str]:
    """Reconstruct the (system, user) prompt that was sent to the LLM."""
    if mutation_type == "point":
        return point_mutation_prompt(parent_code, function_spec, guidance)
    elif mutation_type == "structural":
        return structural_mutation_prompt(parent_code, function_spec, guidance)
    elif mutation_type == "crossover":
        return crossover_prompt(
            parent_code, partner_code, function_spec,
            parent_fitness, partner_fitness,
        )
    elif mutation_type == "guided":
        return guided_mutation_prompt(parent_code, function_spec, guidance)
    else:
        return structural_mutation_prompt(parent_code, function_spec, guidance)


def _format_response(code: str, description: str = "") -> str:
    """Format a mutation response in the expected output format."""
    desc = description or "Code mutation"
    return f"DESCRIPTION: {desc}\n\n```python\n{code}\n```"


def _compute_reward(
    parent_fitness: float,
    child_fitness: float,
    child_code: str,
    parent_code: str,
    had_error: bool,
    guidance: str = "",
) -> tuple[float, dict]:
    """Compute a general-purpose reward for a mutation.

    Rewards general skills, NOT problem-specific knowledge:
    - Format compliance (code compiles, has right signature)
    - Guidance following (if guidance was given)
    - Meaningful change (not a clone)
    - Fitness improvement (relative, not absolute)

    Returns (reward, reward_breakdown).
    """
    reward = 0.0
    breakdown = {}

    # 1. Basic validity: code ran without errors
    if had_error:
        breakdown["validity"] = -1.0
        return -1.0, breakdown
    breakdown["validity"] = 0.2
    reward += 0.2

    # 2. Non-trivial change: not identical to parent
    if _code_hash(child_code) == _code_hash(parent_code):
        breakdown["novelty"] = -0.5
        return reward - 0.5, breakdown
    breakdown["novelty"] = 0.1
    reward += 0.1

    # 3. Fitness improvement (relative to parent)
    if parent_fitness > 0:
        delta = child_fitness - parent_fitness
        relative_delta = delta / max(abs(parent_fitness), 0.01)
        # Clip to [-1, 1] range
        improvement_score = max(-1.0, min(1.0, relative_delta * 5))
        breakdown["improvement"] = round(improvement_score, 3)
        reward += improvement_score * 0.5  # weight improvement at 0.5
    else:
        # Parent had 0 fitness, any positive child is good
        if child_fitness > 0:
            breakdown["improvement"] = 0.5
            reward += 0.25
        else:
            breakdown["improvement"] = -0.3
            reward -= 0.15

    # 4. Uses p dict pattern (general skill: parameterization)
    if 'p = {' in child_code or 'p={' in child_code:
        breakdown["p_dict"] = 0.1
        reward += 0.1
    elif 'p = {' in parent_code:
        # Parent had p dict, child dropped it — bad
        breakdown["p_dict"] = -0.2
        reward -= 0.2
    else:
        breakdown["p_dict"] = 0.0

    # 5. Guidance following (general skill: instruction compliance)
    if guidance:
        # Check if key terms from guidance appear in the code
        guidance_lower = guidance.lower()
        code_lower = child_code.lower()
        key_terms = []
        for term in ["2-opt", "2opt", "or-opt", "oropt", "3-opt", "greedy",
                      "insertion", "nearest neighbor", "multi-start",
                      "perturbation", "local search", "swap", "reverse"]:
            if term in guidance_lower:
                key_terms.append(term)

        if key_terms:
            matched = sum(1 for t in key_terms if t in code_lower)
            ratio = matched / len(key_terms)
            breakdown["guidance_follow"] = round(ratio * 0.3, 3)
            reward += ratio * 0.3
        else:
            breakdown["guidance_follow"] = 0.0
    else:
        breakdown["guidance_follow"] = 0.0

    return round(reward, 3), breakdown


def _code_hash(code: str) -> str:
    return hashlib.sha256(code.strip().encode()).hexdigest()[:16]


def extract_from_run(
    run_dir: Path,
    function_spec: str = "",
) -> list[dict]:
    """Extract training examples from one run directory."""
    log_path = run_dir / "evolution.jsonl"
    if not log_path.exists():
        return []

    with open(log_path) as f:
        events = [json.loads(l) for l in f if l.strip()]

    # Build lookup: id → (code, fitness, mutation_type)
    individuals = {}
    best_codes = {}
    guidance_history = []

    for ev in events:
        if ev["type"] == "best_code":
            gen = ev["data"]["generation"]
            best_codes[gen] = ev["data"]

        if ev["type"] == "evaluation":
            d = ev["data"]
            individuals[d["id"]] = {
                "fitness": d["fitness"],
                "mutation_type": d.get("mutation_type"),
                "error": d.get("error"),
            }

        if ev["type"] == "analysis":
            guidance_history.append(ev["data"].get("mutation_guidance", ""))

    # Get function spec from config if not provided
    if not function_spec:
        # Use a generic spec placeholder
        function_spec = "# Optimize the given function"

    # Build training pairs from best_code entries
    # Each gen's best_code gives us (parent_code, child_code, fitness_delta)
    examples = []
    prev_best = None

    for gen in sorted(best_codes.keys()):
        bc = best_codes[gen]
        child_code = bc["code"]
        child_fitness = bc["fitness"]
        child_id = bc["id"]

        # Get mutation info
        ind_info = individuals.get(child_id, {})
        mutation_type = ind_info.get("mutation_type", "structural")
        had_error = ind_info.get("error") is not None

        if prev_best is None:
            prev_best = bc
            continue

        parent_code = prev_best["code"]
        parent_fitness = prev_best["fitness"]

        # Get guidance that was active at this generation
        guidance = ""
        if guidance_history:
            # Use latest guidance before this gen
            guidance_idx = min(gen // 5, len(guidance_history) - 1)
            if guidance_idx >= 0:
                guidance = guidance_history[guidance_idx]

        # Reconstruct prompt
        if mutation_type and mutation_type != "None":
            system, user = _reconstruct_prompt(
                mutation_type, parent_code, function_spec,
                guidance=guidance,
                parent_fitness=parent_fitness,
            )
        else:
            system, user = structural_mutation_prompt(
                parent_code, function_spec, guidance,
            )

        # Compute reward
        reward, breakdown = _compute_reward(
            parent_fitness, child_fitness, child_code, parent_code,
            had_error, guidance,
        )

        # Format response
        response = _format_response(child_code)

        examples.append({
            "system": system,
            "user": user,
            "response": response,
            "reward": reward,
            "reward_breakdown": breakdown,
            "metadata": {
                "run": run_dir.name,
                "generation": gen,
                "mutation_type": mutation_type,
                "parent_fitness": parent_fitness,
                "child_fitness": child_fitness,
                "fitness_delta": round(child_fitness - parent_fitness, 6),
            },
        })

        prev_best = bc

    # Also extract ALL evaluations (not just best) for richer data
    all_evals = []
    code_by_id = {}

    for ev in events:
        if ev["type"] == "best_code":
            code_by_id[ev["data"]["id"]] = ev["data"]["code"]

    # For non-best individuals, we need their code from the log
    # Currently we only have best_code logged per gen, so we work with what we have

    return examples


def extract_all_runs(
    runs_dir: Path,
    function_spec: str = "",
) -> list[dict]:
    """Extract training data from all runs."""
    all_examples = []
    for run_dir in sorted(runs_dir.iterdir()):
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            examples = extract_from_run(run_dir, function_spec)
            all_examples.extend(examples)
            if examples:
                print(f"  {run_dir.name}: {len(examples)} examples")
    return all_examples


def to_grpo_pairs(examples: list[dict]) -> list[dict]:
    """Convert to GRPO format: pairs of (chosen, rejected) responses.

    Groups by similar prompts and pairs high-reward with low-reward responses.
    """
    # Sort by reward
    positive = [e for e in examples if e["reward"] > 0.2]
    negative = [e for e in examples if e["reward"] <= 0]

    pairs = []
    # Pair positives with negatives
    for pos in positive:
        for neg in negative:
            # Same mutation type = comparable pair
            if (pos["metadata"]["mutation_type"] == neg["metadata"]["mutation_type"]):
                pairs.append({
                    "prompt": [
                        {"role": "system", "content": pos["system"]},
                        {"role": "user", "content": pos["user"]},
                    ],
                    "chosen": pos["response"],
                    "rejected": neg["response"],
                    "chosen_reward": pos["reward"],
                    "rejected_reward": neg["reward"],
                })
                break  # One pair per positive

    return pairs


def to_sft(examples: list[dict], min_reward: float = 0.3) -> list[dict]:
    """Convert to SFT format: only positive examples as instruction-response pairs."""
    return [
        {
            "messages": [
                {"role": "system", "content": e["system"]},
                {"role": "user", "content": e["user"]},
                {"role": "assistant", "content": e["response"]},
            ],
            "reward": e["reward"],
        }
        for e in examples
        if e["reward"] >= min_reward
    ]


def main():
    parser = argparse.ArgumentParser(description="Extract GRPO training data from evolution runs")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("training_data.jsonl"))
    parser.add_argument("--format", choices=["raw", "grpo", "sft"], default="raw")
    parser.add_argument("--min-reward", type=float, default=0.3, help="Min reward for SFT examples")
    args = parser.parse_args()

    print(f"Extracting from {args.runs_dir}...")
    examples = extract_all_runs(args.runs_dir)
    print(f"\nTotal examples: {len(examples)}")

    if not examples:
        print("No training data found.")
        return

    # Stats
    rewards = [e["reward"] for e in examples]
    positive = sum(1 for r in rewards if r > 0)
    negative = sum(1 for r in rewards if r <= 0)
    print(f"Positive (reward > 0): {positive}")
    print(f"Negative (reward <= 0): {negative}")
    print(f"Avg reward: {sum(rewards)/len(rewards):.3f}")
    print(f"Max reward: {max(rewards):.3f}")
    print(f"Min reward: {min(rewards):.3f}")

    # Format output
    if args.format == "grpo":
        output = to_grpo_pairs(examples)
        print(f"GRPO pairs: {len(output)}")
    elif args.format == "sft":
        output = to_sft(examples, args.min_reward)
        print(f"SFT examples (reward >= {args.min_reward}): {len(output)}")
    else:
        output = examples

    # Write
    with open(args.output, "w") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
