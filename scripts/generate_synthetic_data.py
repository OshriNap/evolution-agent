"""Generate synthetic training data for the evolution mutator LoRA.

Creates high-quality (prompt, response) pairs across diverse problem domains
using Claude to generate gold-standard mutations. Teaches general mutation
skills: format compliance, guidance following, p-dict usage, valid code.

Usage:
    python scripts/generate_synthetic_data.py --output synthetic_data.jsonl
    python scripts/generate_synthetic_data.py --dry-run  # show prompts only
    python scripts/generate_synthetic_data.py --concurrency 5 --model sonnet
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evolution_agent.evaluation.sandbox import CodeSandbox, lint_code
from evolution_agent.mutation.prompts import (
    crossover_prompt,
    guided_mutation_prompt,
    point_mutation_prompt,
    structural_mutation_prompt,
)
from evolution_agent.mutation.strategies import _parse_mutation_response


# ---------------------------------------------------------------------------
# Problem Registry
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    domain: str
    name: str
    function_name: str
    spec: str
    seeds: list[str]
    seed_fitnesses: list[float] = field(default_factory=lambda: [0.3, 0.6, 0.85])


PROBLEMS = [
    # --- Sorting ---
    Problem(
        domain="sorting", name="sort_array", function_name="solve",
        spec="""\
def solve(arr, p=None):
    # Sort a list of numbers in ascending order.
    # Args: arr: list of integers/floats
    #        p: optional dict of tunable parameters
    # Returns: sorted list
    # Available: math, range, len, sorted, min, max, list, int, float, abs
    # Goal: minimize comparisons while producing correct sorted output.""",
        seeds=[
            '''\
def solve(arr, p=None):
    if p is None:
        p = {"threshold": 10}
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr''',
            '''\
def solve(arr, p=None):
    if p is None:
        p = {"pivot_strategy": 0.5}
    if len(arr) <= 1:
        return arr
    pivot = arr[int(len(arr) * p["pivot_strategy"])]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return solve(left, p) + middle + solve(right, p)''',
        ],
    ),
    # --- Search ---
    Problem(
        domain="search", name="binary_search", function_name="solve",
        spec="""\
def solve(arr, target, p=None):
    # Find target in a sorted array. Return index or -1 if not found.
    # Args: arr: sorted list of numbers, target: number to find
    #        p: optional dict of tunable parameters
    # Returns: index of target or -1
    # Available: math, range, len, int, float
    # Goal: minimize number of comparisons.""",
        seeds=[
            '''\
def solve(arr, target, p=None):
    if p is None:
        p = {"interpolation_weight": 0.0}
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1''',
            '''\
def solve(arr, target, p=None):
    if p is None:
        p = {"interpolation_weight": 0.0}
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1''',
        ],
    ),
    # --- String ---
    Problem(
        domain="string", name="edit_distance", function_name="solve",
        spec="""\
def solve(s1, s2, p=None):
    # Compute minimum edit distance (Levenshtein) between two strings.
    # Args: s1, s2: strings
    #        p: optional dict of tunable parameters (e.g. operation costs)
    # Returns: integer minimum edit distance
    # Available: math, range, len, min, max, int, list
    # Goal: correct edit distance with minimal time/space.""",
        seeds=[
            '''\
def solve(s1, s2, p=None):
    if p is None:
        p = {"insert_cost": 1, "delete_cost": 1, "replace_cost": 1}
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i * p["delete_cost"]
    for j in range(n + 1):
        dp[0][j] = j * p["insert_cost"]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + p["delete_cost"],
                    dp[i][j-1] + p["insert_cost"],
                    dp[i-1][j-1] + p["replace_cost"],
                )
    return dp[m][n]''',
        ],
    ),
    # --- Math optimization ---
    Problem(
        domain="optimization", name="gradient_descent", function_name="solve",
        spec="""\
def solve(f, grad_f, x0, p=None):
    # Minimize function f starting from x0 using gradient-based method.
    # Args: f: callable(x) -> float, grad_f: callable(x) -> float (derivative)
    #        x0: float starting point
    #        p: optional dict of tunable parameters
    # Returns: float x that approximately minimizes f
    # Available: math, range, len, abs, float, int
    # Goal: find minimum with fewest function evaluations.""",
        seeds=[
            '''\
def solve(f, grad_f, x0, p=None):
    if p is None:
        p = {"lr": 0.01, "max_iter": 100, "tol": 1e-6}
    x = x0
    for _ in range(int(p["max_iter"])):
        g = grad_f(x)
        if abs(g) < p["tol"]:
            break
        x = x - p["lr"] * g
    return x''',
            '''\
def solve(f, grad_f, x0, p=None):
    if p is None:
        p = {"lr": 0.1, "momentum": 0.9, "max_iter": 200, "tol": 1e-8}
    x = x0
    v = 0.0
    for _ in range(int(p["max_iter"])):
        g = grad_f(x)
        if abs(g) < p["tol"]:
            break
        v = p["momentum"] * v - p["lr"] * g
        x = x + v
    return x''',
        ],
    ),
    # --- Graph ---
    Problem(
        domain="graph", name="shortest_path", function_name="solve",
        spec="""\
def solve(graph, start, end, p=None):
    # Find shortest path in weighted graph from start to end.
    # Args: graph: dict {node: [(neighbor, weight), ...]}, start: node, end: node
    #        p: optional dict of tunable parameters
    # Returns: tuple (distance, path_as_list) or (float('inf'), []) if unreachable
    # Available: math, range, len, min, max, list, dict, set, float, int, tuple
    # Goal: correct shortest path with good performance.""",
        seeds=[
            '''\
def solve(graph, start, end, p=None):
    if p is None:
        p = {"max_iterations": 1000}
    dist = {start: 0}
    prev = {start: None}
    visited = set()
    for _ in range(int(p["max_iterations"])):
        u = None
        for node in dist:
            if node not in visited:
                if u is None or dist[node] < dist[u]:
                    u = node
        if u is None or u == end:
            break
        visited.add(u)
        for v, w in graph.get(u, []):
            alt = dist[u] + w
            if v not in dist or alt < dist[v]:
                dist[v] = alt
                prev[v] = u
    if end not in dist:
        return (float("inf"), [])
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev[node]
    return (dist[end], path[::-1])''',
        ],
    ),
    # --- Knapsack ---
    Problem(
        domain="combinatorial", name="knapsack", function_name="solve",
        spec="""\
def solve(items, capacity, p=None):
    # Solve 0/1 knapsack: maximize total value within weight capacity.
    # Args: items: list of (weight, value) tuples, capacity: int max weight
    #        p: optional dict of tunable parameters
    # Returns: tuple (total_value, selected_indices)
    # Available: math, range, len, max, min, list, int, float, sorted
    # Goal: maximize value, ideally optimal or near-optimal.""",
        seeds=[
            '''\
def solve(items, capacity, p=None):
    if p is None:
        p = {"greedy_ratio": 1.0}
    indexed = [(v / max(w, 1), w, v, i) for i, (w, v) in enumerate(items)]
    indexed.sort(reverse=True)
    total_w, total_v = 0, 0
    selected = []
    for ratio, w, v, i in indexed:
        if total_w + w <= capacity:
            total_w += w
            total_v += v
            selected.append(i)
    return (total_v, selected)''',
            '''\
def solve(items, capacity, p=None):
    if p is None:
        p = {"use_dp": 1}
    n = len(items)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = items[i - 1]
        for c in range(capacity + 1):
            dp[i][c] = dp[i-1][c]
            if w <= c and dp[i-1][c - w] + v > dp[i][c]:
                dp[i][c] = dp[i-1][c - w] + v
    selected = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i-1][c]:
            selected.append(i - 1)
            c -= items[i-1][0]
    return (dp[n][capacity], selected[::-1])''',
        ],
    ),
    # --- Numerical ---
    Problem(
        domain="numerical", name="numerical_integration", function_name="solve",
        spec="""\
def solve(f, a, b, p=None):
    # Numerically integrate f(x) from a to b.
    # Args: f: callable(x) -> float, a: float lower bound, b: float upper bound
    #        p: optional dict of tunable parameters
    # Returns: float approximate integral
    # Available: math, range, len, abs, float, int, sum
    # Goal: accurate integral estimate with minimal function evaluations.""",
        seeds=[
            '''\
def solve(f, a, b, p=None):
    if p is None:
        p = {"n_intervals": 100}
    n = int(p["n_intervals"])
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        total += f(a + i * h)
    return total * h''',
        ],
    ),
    # --- Data structures ---
    Problem(
        domain="data_structures", name="lru_operations", function_name="solve",
        spec="""\
def solve(operations, p=None):
    # Simulate an LRU cache processing a sequence of operations.
    # Args: operations: list of (op, key, value) tuples where op is 'get' or 'put'
    #        p: optional dict with cache parameters
    # Returns: list of results for 'get' operations (-1 if not found)
    # Available: math, range, len, list, dict, int
    # Goal: correct LRU behavior with efficient operations.""",
        seeds=[
            '''\
def solve(operations, p=None):
    if p is None:
        p = {"capacity": 10}
    cache = {}
    order = []
    results = []
    cap = int(p["capacity"])
    for op, key, value in operations:
        if op == "get":
            if key in cache:
                order.remove(key)
                order.append(key)
                results.append(cache[key])
            else:
                results.append(-1)
        elif op == "put":
            if key in cache:
                order.remove(key)
            elif len(cache) >= cap:
                oldest = order.pop(0)
                del cache[oldest]
            cache[key] = value
            order.append(key)
    return results''',
        ],
    ),
]

# Domain-agnostic guidance strings
GUIDANCE_LIBRARY = [
    "Add memoization or caching to avoid redundant computations.",
    "Use a different data structure — try a hash map instead of a list for lookups.",
    "Add early termination: stop as soon as the answer is found or provably optimal.",
    "Reduce time complexity — the current approach is O(n^2), try to make it O(n log n).",
    "Add adaptive step sizes or thresholds that adjust based on input characteristics.",
    "Use a divide-and-conquer approach instead of iterating over the full input.",
    "Add a greedy preprocessing step before the main algorithm.",
    "Handle edge cases: empty input, single element, already-sorted input.",
    "Use space-time tradeoff: precompute a lookup table for faster repeated access.",
    "Replace the brute-force search with a priority queue or heap.",
    "Add a local search improvement pass after the initial solution.",
    "Use bit manipulation or mathematical properties to speed up the computation.",
    "Implement iterative deepening instead of fixed-depth search.",
    "Add randomization: try multiple random starting points and keep the best.",
    "Use dynamic programming instead of recursion to avoid stack overflow.",
    "Reduce space usage from O(n^2) to O(n) by only keeping the previous row.",
    "Add parameter validation and handle degenerate inputs gracefully.",
    "Use sentinel values to simplify boundary condition checks.",
    "Implement a two-pointer technique to reduce nested loops.",
    "Add convergence detection: stop iterating when changes are below a threshold.",
]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

async def generate_one(
    llm_fn,
    problem: Problem,
    mutation_type: str,
    seed_idx: int,
    guidance: str = "",
    partner_idx: int | None = None,
    max_retries: int = 3,
) -> dict | None:
    """Generate one synthetic training example."""
    seed_code = problem.seeds[seed_idx]
    sandbox = CodeSandbox()

    # Build prompt
    if mutation_type == "point":
        system, user = point_mutation_prompt(seed_code, problem.spec, guidance)
    elif mutation_type == "structural":
        system, user = structural_mutation_prompt(seed_code, problem.spec, guidance)
    elif mutation_type == "guided" and guidance:
        system, user = guided_mutation_prompt(seed_code, problem.spec, guidance)
    elif mutation_type == "crossover" and partner_idx is not None:
        partner_code = problem.seeds[partner_idx]
        system, user = crossover_prompt(
            seed_code, partner_code, problem.spec,
            problem.seed_fitnesses[seed_idx],
            problem.seed_fitnesses[partner_idx],
        )
    else:
        system, user = structural_mutation_prompt(seed_code, problem.spec, guidance)

    # Add anti-import instruction
    user += "\n\nIMPORTANT: Do NOT use import statements. Available modules: math, deque, Counter, defaultdict."

    for attempt in range(max_retries):
        try:
            raw = await llm_fn(system, user)
            code, desc = _parse_mutation_response(raw)

            if not code:
                continue

            # Validate
            errors = lint_code(code)
            if errors:
                continue

            # Check function exists
            fn = sandbox.compile_function(code, problem.function_name)
            if fn is None:
                continue

            # Check not identical to parent
            parent_hash = hashlib.sha256(seed_code.strip().encode()).hexdigest()[:16]
            child_hash = hashlib.sha256(code.strip().encode()).hexdigest()[:16]
            if parent_hash == child_hash:
                continue

            # Build training example
            response = f"DESCRIPTION: {desc}\n\n```python\n{code}\n```"

            return {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": response},
                ],
                "reward": 0.8,
                "metadata": {
                    "domain": problem.domain,
                    "problem": problem.name,
                    "mutation_type": mutation_type,
                    "synthetic": True,
                    "has_guidance": bool(guidance),
                },
            }
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  FAILED {problem.name}/{mutation_type}: {e}")

    return None


async def generate_all(
    llm_fn,
    output_path: Path,
    concurrency: int = 5,
    dry_run: bool = False,
):
    """Generate all synthetic training examples."""
    import random
    random.seed(42)

    tasks = []

    for problem in PROBLEMS:
        for seed_idx in range(len(problem.seeds)):
            # Point mutations (3 per seed)
            for _ in range(3):
                tasks.append((problem, "point", seed_idx, "", None))

            # Structural mutations (2 per seed)
            for _ in range(2):
                tasks.append((problem, "structural", seed_idx, "", None))

            # Guided mutations (2 per seed, random guidance)
            for _ in range(2):
                guidance = random.choice(GUIDANCE_LIBRARY)
                tasks.append((problem, "guided", seed_idx, guidance, None))

        # Crossover (1 per pair of seeds)
        if len(problem.seeds) >= 2:
            tasks.append((problem, "crossover", 0, "", 1))

    print(f"Total tasks: {len(tasks)}")

    if dry_run:
        for problem, mtype, seed_idx, guidance, partner in tasks[:5]:
            print(f"  {problem.domain}/{problem.name} [{mtype}] seed={seed_idx} guidance={guidance[:50] if guidance else 'none'}")
        print(f"  ... and {len(tasks) - 5} more")
        return

    sem = asyncio.Semaphore(concurrency)
    results = []
    done = 0

    async def run_one(task):
        nonlocal done
        problem, mtype, seed_idx, guidance, partner = task
        async with sem:
            result = await generate_one(
                llm_fn, problem, mtype, seed_idx, guidance, partner,
            )
            done += 1
            if done % 10 == 0:
                print(f"  Progress: {done}/{len(tasks)} ({len(results)} valid)")
            if result:
                results.append(result)

    await asyncio.gather(*[run_one(t) for t in tasks])

    # Deduplicate by code hash
    seen = set()
    unique = []
    for r in results:
        code = r["messages"][2]["content"]
        h = hashlib.sha256(code.encode()).hexdigest()[:16]
        if h not in seen:
            seen.add(h)
            unique.append(r)

    print(f"\nGenerated: {len(results)}, unique: {len(unique)}")

    # Write
    with open(output_path, "w") as f:
        for item in unique:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Written to {output_path}")

    # Stats
    by_domain = {}
    by_type = {}
    for r in unique:
        d = r["metadata"]["domain"]
        t = r["metadata"]["mutation_type"]
        by_domain[d] = by_domain.get(d, 0) + 1
        by_type[t] = by_type.get(t, 0) + 1

    print("\nBy domain:")
    for d, c in sorted(by_domain.items()):
        print(f"  {d}: {c}")
    print("By mutation type:")
    for t, c in sorted(by_type.items()):
        print(f"  {t}: {c}")


def make_llm_fn(model: str):
    """Create an async LLM function using claude-code CLI."""
    from evolution_agent.llm.claude_code_client import ClaudeCodeClient

    client = ClaudeCodeClient(model=model, timeout_s=60.0)

    async def llm_fn(system: str, user: str) -> str:
        return await client.complete(system, [{"role": "user", "content": user}])

    return llm_fn


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic mutation training data")
    parser.add_argument("--output", type=Path, default=Path("synthetic_data.jsonl"))
    parser.add_argument("--model", default="haiku", help="Claude model for generation (haiku/sonnet)")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    llm_fn = make_llm_fn(args.model)

    asyncio.run(generate_all(
        llm_fn, args.output,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
