"""Wrapper for the BLIS inference simulator CLI."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BLISConfig:
    """Configuration for the BLIS simulator binary."""

    binary_path: str = "./blis"
    model: str = "meta-llama/llama-3.1-8b-instruct"
    num_requests: int = 100
    seed: int = 42
    hardware: str = ""
    tp: int = 0
    latency_model: str = ""
    workload_spec: str = ""
    timeout_seconds: int = 120
    base_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class BLISResult:
    """Parsed results from a BLIS simulation run."""

    completed_requests: int = 0
    tokens_per_sec: float = 0.0
    ttft_mean_ms: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p90_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    e2e_mean_ms: float = 0.0
    e2e_p50_ms: float = 0.0
    e2e_p90_ms: float = 0.0
    e2e_p95_ms: float = 0.0
    e2e_p99_ms: float = 0.0
    itl_mean_ms: float = 0.0
    itl_p99_ms: float = 0.0
    raw_json: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str = ""


_BLIS_VALID_FLAGS = {
    "admission_latency", "admission_policy", "alpha_coeffs", "beta_coeffs",
    "block_size_in_tokens", "counterfactual_k", "fitness_weights",
    "hardware", "hardware_config", "horizon", "kv_cpu_blocks",
    "kv_offload_threshold", "kv_transfer_bandwidth", "kv_transfer_base_latency",
    "latency_model", "log", "long_prefill_token_threshold",
    "max_num_running_reqs", "max_num_scheduled_tokens", "model",
    "model_config_folder", "num_instances", "num_requests",
    "output_tokens", "output_tokens_max", "output_tokens_min", "output_tokens_stdev",
    "policy_config", "prefix_tokens", "priority_policy",
    "prompt_tokens", "prompt_tokens_max", "prompt_tokens_min", "prompt_tokens_stdev",
    "rate", "results_path", "routing_latency", "routing_policy", "routing_scorers",
    "scheduler", "seed", "snapshot_refresh_interval", "token_bucket_capacity",
    "token_bucket_refill_rate", "total_kv_blocks", "tp", "workload", "workload_spec",
    "external_policy_address",
}

_BLIS_INT_FLAGS = {
    "admission_latency", "block_size_in_tokens", "counterfactual_k",
    "horizon", "kv_cpu_blocks", "kv_transfer_base_latency",
    "long_prefill_token_threshold", "max_num_running_reqs",
    "max_num_scheduled_tokens", "num_instances", "num_requests",
    "output_tokens", "output_tokens_max", "output_tokens_min", "output_tokens_stdev",
    "prefix_tokens", "prompt_tokens", "prompt_tokens_max", "prompt_tokens_min",
    "prompt_tokens_stdev", "routing_latency", "seed", "snapshot_refresh_interval",
    "total_kv_blocks", "tp",
}


class BLISRunner:
    """Runs the BLIS inference simulator as a subprocess."""

    def __init__(self, config: BLISConfig):
        self.config = config
        self._validate_binary()

    def _validate_binary(self) -> None:
        path = Path(self.config.binary_path)
        if not path.exists():
            raise FileNotFoundError(
                f"BLIS binary not found at {self.config.binary_path}. "
                f"Build it with: cd inference-sim && go build -o blis main.go"
            )

    def run(self, params: dict[str, Any]) -> BLISResult:
        """Execute BLIS with the given parameter overrides.

        Args:
            params: CLI flag overrides from the optimizer.
                    Keys use underscores (e.g., num_instances), converted
                    to CLI flags (--num-instances).
        """
        cmd = self._build_command(params)
        logger.info("BLIS run: %s", " ".join(cmd))

        try:
            # Run from the BLIS binary's directory so it finds defaults.yaml
            cwd = str(Path(self.config.binary_path).resolve().parent)
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired:
            return BLISResult(
                success=False,
                error=f"BLIS timed out after {self.config.timeout_seconds}s",
            )
        except FileNotFoundError:
            return BLISResult(
                success=False,
                error=f"BLIS binary not found: {self.config.binary_path}",
            )

        if proc.returncode != 0:
            return BLISResult(
                success=False,
                error=f"BLIS exit code {proc.returncode}: {proc.stderr[:500]}",
            )

        return self._parse_output(proc.stdout)

    def _build_command(self, params: dict[str, Any]) -> list[str]:
        cmd = [
            str(Path(self.config.binary_path).resolve()), "run",
            "--model", self.config.model,
            "--seed", str(self.config.seed),
        ]
        if self.config.num_requests > 0:
            cmd.extend(["--num-requests", str(self.config.num_requests)])

        if self.config.hardware:
            cmd.extend(["--hardware", self.config.hardware])
        if self.config.tp > 0:
            cmd.extend(["--tp", str(self.config.tp)])
        if self.config.latency_model:
            cmd.extend(["--latency-model", self.config.latency_model])
        if self.config.workload_spec:
            cmd.extend(["--workload-spec", self.config.workload_spec])

        # Base args (fixed, not optimized)
        for flag, value in self.config.base_args.items():
            cli_flag = f"--{flag.replace('_', '-')}"
            cmd.extend([cli_flag, str(value)])

        # Optimizer parameter overrides (filter invalid, round integers)
        for flag, value in params.items():
            if flag not in _BLIS_VALID_FLAGS:
                logger.warning("Skipping unknown BLIS flag: %s", flag)
                continue
            cli_flag = f"--{flag.replace('_', '-')}"
            if flag in _BLIS_INT_FLAGS:
                cmd.extend([cli_flag, str(max(1, int(round(float(value)))))])
            elif isinstance(value, float):
                cmd.extend([cli_flag, f"{value:.4f}"])
            else:
                cmd.extend([cli_flag, str(value)])

        return cmd

    def _parse_output(self, stdout: str) -> BLISResult:
        json_str = self._extract_json(stdout)
        if json_str is None:
            return BLISResult(
                success=False,
                error=f"No JSON found in BLIS output: {stdout[:500]}",
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return BLISResult(
                success=False,
                error=f"JSON parse error: {e}",
            )

        return BLISResult(
            completed_requests=data.get("completed_requests", 0),
            tokens_per_sec=data.get("tokens_per_sec", 0.0),
            ttft_mean_ms=data.get("ttft_mean_ms", 0.0),
            ttft_p50_ms=data.get("ttft_p50_ms", 0.0),
            ttft_p90_ms=data.get("ttft_p90_ms", 0.0),
            ttft_p95_ms=data.get("ttft_p95_ms", 0.0),
            ttft_p99_ms=data.get("ttft_p99_ms", 0.0),
            e2e_mean_ms=data.get("e2e_mean_ms", 0.0),
            e2e_p50_ms=data.get("e2e_p50_ms", 0.0),
            e2e_p90_ms=data.get("e2e_p90_ms", 0.0),
            e2e_p95_ms=data.get("e2e_p95_ms", 0.0),
            e2e_p99_ms=data.get("e2e_p99_ms", 0.0),
            itl_mean_ms=data.get("itl_mean_ms", 0.0),
            itl_p99_ms=data.get("itl_p99_ms", 0.0),
            raw_json=data,
            success=True,
        )

    def run_with_external_policy(
        self,
        params: dict[str, Any],
        policy_server: Any,  # BLISPolicyServer — import avoided for circular deps
    ) -> BLISResult:
        """Run BLIS using an external policy controller for routing decisions.

        The policy_server must already be started (call server.start() first).
        BLIS will connect to the policy server over TCP and request a routing
        decision for every incoming request.

        The simulation clock does NOT advance while waiting for Python's routing
        decision, so Python's wall-clock latency does not affect simulated metrics.

        Args:
            params: CLI flag overrides (same as run()). routing_policy and
                    external_policy_address are set automatically.
            policy_server: A started BLISPolicyServer instance.

        Returns:
            BLISResult with simulation metrics.
        """
        merged_params = dict(params)
        merged_params["routing_policy"] = "external"
        merged_params["external_policy_address"] = f"localhost:{policy_server.port}"
        return self.run(merged_params)

    @staticmethod
    def _extract_json(stdout: str) -> str | None:
        """Extract the cluster-aggregate JSON from BLIS stdout.

        BLIS outputs multiple JSON blocks: one per instance + one cluster
        aggregate (the last block, with ``instance_id: "cluster"``).
        We want the cluster aggregate, so we return the **last** complete
        JSON object found in the output.
        """
        last_json: str | None = None
        search_start = 0
        while True:
            pos = stdout.find("{", search_start)
            if pos < 0:
                break
            depth = 0
            for i, ch in enumerate(stdout[pos:], pos):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        last_json = stdout[pos : i + 1]
                        search_start = i + 1
                        break
            else:
                break  # unclosed brace — stop
        return last_json
