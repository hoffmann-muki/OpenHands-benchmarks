"""Validate the tracked design inputs for the single-agent 5x1 experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FRAMEWORKS = ("opencode", "openhands", "hermes")
BENCHMARKS = {
    "swe-bench-verified": "princeton-nlp/SWE-bench_Verified",
    "terminal-bench-2.1": "terminal-bench/terminal-bench-2-1",
}
REQUIRED_CONFIGURATION = {
    "model": "openrouter/poolside/laguna-s-2.1:free",
    "temperature": 0.1,
    "agent_topology": "single-agent",
    "delegation_enabled": False,
    "agent_budget": 24,
    "agent_timeout_seconds": 900,
    "provider_attempts_per_turn": 1,
    "benchmark_attempts_per_instance": 1,
    "inference_workers": 1,
    "evaluation_workers": 1,
    "swe_evaluation_timeout_seconds": 3600,
    "infrastructure_retries": 0,
    "semantic_tracing": True,
    "agentsight_profiling": True,
    "local_docker_evaluation": True,
    "harbor_version": "0.20.0",
}


def main() -> None:
    manifest = load_manifest()
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest.schema_version must be 1")

    design = require_mapping(manifest.get("design"), "design")
    frameworks = require_string_list(design.get("frameworks"), "design.frameworks")
    if frameworks != list(FRAMEWORKS):
        raise ValueError("design.frameworks must contain the three canonical harnesses")

    repetitions = require_positive_int(
        design.get("repetitions_per_instance"),
        "design.repetitions_per_instance",
    )
    instances_per_benchmark = require_positive_int(
        design.get("instances_per_benchmark"),
        "design.instances_per_benchmark",
    )
    if repetitions != 1 or instances_per_benchmark != 5:
        raise ValueError("the paired design must use five instances and one run each")

    expected_runs = (
        len(frameworks) * len(BENCHMARKS) * repetitions * instances_per_benchmark
    )
    if design.get("expected_agent_runs") != expected_runs:
        raise ValueError(
            "design.expected_agent_runs does not match the experiment dimensions"
        )

    configuration = require_mapping(manifest.get("configuration"), "configuration")
    if configuration != REQUIRED_CONFIGURATION:
        raise ValueError("configuration differs from the paired single-agent policy")

    benchmarks = require_mapping(manifest.get("benchmarks"), "benchmarks")
    if set(benchmarks) != set(BENCHMARKS):
        raise ValueError("benchmarks must contain only Verified and Terminal-Bench 2.1")
    all_ids: set[str] = set()
    for benchmark, dataset in BENCHMARKS.items():
        definition = require_mapping(
            benchmarks.get(benchmark), f"benchmarks.{benchmark}"
        )
        if definition.get("dataset") != dataset:
            raise ValueError(f"{benchmark} has the wrong canonical dataset")
        manifest_ids = require_string_list(
            definition.get("instances"),
            f"benchmarks.{benchmark}.instances",
        )
        selection_ids = [
            line.strip()
            for line in (ROOT / f"{benchmark}.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if len(manifest_ids) != instances_per_benchmark:
            raise ValueError(
                f"{benchmark} does not contain {instances_per_benchmark} instances"
            )
        if len(set(manifest_ids)) != len(manifest_ids):
            raise ValueError(f"{benchmark} contains duplicate instance IDs")
        if selection_ids != manifest_ids:
            raise ValueError(f"{benchmark}.txt differs from manifest.json")
        overlap = all_ids.intersection(manifest_ids)
        if overlap:
            raise ValueError(
                f"instance IDs overlap across benchmarks: {sorted(overlap)}"
            )
        all_ids.update(manifest_ids)

    print(f"valid experiment: {expected_runs} agent runs")


def load_manifest() -> dict[str, Any]:
    value = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    return require_mapping(value, "manifest")


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def require_string_list(value: Any, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"{name} must be a non-empty string array")
    return value


def require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


if __name__ == "__main__":
    main()
