"""Validate the tracked design inputs for the single-agent 10x3 experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BENCHMARKS = ("swe-bench-lite", "swe-bench-verified")


def main() -> None:
    manifest = load_manifest()
    design = require_mapping(manifest.get("design"), "design")
    frameworks = require_string_list(design.get("frameworks"), "design.frameworks")
    repetitions = require_positive_int(
        design.get("repetitions_per_instance"),
        "design.repetitions_per_instance",
    )
    instances_per_benchmark = require_positive_int(
        design.get("instances_per_benchmark"),
        "design.instances_per_benchmark",
    )
    expected_runs = (
        len(frameworks) * len(BENCHMARKS) * repetitions * instances_per_benchmark
    )
    if design.get("expected_agent_runs") != expected_runs:
        raise ValueError(
            "design.expected_agent_runs does not match the experiment dimensions"
        )

    benchmarks = require_mapping(manifest.get("benchmarks"), "benchmarks")
    for benchmark in BENCHMARKS:
        definition = require_mapping(
            benchmarks.get(benchmark), f"benchmarks.{benchmark}"
        )
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
