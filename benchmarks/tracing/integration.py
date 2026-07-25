"""Benchmark-agnostic trace run coordination and harness integration."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from benchmarks.tracing.constants import CONTRACT_VERSION
from benchmarks.tracing.models import JsonObject, JsonValue
from benchmarks.tracing.recorder import (
    ContractValidator,
    write_run_index,
)
from benchmarks.tracing.storage import read_jsonl, utc_now


TraceSelectionStrategy = Literal[
    "explicit_ids",
    "full_dataset",
    "ordered_window",
]


@dataclass(frozen=True, slots=True)
class TraceRun:
    """Private trace root shared by one benchmark invocation."""

    id: str
    root: Path
    created_at: str
    benchmark: str
    framework: str


@dataclass(frozen=True, slots=True)
class TraceSelection:
    """Resolved benchmark selection and minimum trace coverage."""

    instance_ids: tuple[str, ...]
    strategy: TraceSelectionStrategy
    minimum_attempts_per_instance: int = 1

    def __post_init__(self) -> None:
        if not self.instance_ids:
            raise ValueError("Trace selection cannot be empty")
        if any(not instance_id.strip() for instance_id in self.instance_ids):
            raise ValueError("Trace selection instance IDs cannot be empty")
        if len(set(self.instance_ids)) != len(self.instance_ids):
            raise ValueError("Trace selection instance IDs must be unique")
        if (
            isinstance(self.minimum_attempts_per_instance, bool)
            or not isinstance(self.minimum_attempts_per_instance, int)
            or self.minimum_attempts_per_instance < 1
        ):
            raise ValueError("Trace selection requires at least one attempt")


class TraceHarnessAdapter(Protocol):
    """Harness-owned selection and finalization behavior.

    Agent-framework adapters own native agent events. Harness adapters only
    expose execution topology that the benchmark runner cannot provide
    directly, such as a resolved task order or infrastructure retry count.
    """

    def prepare_finalization(self, run: TraceRun) -> None:
        """Finish harness-owned bookkeeping before attempt discovery."""

    def resolve_selection(
        self,
        run: TraceRun,
        observed_instance_ids: Sequence[str],
    ) -> TraceSelection:
        """Return the canonical instance order and required attempt coverage."""
        ...


@dataclass(frozen=True, slots=True)
class DirectTraceHarness:
    """Selection adapter for benchmarks whose runner owns the instance list."""

    selection: TraceSelection

    def prepare_finalization(self, run: TraceRun) -> None:
        del run

    def resolve_selection(
        self,
        run: TraceRun,
        observed_instance_ids: Sequence[str],
    ) -> TraceSelection:
        del run, observed_instance_ids
        return self.selection


def create_trace_run(
    base_directory: Path,
    *,
    benchmark: str,
    framework: str,
) -> TraceRun:
    """Create a private run root before any provider or workspace work."""

    if not benchmark.strip() or not framework.strip():
        raise ValueError("Trace benchmark and framework cannot be empty")
    expanded = base_directory.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"Trace base cannot be a symbolic link: {expanded}")
    base = expanded.resolve()
    existed = base.exists()
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not base.is_dir() or base.is_symlink():
        raise ValueError(f"Trace base must be a real directory: {base}")
    if not existed and os.name != "nt":
        base.chmod(0o700)
    run_id = f"trace-run-{uuid.uuid4().hex}"
    root = base / run_id
    root.mkdir(mode=0o700)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Trace root must be a real directory: {root}")
    if os.name != "nt":
        root.chmod(0o700)
    return TraceRun(
        id=run_id,
        root=root,
        created_at=utc_now(),
        benchmark=benchmark,
        framework=framework,
    )


def attach_trace_run(
    *,
    root: Path,
    run_id: str,
    created_at: str,
    benchmark: str,
    framework: str,
) -> TraceRun:
    """Reconstruct a run handle in a worker or evaluation coordinator."""

    resolved = root.resolve()
    if (
        not run_id
        or not created_at
        or not benchmark.strip()
        or not framework.strip()
        or not resolved.is_dir()
        or root.is_symlink()
    ):
        raise ValueError("Existing trace run identity or root is invalid")
    return TraceRun(
        id=run_id,
        root=resolved,
        created_at=created_at,
        benchmark=benchmark,
        framework=framework,
    )


def finalize_trace_run(
    run: TraceRun,
    harness: TraceHarnessAdapter,
) -> Path:
    """Validate every discovered attempt and write the canonical run index."""

    if run.root.is_symlink() or not run.root.resolve().is_dir():
        raise ValueError("Trace run root must be a real directory")
    harness.prepare_finalization(run)
    validator = ContractValidator()
    attempts: list[JsonObject] = []
    observed: dict[str, set[int]] = {}

    for manifest_path in sorted(run.root.glob("instances/*/attempt-*/manifest.json")):
        attempt_dir = manifest_path.parent
        report = validator.validate_attempt(attempt_dir)
        if not report.valid:
            raise ValueError(f"Trace attempt failed validation: {attempt_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"Trace manifest must be an object: {manifest_path}")
        instance_id = manifest.get("instance_id")
        attempt_number = manifest.get("attempt")
        if (
            manifest.get("run_id") != run.id
            or manifest.get("benchmark") != run.benchmark
            or manifest.get("framework") != run.framework
            or not isinstance(instance_id, str)
            or not instance_id.strip()
            or isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or attempt_number < 1
        ):
            raise ValueError(f"Trace attempt does not belong to its run: {attempt_dir}")
        if attempt_number in observed.setdefault(instance_id, set()):
            raise ValueError(
                f"Duplicate trace attempt {attempt_number} for {instance_id}"
            )

        events = read_jsonl(
            attempt_dir / "events.jsonl",
            allow_torn_final_line=False,
        ).records
        terminal_status = next(
            (
                event["status"]
                for event in reversed(events)
                if event["event_type"] in {"attempt.end", "instance.end"}
            ),
            "degraded",
        )
        observed[instance_id].add(attempt_number)
        attempts.append(
            {
                "trace_id": str(manifest["trace_id"]),
                "instance_id": instance_id,
                "attempt": attempt_number,
                "path": attempt_dir.relative_to(run.root).as_posix(),
                "status": str(terminal_status),
            }
        )

    selection = harness.resolve_selection(run, tuple(observed))
    if set(selection.instance_ids) != set(observed):
        raise ValueError(
            "Trace run index omitted because selected instances lack finalized traces"
        )
    if any(
        len(observed[instance_id]) < selection.minimum_attempts_per_instance
        for instance_id in selection.instance_ids
    ):
        raise ValueError(
            "Trace run index omitted because an instance lacks a requested attempt"
        )

    order = {
        instance_id: index for index, instance_id in enumerate(selection.instance_ids)
    }

    def attempt_order(value: JsonObject) -> tuple[int, int]:
        instance_id = value.get("instance_id")
        attempt_number = value.get("attempt")
        if not isinstance(instance_id, str) or not isinstance(attempt_number, int):
            raise ValueError("Validated trace attempt identity is invalid")
        return order[instance_id], attempt_number

    attempts.sort(key=attempt_order)
    selected_values: list[JsonValue] = [
        instance_id for instance_id in selection.instance_ids
    ]
    attempt_values: list[JsonValue] = [attempt for attempt in attempts]
    document: JsonObject = {
        "schema_version": "benchmark-trace/v1",
        "contract": {
            "name": "benchmark-trace",
            "version": CONTRACT_VERSION,
            "schema_digest": validator.schema_digest,
        },
        "run_id": run.id,
        "benchmark": run.benchmark,
        "framework": run.framework,
        "created_at": run.created_at,
        "finalized_at": utc_now(),
        "selection": {
            "strategy": selection.strategy,
            "requested_count": len(selection.instance_ids),
            "instance_ids": selected_values,
        },
        "attempts": attempt_values,
    }
    return write_run_index(run.root, document, validator=validator)
