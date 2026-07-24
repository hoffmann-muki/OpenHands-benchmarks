"""Version-aware, read-only analysis of normalized benchmark traces."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from benchmarks.tracing.constants import (
    CONTRACT_NAME,
    CONTRACT_VERSION,
    SCHEMA_VERSION,
)
from benchmarks.tracing.errors import TraceValidationError
from benchmarks.tracing.models import JsonObject, JsonValue
from benchmarks.tracing.storage import read_jsonl
from benchmarks.tracing.timeline import build_timeline
from benchmarks.tracing.validation import (
    ContractValidator,
    ValidationIssue,
    ValidationReport,
)


TraceTargetKind = Literal["run", "attempt"]


@dataclass(frozen=True, slots=True)
class TraceTarget:
    """A resolved v1 run or attempt directory."""

    kind: TraceTargetKind
    path: Path
    document: JsonObject


def discover_trace_targets(path: Path) -> tuple[TraceTarget, ...]:
    """Resolve one trace or discover finalized runs beneath a trace base."""

    candidate = path.expanduser().resolve()
    direct = _resolve_direct_target(candidate)
    if direct is not None:
        return (direct,)
    if not candidate.is_dir() or candidate.is_symlink():
        raise TraceValidationError(f"Trace path is not a real directory: {candidate}")

    documents = sorted(
        {
            *candidate.glob("*/run.json"),
            *candidate.glob("*/*/run.json"),
        }
    )
    targets = tuple(
        target
        for document in documents
        if (target := _resolve_direct_target(document)) is not None
    )
    if not targets:
        raise TraceValidationError(
            f"No finalized benchmark-trace runs were found beneath: {candidate}"
        )
    return targets


def resolve_trace_target(path: Path) -> TraceTarget:
    """Resolve exactly one finalized trace run or attempt."""

    targets = discover_trace_targets(path)
    if len(targets) != 1:
        raise TraceValidationError(
            "Trace path resolves to multiple runs; select one run or use a "
            "multi-path command"
        )
    return targets[0]


def validate_trace_target(
    target: TraceTarget,
    *,
    validator: ContractValidator | None = None,
) -> ValidationReport:
    active = validator or ContractValidator()
    if target.kind == "run":
        return active.validate_run(target.path)
    return active.validate_attempt(target.path)


def validation_as_json(
    target: TraceTarget,
    *,
    validator: ContractValidator | None = None,
) -> JsonObject:
    report = validate_trace_target(target, validator=validator)
    return {
        "path": str(target.path),
        "kind": target.kind,
        "schema_version": SCHEMA_VERSION,
        "valid": report.valid,
        "issues": [_issue_as_json(issue) for issue in report.issues],
    }


def inspect_trace(target: TraceTarget) -> JsonObject:
    """Return identity, provenance, configuration, health, and capabilities."""

    _require_valid(target)
    if target.kind == "attempt":
        return _inspect_attempt(target.path)

    manifests: list[JsonObject] = []
    health_documents: list[JsonObject] = []
    capability_documents: list[JsonObject] = []
    attempts = []
    for attempt_dir in _attempt_directories(target):
        manifest = _load_object(attempt_dir / "manifest.json")
        health = _load_object(attempt_dir / "health.json")
        capabilities = _load_object(attempt_dir / "capabilities.json")
        manifests.append(manifest)
        health_documents.append(health)
        capability_documents.append(capabilities)
        attempts.append(
            {
                "trace_id": _string(manifest, "trace_id"),
                "instance_id": _string(manifest, "instance_id"),
                "attempt": _integer(manifest, "attempt"),
                "path": str(attempt_dir),
                "complete": _boolean(manifest, "complete"),
                "health": _string(health, "status"),
                "finalization": _string(health, "finalization"),
            }
        )
    producers = _unique_objects(
        [_object(manifest, "producer") for manifest in manifests]
    )
    provenance = _unique_objects(
        [_object(manifest, "provenance") for manifest in manifests]
    )
    executions = _unique_objects(
        [_object(manifest, "execution") for manifest in manifests]
    )
    return {
        "kind": "run",
        "path": str(target.path),
        "schema_version": SCHEMA_VERSION,
        "contract": target.document["contract"],
        "identity": {
            "run_id": _string(target.document, "run_id"),
            "benchmark": _string(target.document, "benchmark"),
            "framework": _string(target.document, "framework"),
        },
        "lifecycle": {
            "created_at": _string(target.document, "created_at"),
            "finalized_at": _string(target.document, "finalized_at"),
        },
        "selection": _object(target.document, "selection"),
        "producer": {
            "consistent": len(producers) == 1,
            "variants": producers,
        },
        "provenance": {
            "consistent": len(provenance) == 1,
            "variants": provenance,
        },
        "execution": {
            "consistent": len(executions) == 1,
            "variants": executions,
        },
        "health": {
            "status": _counter(
                _string(health, "status") for health in health_documents
            ),
            "finalization": _counter(
                _string(health, "finalization") for health in health_documents
            ),
        },
        "capabilities": _summarize_capabilities(capability_documents),
        "attempts": attempts,
    }


def summarize_trace(target: TraceTarget) -> JsonObject:
    """Aggregate normalized activity without reading artifact contents."""

    _require_valid(target)
    attempts = _attempt_directories(target)
    manifests = [_load_object(path / "manifest.json") for path in attempts]
    health_documents = [_load_object(path / "health.json") for path in attempts]
    capability_documents = [
        _load_object(path / "capabilities.json") for path in attempts
    ]
    event_streams = [
        read_jsonl(path / "events.jsonl", allow_torn_final_line=False).records
        for path in attempts
    ]
    native_streams = [
        read_jsonl(path / "native" / "index.jsonl", allow_torn_final_line=False).records
        for path in attempts
    ]
    events = tuple(event for stream in event_streams for event in stream)
    family_counts = Counter(_string(event, "event_family") for event in events)
    type_counts = Counter(_string(event, "event_type") for event in events)
    status_counts = Counter(_string(event, "status") for event in events)
    durations = [
        duration for event in events if (duration := _event_duration(event)) is not None
    ]
    durations_by_family: dict[str, list[float]] = {}
    for event in events:
        duration = _event_duration(event)
        if duration is None:
            continue
        durations_by_family.setdefault(_string(event, "event_family"), []).append(
            duration
        )

    counters = [_object(health, "counters") for health in health_documents]
    executions = _unique_objects(
        [_object(manifest, "execution") for manifest in manifests]
    )
    instance_ids = tuple(
        dict.fromkeys(_string(manifest, "instance_id") for manifest in manifests)
    )
    selected: list[JsonValue]
    if target.kind == "run":
        selection = _object(target.document, "selection")
        selected = _string_list(selection, "instance_ids")
        run_id = _string(target.document, "run_id")
        benchmark = _string(target.document, "benchmark")
        framework = _string(target.document, "framework")
    else:
        selected = [cast(JsonValue, instance_id) for instance_id in instance_ids]
        run_id = _string(manifests[0], "run_id")
        benchmark = _string(manifests[0], "benchmark")
        framework = _string(manifests[0], "framework")

    identity: JsonObject = {
        "run_id": run_id,
        "benchmark": benchmark,
        "framework": framework,
        "instance_ids": selected,
    }
    coverage: JsonObject = {
        "instances": len(instance_ids),
        "attempts": len(attempts),
        "complete_attempts": sum(
            1 for manifest in manifests if _boolean(manifest, "complete")
        ),
        "health": _counter(_string(health, "status") for health in health_documents),
        "finalization": _counter(
            _string(health, "finalization") for health in health_documents
        ),
    }
    execution: JsonObject = {
        "consistent": len(executions) == 1,
        "variants": executions,
    }
    event_summary: JsonObject = {
        "total": len(events),
        "native_records": sum(len(stream) for stream in native_streams),
        "families": _counter_json(family_counts),
        "types": _counter_json(type_counts),
        "statuses": _counter_json(status_counts),
    }
    activity: JsonObject = {
        "model_turns": _model_turn_count(events),
        "tool_calls": _activity_span_count(
            events,
            {"tool", "shell", "file", "search", "browser"},
        ),
        "delegations": _activity_span_count(events, {"delegation"}),
        "compactions": _activity_span_count(
            tuple(
                event
                for event in events
                if "compaction" in _string(event, "event_type")
            ),
            {"context"},
        ),
        "trace_issues": type_counts["trace.issue"],
        "failed_or_degraded_events": sum(
            status_counts[status] for status in ("failed", "degraded")
        ),
    }
    timing_by_family: JsonObject = {
        family: _duration_statistics(values)
        for family, values in sorted(durations_by_family.items())
    }
    timing: JsonObject = {
        "recorded_durations_ms": _duration_statistics(durations),
        "by_family": timing_by_family,
    }
    storage: JsonObject = {
        "artifacts": sum(_integer(value, "artifacts_written") for value in counters),
        "artifact_bytes": sum(
            _integer(value, "artifact_bytes_written") for value in counters
        ),
        "redactions": sum(_integer(value, "redactions_applied") for value in counters),
        "dropped_events": sum(_integer(value, "dropped_events") for value in counters),
        "sequence_gaps": sum(_integer(value, "sequence_gaps") for value in counters),
    }
    return {
        "path": str(target.path),
        "kind": target.kind,
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "identity": identity,
        "coverage": coverage,
        "execution": execution,
        "events": event_summary,
        "activity": activity,
        "timing": timing,
        "storage": storage,
        "capabilities": _summarize_capabilities(capability_documents),
    }


def compare_traces(targets: tuple[TraceTarget, ...]) -> JsonObject:
    """Compare normalized summaries and state explicit comparability checks."""

    if len(targets) < 2:
        raise TraceValidationError("Trace comparison requires at least two traces")
    summaries = [summarize_trace(target) for target in targets]
    identities = [_object(summary, "identity") for summary in summaries]
    executions = [_object(summary, "execution") for summary in summaries]
    execution_variants = [_array(execution, "variants") for execution in executions]
    checks: JsonObject = {
        "schema_version": _same(
            [_string(summary, "schema_version") for summary in summaries]
        ),
        "contract_version": _same(
            [_string(summary, "contract_version") for summary in summaries]
        ),
        "benchmark": _same([_string(identity, "benchmark") for identity in identities]),
        "instance_selection": _same(
            [_array(identity, "instance_ids") for identity in identities]
        ),
        "execution_consistent": all(
            execution.get("consistent") is True for execution in executions
        ),
        "execution": _same(execution_variants),
    }
    rows: list[JsonValue] = []
    for summary in summaries:
        identity = _object(summary, "identity")
        coverage = _object(summary, "coverage")
        events = _object(summary, "events")
        activity = _object(summary, "activity")
        storage = _object(summary, "storage")
        rows.append(
            {
                "path": _string(summary, "path"),
                "benchmark": _string(identity, "benchmark"),
                "framework": _string(identity, "framework"),
                "instances": _integer(coverage, "instances"),
                "attempts": _integer(coverage, "attempts"),
                "events": _integer(events, "total"),
                "native_records": _integer(events, "native_records"),
                "model_turns": _integer(activity, "model_turns"),
                "tool_calls": _integer(activity, "tool_calls"),
                "delegations": _integer(activity, "delegations"),
                "compactions": _integer(activity, "compactions"),
                "trace_issues": _integer(activity, "trace_issues"),
                "dropped_events": _integer(storage, "dropped_events"),
                "redactions": _integer(storage, "redactions"),
            }
        )
    return {
        "comparable": all(value is True for value in checks.values()),
        "checks": checks,
        "traces": rows,
        "capability_differences": _capability_differences(summaries),
    }


def render_trace(target: TraceTarget) -> JsonObject:
    """Build deterministic timelines for every attempt in a target."""

    _require_valid(target)
    timelines: list[JsonValue] = []
    for attempt_dir in _attempt_directories(target):
        manifest = _load_object(attempt_dir / "manifest.json")
        timelines.append(
            {
                "trace_id": _string(manifest, "trace_id"),
                "instance_id": _string(manifest, "instance_id"),
                "attempt": _integer(manifest, "attempt"),
                "path": str(attempt_dir),
                "entries": [entry.as_json() for entry in build_timeline(attempt_dir)],
            }
        )
    return {
        "path": str(target.path),
        "schema_version": SCHEMA_VERSION,
        "timelines": timelines,
    }


def _resolve_direct_target(path: Path) -> TraceTarget | None:
    if path.is_file():
        if path.name == "run.json":
            return _target("run", path.parent, path)
        if path.name == "manifest.json":
            return _target("attempt", path.parent, path)
        return None
    if not path.is_dir():
        return None
    if (path / "run.json").is_file():
        return _target("run", path, path / "run.json")
    if (path / "manifest.json").is_file():
        return _target("attempt", path, path / "manifest.json")
    return None


def _target(kind: TraceTargetKind, path: Path, document_path: Path) -> TraceTarget:
    if path.is_symlink():
        raise TraceValidationError(f"Trace path cannot be a symbolic link: {path}")
    document = _load_object(document_path)
    version = document.get("schema_version")
    if version != SCHEMA_VERSION:
        raise TraceValidationError(
            f"Unsupported trace schema version {version!r}: {document_path}"
        )
    contract = _object(document, "contract")
    if (
        contract.get("name") != CONTRACT_NAME
        or contract.get("version") != CONTRACT_VERSION
    ):
        raise TraceValidationError(
            f"Unsupported trace contract header: {document_path}"
        )
    return TraceTarget(kind=kind, path=path, document=document)


def _require_valid(target: TraceTarget) -> None:
    validate_trace_target(target).require_valid()


def _inspect_attempt(path: Path) -> JsonObject:
    manifest = _load_object(path / "manifest.json")
    health = _load_object(path / "health.json")
    capabilities = _load_object(path / "capabilities.json")
    events = read_jsonl(path / "events.jsonl", allow_torn_final_line=False).records
    native = read_jsonl(
        path / "native" / "index.jsonl", allow_torn_final_line=False
    ).records
    return {
        "kind": "attempt",
        "path": str(path),
        "schema_version": SCHEMA_VERSION,
        "contract": manifest["contract"],
        "identity": {
            key: manifest[key]
            for key in (
                "trace_id",
                "run_id",
                "benchmark",
                "framework",
                "instance_id",
                "attempt",
            )
        },
        "lifecycle": {
            "created_at": _string(manifest, "created_at"),
            "finalized_at": _string(manifest, "finalized_at"),
            "complete": _boolean(manifest, "complete"),
        },
        "producer": _object(manifest, "producer"),
        "provenance": _object(manifest, "provenance"),
        "execution": _object(manifest, "execution"),
        "health": health,
        "capabilities": _array(capabilities, "capabilities"),
        "records": {
            "events": len(events),
            "native": len(native),
        },
    }


def _attempt_directories(target: TraceTarget) -> tuple[Path, ...]:
    if target.kind == "attempt":
        return (target.path,)
    attempts = _array(target.document, "attempts")
    directories = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise TraceValidationError("Run index contains an invalid attempt")
        relative = attempt.get("path")
        if not isinstance(relative, str):
            raise TraceValidationError("Run attempt path is invalid")
        directory = (target.path / relative).resolve()
        if not directory.is_relative_to(target.path):
            raise TraceValidationError("Run attempt escapes its trace root")
        directories.append(directory)
    return tuple(directories)


def _summarize_capabilities(documents: list[JsonObject]) -> JsonObject:
    values: dict[str, dict[str, set[str]]] = {}
    for document in documents:
        for capability in _array(document, "capabilities"):
            if not isinstance(capability, dict):
                raise TraceValidationError("Capability entry is invalid")
            category = capability.get("category")
            if not isinstance(category, str):
                raise TraceValidationError("Capability category is invalid")
            summary = values.setdefault(
                category,
                {"states": set(), "coverage": set(), "timing": set()},
            )
            for source, destination in (
                ("state", "states"),
                ("coverage", "coverage"),
                ("timing", "timing"),
            ):
                value = capability.get(source)
                if not isinstance(value, str):
                    raise TraceValidationError("Capability value is invalid")
                summary[destination].add(value)
    result: JsonObject = {}
    for category, summary in sorted(values.items()):
        entry: JsonObject = {}
        for key, items in sorted(summary.items()):
            entry[key] = [cast(JsonValue, item) for item in sorted(items)]
        result[category] = entry
    return result


def _capability_differences(summaries: list[JsonObject]) -> JsonObject:
    matrices = [_object(summary, "capabilities") for summary in summaries]
    categories = sorted({key for matrix in matrices for key in matrix})
    result: JsonObject = {}
    for category in categories:
        items: list[JsonValue] = [
            matrix.get(category, {"states": [], "coverage": [], "timing": []})
            for matrix in matrices
        ]
        if not _same(items):
            result[category] = items
    return result


def _model_turn_count(events: tuple[JsonObject, ...]) -> int:
    attempts: dict[str, list[JsonObject]] = {}
    for event in events:
        attempts.setdefault(_string(event, "trace_id"), []).append(event)

    total = 0
    for attempt in attempts.values():
        starts = tuple(
            event
            for event in attempt
            if _string(event, "event_family") == "model"
            and _string(event, "phase") == "start"
        )
        if starts:
            total += _unique_span_count(starts)
            continue
        responses = tuple(
            event
            for event in attempt
            if _string(event, "event_family") == "model"
            and _string(event, "phase") == "instant"
            and _string(event, "event_type").endswith((".request", ".response"))
        )
        total += _unique_span_count(responses)
    return total


def _activity_span_count(
    events: tuple[JsonObject, ...],
    families: set[str],
) -> int:
    return _unique_span_count(
        tuple(
            event
            for event in events
            if _string(event, "event_family") in families
            and _string(event, "phase") in {"start", "instant"}
        )
    )


def _unique_span_count(events: tuple[JsonObject, ...]) -> int:
    return len(
        {
            (
                _string(event, "trace_id"),
                _string(event, "span_id"),
            )
            for event in events
        }
    )


def _event_duration(event: JsonObject) -> float | None:
    timing = _object(event, "timing")
    value = timing.get("duration_ms")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TraceValidationError("Trace duration is not numeric")
    return float(value)


def _duration_statistics(values: list[float]) -> JsonObject:
    if not values:
        return {
            "count": 0,
            "total": 0.0,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "total": sum(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": _percentile(ordered, 0.5),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _unique_objects(values: list[JsonObject]) -> list[JsonValue]:
    unique: dict[str, JsonObject] = {}
    for value in values:
        unique.setdefault(
            json.dumps(
                value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ),
            value,
        )
    return [unique[key] for key in sorted(unique)]


def _counter(values: Iterable[str]) -> JsonObject:
    return _counter_json(Counter(values))


def _counter_json(counter: Counter[str]) -> JsonObject:
    return {key: counter[key] for key in sorted(counter)}


def _same(values: Sequence[object]) -> bool:
    if not values:
        return True
    baseline = values[0]
    return all(value == baseline for value in values[1:])


def _issue_as_json(issue: ValidationIssue) -> JsonObject:
    return {
        "severity": issue.severity,
        "code": issue.code,
        "path": issue.path,
        "message": issue.message,
    }


def _load_object(path: Path) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceValidationError(f"Trace document is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise TraceValidationError(f"Trace document is not an object: {path}")
    return cast(JsonObject, value)


def _object(value: JsonObject, key: str) -> JsonObject:
    item = value.get(key)
    if not isinstance(item, dict):
        raise TraceValidationError(f"Trace field {key!r} is not an object")
    return item


def _array(value: JsonObject, key: str) -> list[JsonValue]:
    item = value.get(key)
    if not isinstance(item, list):
        raise TraceValidationError(f"Trace field {key!r} is not an array")
    return item


def _string(value: JsonObject, key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise TraceValidationError(f"Trace field {key!r} is not a string")
    return item


def _string_list(value: JsonObject, key: str) -> list[JsonValue]:
    items = _array(value, key)
    if any(not isinstance(item, str) for item in items):
        raise TraceValidationError(f"Trace field {key!r} is not a string array")
    return items


def _integer(value: JsonObject, key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise TraceValidationError(f"Trace field {key!r} is not an integer")
    return item


def _boolean(value: JsonObject, key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise TraceValidationError(f"Trace field {key!r} is not a boolean")
    return item
