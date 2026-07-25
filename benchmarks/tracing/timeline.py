"""Deterministic timeline reconstruction from normalized trace events."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from benchmarks.tracing.errors import TraceStorageError, TraceValidationError
from benchmarks.tracing.models import JsonObject
from benchmarks.tracing.storage import read_jsonl
from benchmarks.tracing.validation import ContractValidator


@dataclass(frozen=True, slots=True)
class TimelineEntry:
    sequence: int
    event_id: str
    event_type: str
    phase: str
    status: str
    occurred_at: str
    recorded_at: str
    relative_ms: float
    captured_relative_ms: float
    capture_delay_ms: float
    duration_ms: float | None
    depth: int
    span_id: str
    parent_span_id: str | None
    actor: str
    lane: str
    detail: str | None
    artifacts: tuple[JsonObject, ...]
    artifact_roles: tuple[str, ...]

    def as_json(self) -> JsonObject:
        value: JsonObject = {
            "sequence": self.sequence,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "phase": self.phase,
            "status": self.status,
            "occurred_at": self.occurred_at,
            "recorded_at": self.recorded_at,
            "relative_ms": self.relative_ms,
            "captured_relative_ms": self.captured_relative_ms,
            "capture_delay_ms": self.capture_delay_ms,
            "depth": self.depth,
            "span_id": self.span_id,
            "actor": self.actor,
            "lane": self.lane,
            "artifacts": [dict(artifact) for artifact in self.artifacts],
            "artifact_roles": list(self.artifact_roles),
        }
        if self.duration_ms is not None:
            value["duration_ms"] = self.duration_ms
        if self.parent_span_id is not None:
            value["parent_span_id"] = self.parent_span_id
        if self.detail is not None:
            value["detail"] = self.detail
        return value


def build_timeline(
    attempt_dir: Path,
    *,
    validator: ContractValidator | None = None,
    order: Literal["source", "capture", "sequence"] = "source",
) -> tuple[TimelineEntry, ...]:
    active_validator = validator or ContractValidator()
    try:
        events = read_jsonl(
            attempt_dir / "events.jsonl",
            allow_torn_final_line=False,
        ).records
    except TraceStorageError as exc:
        raise TraceValidationError(
            "Timeline source is not valid finalized JSONL"
        ) from exc
    for index, event in enumerate(events, start=1):
        issues = active_validator.validate_document(
            "event.schema.json",
            event,
            path=f"events.jsonl:{index}",
        )
        if issues:
            raise TraceValidationError("Timeline source contains an invalid event")
    if not events:
        return ()

    source_baseline = min(
        _parse_timestamp(str(event["occurred_at"])) for event in events
    )
    capture_baseline = min(
        _parse_timestamp(str(event["recorded_at"])) for event in events
    )
    parents = {
        str(event["span_id"]): (
            str(event["parent_span_id"])
            if event.get("parent_span_id") is not None
            else None
        )
        for event in events
    }
    depths = {span_id: _span_depth(span_id, parents, set()) for span_id in parents}
    entries: list[TimelineEntry] = []
    for event in events:
        occurred_at = str(event["occurred_at"])
        recorded_at = str(event["recorded_at"])
        occurred = _parse_timestamp(occurred_at)
        recorded = _parse_timestamp(recorded_at)
        timing = event["timing"]
        assert isinstance(timing, dict)
        duration = timing.get("duration_ms")
        assert duration is None or isinstance(duration, int | float)
        artifacts = event["artifacts"]
        assert isinstance(artifacts, list)
        entries.append(
            TimelineEntry(
                sequence=_integer(event["sequence"]),
                event_id=str(event["event_id"]),
                event_type=str(event["event_type"]),
                phase=str(event["phase"]),
                status=str(event["status"]),
                occurred_at=occurred_at,
                recorded_at=recorded_at,
                relative_ms=(occurred - source_baseline).total_seconds() * 1000,
                captured_relative_ms=(recorded - capture_baseline).total_seconds()
                * 1000,
                capture_delay_ms=(recorded - occurred).total_seconds() * 1000,
                duration_ms=float(duration) if duration is not None else None,
                depth=depths[str(event["span_id"])],
                span_id=str(event["span_id"]),
                parent_span_id=(
                    str(event["parent_span_id"])
                    if event.get("parent_span_id") is not None
                    else None
                ),
                actor=str(
                    event.get("agent_id") or event.get("session_id") or "harness"
                ),
                lane=_event_lane(event),
                detail=_event_detail(event),
                artifacts=tuple(
                    dict(artifact)
                    for artifact in artifacts
                    if isinstance(artifact, dict)
                ),
                artifact_roles=tuple(
                    str(artifact["role"])
                    for artifact in artifacts
                    if isinstance(artifact, dict)
                ),
            )
        )
    if order == "sequence":
        return tuple(entries)
    if order == "capture":
        return tuple(
            sorted(
                entries,
                key=lambda entry: (
                    _parse_timestamp(entry.recorded_at),
                    entry.sequence,
                ),
            )
        )
    return tuple(
        sorted(
            entries,
            key=lambda entry: (
                _parse_timestamp(entry.occurred_at),
                entry.sequence,
            ),
        )
    )


def render_timeline(entries: tuple[TimelineEntry, ...]) -> str:
    lines: list[str] = []
    for entry in entries:
        duration = (
            f" duration={entry.duration_ms:.3f}ms"
            if entry.duration_ms is not None
            else ""
        )
        detail = f" {entry.detail}" if entry.detail else ""
        artifacts = (
            " artifacts="
            + ",".join(
                f"{artifact.get('role', 'artifact')}:{artifact.get('path', '?')}"
                for artifact in entry.artifacts
            )
            if entry.artifacts
            else ""
        )
        lines.append(
            f"+{entry.relative_ms:012.3f}ms "
            f"{'  ' * entry.depth}{entry.event_type} "
            f"{entry.phase}/{entry.status} lane={entry.lane} actor={entry.actor}"
            f"{duration}{detail}{artifacts}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TraceValidationError("Timeline contains an invalid timestamp") from exc
    return parsed.astimezone(UTC)


def _span_depth(
    span_id: str,
    parents: dict[str, str | None],
    active: set[str],
) -> int:
    parent = parents.get(span_id)
    if parent is None:
        return 0
    if span_id in active or parent not in parents:
        return 0
    return 1 + _span_depth(parent, parents, {*active, span_id})


def _event_detail(event: JsonObject) -> str | None:
    payload = event["payload"]
    if not isinstance(payload, dict):
        return None
    tool = payload.get("tool")
    if not isinstance(tool, dict):
        return None
    arguments = tool.get("arguments")
    if isinstance(arguments, dict):
        command = arguments.get("command")
        if isinstance(command, str):
            return command
    name = tool.get("name")
    return str(name) if isinstance(name, str) else None


def _event_lane(event: JsonObject) -> str:
    agent_id = event.get("agent_id")
    if isinstance(agent_id, str):
        return f"agent:{agent_id}"
    session_id = event.get("session_id")
    if isinstance(session_id, str):
        return f"session:{session_id}"
    return f"harness:{event['event_family']}"


def _integer(value) -> int:
    if not isinstance(value, int):
        raise TraceValidationError("Timeline sequence is not an integer")
    return value
