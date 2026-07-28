import json
from typing import Literal

from benchmarks.tracing.execution_tree import (
    build_execution_tree,
    execution_tree_event_ids,
)
from benchmarks.tracing.models import JsonObject, TraceIdentity
from benchmarks.tracing.validation import ContractValidator


IDENTITY = TraceIdentity(
    trace_id="trace-tree",
    run_id="run-tree",
    benchmark="custom-benchmark",
    framework="test-agent",
    instance_id="instance-1",
    attempt=1,
)


def _event(
    sequence: int,
    event_type: str,
    phase: Literal["start", "end", "instant"],
    status: str,
    span_id: str,
    occurred_at: str,
    *,
    parent_span_id: str | None = None,
    payload: JsonObject | None = None,
    duration_ms: float | None = None,
) -> JsonObject:
    timing: JsonObject = {"fidelity": "derived"}
    if duration_ms is not None:
        timing["duration_ms"] = duration_ms
    return {
        "event_id": f"event-{sequence:03d}",
        "sequence": sequence,
        "span_id": span_id,
        **({"parent_span_id": parent_span_id} if parent_span_id else {}),
        "occurred_at": occurred_at,
        "recorded_at": occurred_at,
        "event_type": event_type,
        "event_family": event_type.split(".", 1)[0],
        "phase": phase,
        "status": status,
        "origin": {
            "component": "test-agent",
            "capture_method": "derived",
        },
        "timing": timing,
        "payload": payload or {},
        "artifacts": [],
    }


def _tree(events: list[JsonObject]) -> JsonObject:
    content = b"".join(
        json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in events
    )
    validator = ContractValidator()
    return build_execution_tree(
        events,
        identity=IDENTITY,
        schema_digest=validator.schema_digest,
        events_content=content,
    )


def test_execution_tree_pairs_spans_and_marks_concurrent_siblings() -> None:
    events = [
        _event(
            1,
            "attempt.start",
            "start",
            "started",
            "attempt",
            "2026-01-01T00:00:00.000Z",
        ),
        _event(
            2,
            "model.turn_start",
            "start",
            "started",
            "model-a",
            "2026-01-01T00:00:00.100Z",
            parent_span_id="attempt",
            payload={"request": "first"},
        ),
        _event(
            3,
            "model.turn_start",
            "start",
            "started",
            "model-b",
            "2026-01-01T00:00:00.150Z",
            parent_span_id="attempt",
            payload={"request": "second"},
        ),
        _event(
            4,
            "model.turn_end",
            "end",
            "completed",
            "model-b",
            "2026-01-01T00:00:00.250Z",
            parent_span_id="attempt",
            payload={"response": "second"},
            duration_ms=100,
        ),
        _event(
            5,
            "model.turn_end",
            "end",
            "completed",
            "model-a",
            "2026-01-01T00:00:00.300Z",
            parent_span_id="attempt",
            payload={"response": "first"},
            duration_ms=200,
        ),
        _event(
            6,
            "context.compaction",
            "instant",
            "completed",
            "compaction",
            "2026-01-01T00:00:00.400Z",
            parent_span_id="attempt",
        ),
        _event(
            7,
            "attempt.end",
            "end",
            "completed",
            "attempt",
            "2026-01-01T00:00:00.500Z",
            duration_ms=500,
        ),
    ]

    tree = _tree(events)
    root = tree["root"]
    assert isinstance(root, dict)
    root_children = root["children"]
    assert isinstance(root_children, list)
    attempt = root_children[0]
    assert isinstance(attempt, dict)
    children = attempt["children"]
    assert isinstance(children, list)
    first = children[0]
    second = children[1]
    assert isinstance(first, dict)
    assert isinstance(second, dict)

    assert tree["complete"] is True
    assert first["source_event_ids"] == ["event-002", "event-005"]
    assert first["input"] == {"payload": {"request": "first"}, "artifacts": []}
    assert first["output"] == {"payload": {"response": "first"}, "artifacts": []}
    assert first["concurrency_group"] == second["concurrency_group"]
    assert first["overlaps_with"] == [second["node_id"]]
    assert second["overlaps_with"] == [first["node_id"]]
    assert sorted(execution_tree_event_ids(tree)) == sorted(
        str(event["event_id"]) for event in events
    )
    assert not ContractValidator().validate_document(
        "execution-tree.schema.json",
        tree,
    )


def test_execution_tree_duration_is_stable_across_language_runtimes() -> None:
    events = [
        _event(
            1,
            "attempt.start",
            "start",
            "started",
            "attempt",
            "2026-01-01T00:00:00.000Z",
        ),
        _event(
            2,
            "attempt.end",
            "end",
            "completed",
            "attempt",
            "2026-01-01T00:02:08.824Z",
        ),
    ]

    tree = _tree(events)
    root = tree["root"]
    assert isinstance(root, dict)
    children = root["children"]
    assert isinstance(children, list)
    attempt = children[0]
    assert isinstance(attempt, dict)

    assert root["duration_ms"] == 128_824.0
    assert attempt["duration_ms"] == 128_824.0


def test_execution_tree_preserves_unmatched_boundaries_with_warnings() -> None:
    events = [
        _event(
            1,
            "attempt.start",
            "start",
            "started",
            "attempt",
            "2026-01-01T00:00:00.000Z",
        ),
        _event(
            2,
            "tool.end",
            "end",
            "failed",
            "missing-tool",
            "2026-01-01T00:00:00.100Z",
            parent_span_id="attempt",
        ),
    ]

    tree = _tree(events)

    assert tree["complete"] is False
    warnings = tree["warnings"]
    assert isinstance(warnings, list)
    codes: set[str] = set()
    for warning in warnings:
        assert isinstance(warning, dict)
        code = warning["code"]
        assert isinstance(code, str)
        codes.add(code)
    assert codes == {
        "projection.unmatched_end",
        "projection.unmatched_start",
    }
    assert execution_tree_event_ids(tree) == ("event-001", "event-002")


def test_execution_tree_represents_an_empty_recovered_journal() -> None:
    validator = ContractValidator()
    tree = build_execution_tree(
        [],
        identity=IDENTITY,
        schema_digest=validator.schema_digest,
        events_content=b"",
        generated_at="2026-01-01T00:00:00.000Z",
    )

    assert tree["complete"] is True
    assert tree["source"] == {
        "path": "events.jsonl",
        "sha256": ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        "event_count": 0,
        "represented_event_count": 0,
    }
    assert tree["root"] == {
        "node_id": "trace-root",
        "started_at": None,
        "ended_at": None,
        "duration_ms": 0.0,
        "children": [],
    }
    assert not validator.validate_document("execution-tree.schema.json", tree)
