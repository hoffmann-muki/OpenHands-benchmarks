import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmarks.tracing import (
    TraceConfig,
    TraceIdentity,
    TraceProducer,
    TraceRecorder,
    build_timeline,
)
from benchmarks.tracing.adapters.openhands import (
    OpenHandsTraceAdapter,
    openhands_capabilities,
)
from benchmarks.tracing.models import JsonObject
from benchmarks.tracing.storage import read_jsonl
from openhands.sdk.event import (
    ActionEvent,
    Condensation,
    CondensationRequest,
    LLMCompletionLogEvent,
    MessageEvent,
    ObservationEvent,
)
from openhands.sdk.llm import Message, MessageToolCall, TextContent
from openhands.tools.task.definition import TaskAction, TaskObservation
from openhands.tools.terminal.definition import (
    TerminalAction,
    TerminalObservation,
)


def _config(
    tmp_path: Path,
    *,
    delegation_enabled: bool = True,
    condenser_enabled: bool = True,
    completion_logs_enabled: bool = False,
) -> TraceConfig:
    return TraceConfig(
        attempt_dir=tmp_path / "attempt",
        identity=TraceIdentity(
            trace_id="trace-openhands-test",
            run_id="run-openhands-test",
            benchmark="swe-bench-verified",
            framework="openhands",
            instance_id="owner/project__issue-1",
            attempt=1,
        ),
        producer=TraceProducer(
            name="openhands-trace-adapter",
            version="1.1.0",
        ),
        provenance={
            "benchmark": {
                "name": "OpenHands-benchmarks",
                "revision": "benchmark-revision",
            },
            "framework": {
                "name": "OpenHands SDK",
                "revision": "framework-revision",
            },
            "adapter": {
                "name": "benchmarks.tracing.adapters.openhands",
                "revision": "adapter-revision",
            },
        },
        execution={
            "model": "openrouter/qwen/qwen3-coder-next",
            "evaluation_workers": 1,
            "inference_timeout_seconds": 3600,
            "benchmark_retries": 0,
            "provider_attempts": 1,
        },
        capabilities=openhands_capabilities(
            delegation_enabled=delegation_enabled,
            condenser_enabled=condenser_enabled,
            browser_enabled=False,
            completion_logs_enabled=completion_logs_enabled,
        ),
    )


def _adapter(
    tmp_path: Path,
    *,
    delegation_enabled: bool = True,
    condenser_enabled: bool = True,
    completion_logs_enabled: bool = False,
) -> OpenHandsTraceAdapter:
    return OpenHandsTraceAdapter(
        TraceRecorder(
            _config(
                tmp_path,
                delegation_enabled=delegation_enabled,
                condenser_enabled=condenser_enabled,
                completion_logs_enabled=completion_logs_enabled,
            )
        ),
        session_id="conversation-001",
        delegation_enabled=delegation_enabled,
        condenser_enabled=condenser_enabled,
        browser_enabled=False,
        completion_logs_enabled=completion_logs_enabled,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _terminal_action(
    *,
    command: str = "ls -la",
    event_id: str = "action-001",
    timestamp: str = "2026-01-01T00:00:00+00:00",
) -> ActionEvent:
    return ActionEvent(
        id=event_id,
        timestamp=timestamp,
        thought=[TextContent(text="Inspect the workspace")],
        action=TerminalAction(command=command),
        tool_name="terminal",
        tool_call_id="call-001",
        tool_call=MessageToolCall(
            id="call-001",
            name="terminal",
            arguments=json.dumps({"command": command}),
            origin="completion",
        ),
        llm_response_id="response-001",
    )


def _terminal_observation(
    *,
    output: str = "src\nREADME.md\n",
    exit_code: int = 0,
) -> ObservationEvent:
    return ObservationEvent(
        id="observation-001",
        timestamp="2026-01-01T00:00:01.250+00:00",
        tool_name="terminal",
        tool_call_id="call-001",
        action_id="action-001",
        observation=TerminalObservation.from_text(
            output,
            command="ls -la",
            exit_code=exit_code,
        ),
    )


def _documents(attempt_dir: Path) -> tuple[tuple[JsonObject, ...], JsonObject]:
    events = read_jsonl(
        attempt_dir / "events.jsonl",
        allow_torn_final_line=False,
    ).records
    capabilities = json.loads(
        (attempt_dir / "capabilities.json").read_text(encoding="utf-8")
    )
    assert isinstance(capabilities, dict)
    return events, capabilities


def _capability(document: JsonObject, category: str) -> JsonObject:
    values = document["capabilities"]
    assert isinstance(values, list)
    match = next(
        value
        for value in values
        if isinstance(value, dict) and value.get("category") == category
    )
    return match


def test_openhands_shell_trace_is_complete_and_correlated(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    adapter.start()
    adapter(_terminal_action())
    adapter(_terminal_observation())

    result = adapter.finish("completed")
    events, capabilities = _documents(tmp_path / "attempt")
    native = read_jsonl(
        tmp_path / "attempt" / "native" / "index.jsonl",
        allow_torn_final_line=False,
    ).records

    assert result.validation.valid
    assert result.manifest["complete"] is True
    assert result.health["status"] == "healthy"
    assert [event["event_type"] for event in events] == [
        "instance.start",
        "attempt.start",
        "harness.startup_start",
        "harness.startup_end",
        "agent.execution_start",
        "agent.session_start",
        "model.turn_start",
        "model.turn_end",
        "shell.start",
        "shell.end",
        "model.turn_start",
        "model.turn_end",
        "agent.session_end",
        "agent.execution_end",
        "harness.shutdown_start",
        "harness.shutdown_end",
        "attempt.end",
        "instance.end",
    ]
    shell_start = next(
        event for event in events if event["event_type"] == "shell.start"
    )
    shell_end = next(event for event in events if event["event_type"] == "shell.end")
    assert shell_start["span_id"] == shell_end["span_id"]
    assert shell_end["timing"] == {
        "duration_ms": 1250.0,
        "fidelity": "derived",
    }
    relations = shell_end["relations"]
    assert isinstance(relations, list)
    assert relations == [{"event_id": shell_start["event_id"], "type": "caused_by"}]
    assert len(native) == 2
    assert all(record["event_ids"] for record in native)
    assert _capability(capabilities, "shell")["state"] == "captured"
    assert _capability(capabilities, "tool.timing")["state"] == "derived"
    assert _capability(capabilities, "provider.exchange")["state"] == "disabled"
    timeline = build_timeline(tmp_path / "attempt")
    shell_line = next(entry for entry in timeline if entry.event_type == "shell.start")
    assert shell_line.detail == "ls -la"


def test_completion_log_removes_credentials_and_accounting_before_storage(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path, completion_logs_enabled=True)
    adapter.start()
    secret = "sk-" + ("synthetic" * 4)
    adapter(
        LLMCompletionLogEvent(
            id="completion-log-001",
            timestamp="2026-01-01T00:00:02+00:00",
            filename="completion.json",
            model_name="openrouter/qwen/qwen3-coder-next",
            usage_id="default",
            log_data=json.dumps(
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "kwargs": {"api_key": secret, "temperature": 0.1},
                    "response": {"id": "response-001", "content": "world"},
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "usage_summary": {"total_tokens": 15},
                    "cost": 0.25,
                    "latency_sec": 1.5,
                }
            ),
        )
    )

    result = adapter.finish("completed")
    events, capabilities = _documents(tmp_path / "attempt")
    provider = next(
        event for event in events if event["event_type"] == "provider.response"
    )
    artifacts = provider["artifacts"]
    assert isinstance(artifacts, list)
    reference = artifacts[0]
    assert isinstance(reference, dict)
    retained = json.loads(
        (tmp_path / "attempt" / str(reference["path"])).read_text(encoding="utf-8")
    )

    assert result.validation.valid
    assert retained == {
        "kwargs": {"temperature": 0.1},
        "latency_sec": 1.5,
        "messages": [{"content": "hello", "role": "user"}],
        "response": {"content": "world", "id": "response-001"},
    }
    assert provider["timing"] == {
        "duration_ms": 1500.0,
        "fidelity": "native_wall",
    }
    assert _capability(capabilities, "provider.exchange")["state"] == "captured"
    for path in (tmp_path / "attempt").rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            assert secret.encode() not in content
            assert b'"prompt_tokens"' not in content
            assert b'"cost"' not in content


def test_condensation_boundaries_preserve_summary_and_duration(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    adapter.start()
    adapter(
        MessageEvent(
            id="message-before-condensation",
            timestamp="2026-01-01T00:00:02+00:00",
            source="user",
            llm_message=Message(
                role="user",
                content=[TextContent(text="Continue after compacting context")],
            ),
        )
    )
    adapter(
        CondensationRequest(
            id="condensation-request-001",
            timestamp="2026-01-01T00:00:03+00:00",
        )
    )
    adapter(
        Condensation(
            id="condensation-001",
            timestamp="2026-01-01T00:00:05+00:00",
            forgotten_event_ids={"old-event-1", "old-event-2"},
            summary="Inspected the repository and found the relevant module.",
            summary_offset=1,
            llm_response_id="condensation-response-001",
        )
    )

    result = adapter.finish("completed")
    events, capabilities = _documents(tmp_path / "attempt")
    boundaries = [
        event for event in events if str(event["event_type"]).startswith("context.")
    ]

    assert result.validation.valid
    assert [event["event_type"] for event in boundaries] == [
        "context.compaction_start",
        "context.compaction_end",
    ]
    assert boundaries[1]["timing"] == {
        "duration_ms": 2000.0,
        "fidelity": "derived",
    }
    model_end = next(
        event for event in events if event["event_type"] == "model.turn_end"
    )
    model_payload = model_end["payload"]
    assert isinstance(model_payload, dict)
    assert model_end["occurred_at"] == boundaries[0]["occurred_at"]
    assert model_payload["boundary"] == "context_compaction"
    assert _capability(capabilities, "context.compaction")["state"] == "captured"


def test_incomplete_native_action_degrades_trace_without_unbalanced_spans(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    adapter.start()
    adapter(_terminal_action())

    result = adapter.finish("completed")
    events, _ = _documents(tmp_path / "attempt")
    shell_end = next(event for event in events if event["event_type"] == "shell.end")

    assert result.validation.valid
    assert result.manifest["complete"] is False
    assert result.health["status"] == "degraded"
    assert shell_end["status"] == "degraded"
    issues = result.health["issues"]
    assert isinstance(issues, list)
    assert "openhands.incomplete_native_span" in {
        str(issue["code"]) for issue in issues if isinstance(issue, dict)
    }


def test_adapter_normalization_failure_keeps_native_evidence_and_agent_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(tmp_path)
    adapter.start()

    def fail_normalization(*_: object) -> None:
        raise ValueError("synthetic adapter failure")

    monkeypatch.setattr(adapter, "_normalize", fail_normalization)
    event = MessageEvent(
        id="message-001",
        timestamp="2026-01-01T00:00:06+00:00",
        source="user",
        llm_message=Message(
            role="user",
            content=[TextContent(text="Please inspect the repository")],
        ),
    )

    assert adapter(event) is None
    result = adapter.finish("completed")
    events, capabilities = _documents(tmp_path / "attempt")
    native = read_jsonl(
        tmp_path / "attempt" / "native" / "index.jsonl",
        allow_torn_final_line=False,
    ).records

    assert result.validation.valid
    assert result.health["status"] == "failed"
    assert "trace.issue" in {str(item["event_type"]) for item in events}
    assert len(native) == 1
    assert native[0]["event_ids"]
    assert _capability(capabilities, "native.evidence")["state"] == "captured"


def test_duplicate_callback_delivery_is_idempotent(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    adapter.start()
    event = _terminal_action()
    adapter(event)
    adapter(event)
    adapter(_terminal_observation())

    result = adapter.finish("completed")
    events, _ = _documents(tmp_path / "attempt")
    native = read_jsonl(
        tmp_path / "attempt" / "native" / "index.jsonl",
        allow_torn_final_line=False,
    ).records

    assert result.validation.valid
    assert [event["event_type"] for event in events].count("shell.start") == 1
    assert len(native) == 2


def test_durable_child_conversation_is_normalized_under_logical_delegation(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    prompt = "Inspect the implementation and report exact file locations."
    task_action = ActionEvent(
        id="task-action-001",
        timestamp="2026-01-01T00:00:01+00:00",
        thought=[TextContent(text="Delegate repository inspection")],
        action=TaskAction(
            prompt=prompt,
            subagent_type="benchmark-navigator",
            description="Inspect implementation",
        ),
        tool_name="task",
        tool_call_id="task-call-001",
        tool_call=MessageToolCall(
            id="task-call-001",
            name="task",
            arguments=json.dumps(
                {
                    "prompt": prompt,
                    "subagent_type": "benchmark-navigator",
                }
            ),
            origin="completion",
        ),
        llm_response_id="task-response-001",
    )
    task_observation = ObservationEvent(
        id="task-observation-001",
        timestamp="2026-01-01T00:00:05+00:00",
        tool_name="task",
        tool_call_id="task-call-001",
        action_id="task-action-001",
        observation=TaskObservation.from_text(
            "Inspection complete.",
            task_id="task_00000001",
            subagent="benchmark-navigator",
            status="completed",
        ),
    )
    adapter(task_action)
    adapter(task_observation)

    child = tmp_path / "persistence" / "root" / "subagents" / "child-001"
    event_dir = child / "events"
    event_dir.mkdir(parents=True)
    (child / "base_state.json").write_text(
        json.dumps({"execution_status": "finished"}),
        encoding="utf-8",
    )
    child_events = (
        MessageEvent(
            id="child-message-001",
            timestamp="2026-01-01T00:00:01.100+00:00",
            source="user",
            llm_message=Message(
                role="user",
                content=[TextContent(text=prompt)],
            ),
        ),
        _terminal_action(
            command="rg -n 'class Widget' src",
            event_id="child-action-001",
            timestamp="2026-01-01T00:00:02+00:00",
        ).model_copy(
            update={
                "tool_call_id": "child-call-001",
                "tool_call": MessageToolCall(
                    id="child-call-001",
                    name="terminal",
                    arguments=json.dumps(
                        {
                            "command": "rg -n 'class Widget' src",
                        }
                    ),
                    origin="completion",
                ),
                "llm_response_id": "child-response-001",
            }
        ),
        _terminal_observation(
            output="src/widget.py:10:class Widget\n",
        ).model_copy(
            update={
                "id": "child-observation-001",
                "timestamp": "2026-01-01T00:00:03+00:00",
                "tool_call_id": "child-call-001",
                "action_id": "child-action-001",
            }
        ),
    )
    for sequence, event in enumerate(child_events):
        (event_dir / f"event-{sequence:05d}-{event.id}.json").write_text(
            event.model_dump_json(exclude_none=True),
            encoding="utf-8",
        )

    assert adapter.ingest_conversation_directory(tmp_path / "persistence") == 1
    result = adapter.finish("completed")
    events, capabilities = _documents(tmp_path / "attempt")
    delegation = next(
        event for event in events if event["event_type"] == "delegation.start"
    )
    child_session = next(
        event
        for event in events
        if event["event_type"] == "agent.session_start"
        and event.get("agent_id") == "benchmark-navigator"
    )
    child_shell = next(
        event
        for event in events
        if event["event_type"] == "shell.start"
        and event.get("agent_id") == "benchmark-navigator"
    )

    assert result.validation.valid
    assert result.health["status"] == "healthy"
    assert child_session["parent_span_id"] == delegation["span_id"]
    assert child_session["session_id"] == "child-001"
    assert child_shell["session_id"] == "child-001"
    assert child_shell["origin"] == {
        "capture_method": "native_export",
        "component": "openhands.sdk.archived_subagent",
    }
    assert _capability(capabilities, "delegation")["coverage"] == "full"
    assert _capability(capabilities, "native.evidence")["coverage"] == "full"

    archive = tmp_path / "conversation.tar.gz"
    with tarfile.open(archive, mode="w:gz") as output:
        output.add(
            tmp_path / "persistence" / "root",
            arcname="workspace/conversations/root",
        )
    archived_adapter = _adapter(tmp_path / "archive-case")
    archived_adapter(task_action)
    archived_adapter(task_observation)
    assert archived_adapter.ingest_conversation_archive(archive) == 1
    archived_result = archived_adapter.finish("completed")
    archived_events, _ = _documents(tmp_path / "archive-case" / "attempt")
    assert archived_result.validation.valid
    assert any(
        event["event_type"] == "shell.start"
        and event.get("agent_id") == "benchmark-navigator"
        for event in archived_events
    )

    incomplete_adapter = _adapter(tmp_path / "incomplete-case")
    incomplete_adapter(task_action)
    incomplete_adapter(task_observation)
    empty_persistence = tmp_path / "empty-persistence"
    empty_persistence.mkdir()
    assert incomplete_adapter.ingest_conversation_directory(empty_persistence) == 0
    incomplete_result = incomplete_adapter.finish("completed")
    _, incomplete_capabilities = _documents(tmp_path / "incomplete-case" / "attempt")
    assert incomplete_result.health["status"] == "degraded"
    assert _capability(incomplete_capabilities, "delegation")["coverage"] == "partial"
    assert {issue["code"] for issue in incomplete_result.health["issues"]} == {
        "openhands.child_persistence_incomplete"
    }


def test_disabled_features_remain_explicit_in_capability_matrix() -> None:
    capabilities = openhands_capabilities(
        delegation_enabled=False,
        condenser_enabled=False,
        browser_enabled=False,
        completion_logs_enabled=False,
    )
    states = {capability.category: capability.state for capability in capabilities}

    assert len(capabilities) == 18
    assert states["delegation"] == "disabled"
    assert states["context.compaction"] == "disabled"
    assert states["browser"] == "disabled"
    assert states["provider.exchange"] == "disabled"
    assert states["memory"] == "not_exposed"
