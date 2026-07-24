import json
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
            version="1.0.0",
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
        "agent.session_start",
        "model.response",
        "shell.start",
        "shell.end",
        "agent.session_end",
        "attempt.end",
        "instance.end",
    ]
    shell_start = events[4]
    shell_end = events[5]
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
    assert result.health["status"] == "degraded"
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
