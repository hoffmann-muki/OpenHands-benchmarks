import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

from benchmarks.tracing import (
    Capability,
    ContractValidator,
    TraceConfig,
    TraceIdentity,
    TraceProducer,
    TraceRecorder,
    attempt_directory,
    build_timeline,
    encode_instance_id,
    render_timeline,
    write_run_index,
)
from benchmarks.tracing.artifacts import ArtifactStore
from benchmarks.tracing.errors import (
    TraceFinalizationError,
    TraceInitializationError,
    TraceStorageError,
)
from benchmarks.tracing.models import JsonObject
from benchmarks.tracing.redaction import Redactor
from benchmarks.tracing.storage import (
    JsonlJournal,
    create_private_directory,
    read_jsonl,
)


def _capabilities() -> tuple[Capability, ...]:
    return (
        Capability(
            "agent.session",
            "not_observed",
            "none",
            "not_available",
        ),
        Capability(
            "model.turn",
            "not_observed",
            "none",
            "not_available",
        ),
        Capability(
            "provider.exchange",
            "not_exposed",
            "none",
            "not_available",
            limitations=("Provider payloads are not exposed in this mode.",),
        ),
        Capability(
            "tool.invocation",
            "captured",
            "full",
            "native_monotonic",
            evidence=("shell.start",),
        ),
        Capability(
            "tool.result",
            "captured",
            "full",
            "native_monotonic",
            evidence=("shell.end",),
        ),
        Capability(
            "tool.timing",
            "captured",
            "full",
            "native_monotonic",
            evidence=("shell.end",),
        ),
        Capability(
            "shell",
            "captured",
            "full",
            "native_monotonic",
            evidence=("shell.start", "shell.end"),
        ),
        Capability("file", "not_observed", "none", "not_available"),
        Capability("search", "not_observed", "none", "not_available"),
        Capability("browser", "disabled", "none", "not_applicable"),
        Capability("delegation", "not_observed", "none", "not_available"),
        Capability(
            "context.compaction",
            "not_observed",
            "none",
            "not_available",
        ),
        Capability("memory", "disabled", "none", "not_applicable"),
        Capability(
            "harness.lifecycle",
            "captured",
            "full",
            "native_monotonic",
            evidence=("instance.start", "instance.end"),
        ),
        Capability(
            "container.lifecycle",
            "not_observed",
            "none",
            "not_available",
        ),
        Capability(
            "evaluator.lifecycle",
            "not_observed",
            "none",
            "not_available",
        ),
        Capability("patch", "not_observed", "none", "not_available"),
        Capability(
            "native.evidence",
            "captured",
            "full",
            "native_wall",
            evidence=("shell.start", "shell.end"),
        ),
    )


def _config(tmp_path: Path) -> TraceConfig:
    identity = TraceIdentity(
        trace_id="trace-test-001",
        run_id="run-test-001",
        benchmark="swe-bench-verified",
        framework="example-agent",
        instance_id="example/project__issue-1",
        attempt=1,
    )
    return TraceConfig(
        attempt_dir=attempt_directory(
            tmp_path / "run",
            identity.instance_id,
            identity.attempt,
        ),
        identity=identity,
        producer=TraceProducer("python-reference-recorder", "0.1.0"),
        provenance={
            "benchmark": {
                "name": "swe-bench-verified",
                "revision": "dataset-revision",
            },
            "framework": {
                "name": "example-agent",
                "revision": "framework-revision",
            },
            "adapter": {
                "name": "example-adapter",
                "revision": "adapter-revision",
            },
        },
        execution={
            "model": "example/model",
            "evaluation_workers": 1,
            "inference_timeout_seconds": 1800,
            "evaluation_timeout_seconds": 3600,
            "benchmark_retries": 0,
            "provider_attempts": 1,
        },
        capabilities=_capabilities(),
    )


def _record_complete_trace(recorder: TraceRecorder) -> None:
    assert (
        recorder.record_event(
            event_type="instance.start",
            event_family="instance",
            phase="start",
            status="started",
            span_id="span-instance",
            origin={
                "component": "example-harness",
                "capture_method": "generic_harness",
            },
            timing={
                "fidelity": "native_monotonic",
                "clock_id": "process-1",
                "started_monotonic_ns": 1_000_000_000,
            },
            payload={"workspace": "/workspace"},
            occurred_at="2026-01-02T03:04:05.000Z",
        )
        is not None
    )
    shell_start = recorder.record_event(
        event_type="shell.start",
        event_family="shell",
        phase="start",
        status="started",
        span_id="span-shell",
        parent_span_id="span-instance",
        session_id="session-root",
        agent_id="agent-root",
        turn_id="turn-001",
        origin={
            "component": "example-agent",
            "capture_method": "native_hook",
            "native_event_type": "tool_use",
        },
        timing={
            "fidelity": "native_monotonic",
            "clock_id": "process-1",
            "started_monotonic_ns": 1_500_000_000,
        },
        payload={
            "tool": {
                "name": "shell",
                "arguments": {
                    "command": "ls -1",
                    "argv": ["ls", "-1"],
                    "cwd": "/workspace",
                },
            }
        },
        occurred_at="2026-01-02T03:04:05.200Z",
    )
    assert shell_start is not None
    stdout = recorder.store_text_artifact(
        "README.md\nsrc\n",
        media_type="text/plain",
        role="tool.stdout",
    )
    assert stdout is not None
    shell_end = recorder.record_event(
        event_type="shell.end",
        event_family="shell",
        phase="end",
        status="completed",
        span_id="span-shell",
        parent_span_id="span-instance",
        session_id="session-root",
        agent_id="agent-root",
        turn_id="turn-001",
        origin={
            "component": "example-agent",
            "capture_method": "native_hook",
            "native_event_type": "tool_result",
        },
        timing={
            "fidelity": "native_monotonic",
            "clock_id": "process-1",
            "started_monotonic_ns": 1_500_000_000,
            "ended_monotonic_ns": 1_512_500_000,
            "duration_ms": 12.5,
        },
        payload={"tool": {"name": "shell"}, "exit_code": 0},
        artifacts=(stdout,),
        occurred_at="2026-01-02T03:04:05.213Z",
    )
    assert shell_end is not None
    assert (
        recorder.record_native(
            source="example.native_hook",
            content={
                "type": "tool_result",
                "command": "ls -1",
                "status": "completed",
                "usage": {
                    "prompt_tokens": 10,
                    "estimated_cost": 0.01,
                },
            },
            media_type="application/json",
            event_ids=(shell_start, shell_end),
            recorded_at="2026-01-02T03:04:05.214Z",
        )
        is not None
    )
    assert (
        recorder.record_event(
            event_type="instance.end",
            event_family="instance",
            phase="end",
            status="completed",
            span_id="span-instance",
            origin={
                "component": "example-harness",
                "capture_method": "generic_harness",
            },
            timing={
                "fidelity": "native_monotonic",
                "clock_id": "process-1",
                "started_monotonic_ns": 1_000_000_000,
                "ended_monotonic_ns": 3_000_000_000,
                "duration_ms": 2000,
            },
            payload={"result": "completed"},
            occurred_at="2026-01-02T03:04:07.000Z",
        )
        is not None
    )


def _read_json(path: Path) -> JsonObject:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _write_jsonl(path: Path, records: tuple[JsonObject, ...]) -> None:
    path.write_bytes(
        b"".join(
            (
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode()
            for record in records
        )
    )


def test_redactor_removes_structured_secrets_and_accounting() -> None:
    synthetic_secret = "sk-" + "x" * 24
    result = Redactor().sanitize_object(
        {
            "Authorization": f"Bearer {'a' * 24}",
            "usage": {
                "prompt_tokens": 10,
                "estimated_cost": 0.01,
            },
            "metrics": {
                "tokens": {"input": 10, "output": 5},
                "duration_ms": 25,
            },
            "command": f"API_KEY={synthetic_secret}",
            "message": "cost.py and tokens are ordinary research text",
        }
    )

    assert "Authorization" not in result.value
    assert "usage" not in result.value
    assert result.value["metrics"] == {"duration_ms": 25}
    assert synthetic_secret not in str(result.value["command"])
    assert result.value["message"] == "cost.py and tokens are ordinary research text"
    assert result.matches >= 3
    assert {"field.credential", "field.accounting"} <= set(result.rules)


@pytest.mark.parametrize(
    ("value", "marker"),
    [
        ("Bearer " + "a" * 24, "<redacted:authorization>"),
        (
            "https://researcher:synthetic-password@example.invalid/path",
            "<redacted:uri_password>",
        ),
        (
            "-----BEGIN PRIVATE KEY-----\nsynthetic\n-----END PRIVATE KEY-----",
            "<redacted:private_key>",
        ),
    ],
)
def test_redactor_covers_required_credential_shapes(value: str, marker: str) -> None:
    result = Redactor().sanitize_text(value)

    assert marker in result.value
    assert result.matches == 1


def test_redactor_preserves_non_secret_binary_bytes() -> None:
    value = b"\x00\xffresearch-output\x80"

    result = Redactor().sanitize_bytes(value)

    assert result.value == value
    assert result.matches == 0


def test_redaction_markers_are_idempotent() -> None:
    redactor = Redactor()
    first = redactor.sanitize_text(
        "https://researcher:synthetic-password@example.invalid API_KEY=synthetic-value"
    )
    second = redactor.sanitize_text(first.value)

    assert first.matches == 2
    assert second.value == first.value
    assert second.matches == 0


def test_artifact_store_redacts_before_hashing_and_deduplicates(
    tmp_path: Path,
) -> None:
    attempt_dir = tmp_path / "attempt"
    create_private_directory(attempt_dir)
    secret = "sk-" + "z" * 24
    store = ArtifactStore(attempt_dir)

    first = store.put_text(
        f"API_KEY={secret}\n",
        media_type="text/plain",
        role="tool.stdout",
    )
    second = store.put_text(
        f"API_KEY={secret}\n",
        media_type="text/plain",
        role="tool.stdout",
    )
    artifact = attempt_dir / str(first.reference["path"])
    content = artifact.read_bytes()

    assert first.created is True
    assert second.created is False
    assert secret.encode() not in content
    assert hashlib.sha256(content).hexdigest() == first.reference["sha256"]
    assert first.reference["redaction"] == {
        "status": "applied",
        "matches": 1,
        "rules": ["credential.model_api_key"],
    }
    if os.name != "nt":
        assert artifact.stat().st_mode & 0o777 == 0o600
        assert artifact.parent.stat().st_mode & 0o777 == 0o700


def test_journal_recovers_only_a_torn_final_line(tmp_path: Path) -> None:
    directory = tmp_path / "trace"
    create_private_directory(directory)
    path = directory / "journal.jsonl"
    journal = JsonlJournal.create(path)
    journal.append({"sequence": 1})
    journal.append({"sequence": 2})
    journal.close()
    with path.open("ab") as stream:
        stream.write(b'{"sequence":3')

    recovered = read_jsonl(path, allow_torn_final_line=True)

    assert [record["sequence"] for record in recovered.records] == [1, 2]
    assert recovered.torn_final_line is True
    with pytest.raises(TraceStorageError):
        read_jsonl(path, allow_torn_final_line=False)


def test_recorder_finalizes_a_valid_lossless_trace(tmp_path: Path) -> None:
    recorder = TraceRecorder(_config(tmp_path))
    _record_complete_trace(recorder)

    result = recorder.finalize()

    assert result.validation.valid
    assert result.manifest["complete"] is True
    assert result.health["status"] == "healthy"
    counters = result.health["counters"]
    assert isinstance(counters, dict)
    assert counters["events_written"] == 4
    assert counters["artifacts_written"] == 2
    assert counters["redactions_applied"] == 1
    assert (
        recorder.attempt_dir.joinpath("events.jsonl").read_bytes()
        == recorder.attempt_dir.joinpath("journal.jsonl").read_bytes()
    )
    assert ContractValidator().validate_attempt(recorder.attempt_dir).valid
    if os.name != "nt":
        for path in recorder.attempt_dir.rglob("*"):
            expected = 0o700 if path.is_dir() else 0o600
            assert path.stat().st_mode & 0o777 == expected


def test_native_json_omits_accounting_but_retains_activity(
    tmp_path: Path,
) -> None:
    recorder = TraceRecorder(_config(tmp_path))
    _record_complete_trace(recorder)
    recorder.finalize()
    native = read_jsonl(
        recorder.attempt_dir / "native" / "index.jsonl",
        allow_torn_final_line=False,
    ).records[0]
    artifact = native["artifact"]
    assert isinstance(artifact, dict)
    content = json.loads((recorder.attempt_dir / str(artifact["path"])).read_bytes())

    assert content["command"] == "ls -1"
    assert content["status"] == "completed"
    assert "usage" not in content


def test_sensitive_values_never_reach_trace_storage(tmp_path: Path) -> None:
    recorder = TraceRecorder(_config(tmp_path))
    secret = "sk-" + "s" * 24
    assert (
        recorder.record_event(
            event_type="trace.note",
            event_family="trace",
            phase="instant",
            status="completed",
            span_id="span-note",
            origin={
                "component": "example-adapter",
                "capture_method": "derived",
            },
            timing={"fidelity": "not_available"},
            payload={
                "api_key": secret,
                "command": f"API_KEY={secret}",
                "message": "cost.py remains observable",
            },
        )
        is not None
    )
    _record_complete_trace(recorder)

    result = recorder.finalize()

    assert result.validation.valid
    assert result.manifest["complete"] is True
    for path in recorder.attempt_dir.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()


def test_invalid_runtime_event_is_isolated_and_reported(
    tmp_path: Path,
) -> None:
    recorder = TraceRecorder(_config(tmp_path))

    assert (
        recorder.record_event(
            event_type="invalid",
            event_family="trace",
            phase="instant",
            status="completed",
            span_id="span-invalid",
            origin={
                "component": "example-adapter",
                "capture_method": "derived",
            },
            timing={"fidelity": "not_available"},
        )
        is None
    )
    _record_complete_trace(recorder)
    result = recorder.finalize()

    assert result.validation.valid
    assert result.manifest["complete"] is False
    assert result.health["status"] == "degraded"
    issues = result.health["issues"]
    assert isinstance(issues, list)
    assert [issue["code"] for issue in issues if isinstance(issue, dict)] == [
        "event.invalid"
    ]


def test_runtime_journal_failure_does_not_escape_agent_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = TraceRecorder(_config(tmp_path))

    def fail_append(self: JsonlJournal, record: JsonObject) -> None:
        raise TraceStorageError("synthetic write failure")

    with monkeypatch.context() as patch:
        patch.setattr(JsonlJournal, "append", fail_append)
        assert (
            recorder.record_event(
                event_type="trace.note",
                event_family="trace",
                phase="instant",
                status="completed",
                span_id="span-failed-write",
                origin={
                    "component": "example-adapter",
                    "capture_method": "derived",
                },
                timing={"fidelity": "not_available"},
            )
            is None
        )

    _record_complete_trace(recorder)
    result = recorder.finalize()

    assert result.validation.valid
    assert result.health["status"] == "degraded"


def test_finalizer_recovers_torn_event_journal(tmp_path: Path) -> None:
    config = _config(tmp_path)
    recorder = TraceRecorder(config)
    _record_complete_trace(recorder)
    recorder.close()
    with (recorder.attempt_dir / "journal.jsonl").open("ab") as stream:
        stream.write(b'{"schema_version":"benchmark-trace/v1"')

    recovered = TraceRecorder.recover(config)
    result = recovered.finalize()

    assert result.validation.valid
    assert result.manifest["complete"] is False
    assert result.health["status"] == "degraded"
    assert result.health["finalization"] == "recovered"
    counters = result.health["counters"]
    assert isinstance(counters, dict)
    assert counters["dropped_events"] == 1
    assert (
        len(
            read_jsonl(
                recovered.attempt_dir / "events.jsonl",
                allow_torn_final_line=False,
            ).records
        )
        == 4
    )


def test_fresh_process_recovery_is_reported_without_a_torn_line(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    recorder = TraceRecorder(config)
    _record_complete_trace(recorder)
    recorder.close()

    recovered = TraceRecorder.recover(config)
    result = recovered.finalize()

    assert result.validation.valid
    assert result.manifest["complete"] is False
    assert result.health["status"] == "degraded"
    assert result.health["finalization"] == "recovered"
    counters = result.health["counters"]
    assert isinstance(counters, dict)
    assert counters["dropped_events"] == 0


def test_finalizer_rejects_malformed_complete_journal_record(
    tmp_path: Path,
) -> None:
    recorder = TraceRecorder(_config(tmp_path))
    _record_complete_trace(recorder)
    recorder.close()
    with (recorder.attempt_dir / "journal.jsonl").open("ab") as stream:
        stream.write(b"{not-json}\n")

    with pytest.raises(TraceFinalizationError):
        recorder.finalize()


def test_validator_detects_artifact_tampering(tmp_path: Path) -> None:
    recorder = TraceRecorder(_config(tmp_path))
    _record_complete_trace(recorder)
    recorder.finalize()
    events = read_jsonl(
        recorder.attempt_dir / "events.jsonl",
        allow_torn_final_line=False,
    ).records
    artifacts = events[2]["artifacts"]
    assert isinstance(artifacts, list)
    reference = artifacts[0]
    assert isinstance(reference, dict)
    artifact = recorder.attempt_dir / str(reference["path"])
    artifact.write_bytes(b"tampered\n")

    report = ContractValidator().validate_attempt(recorder.attempt_dir)

    assert not report.valid
    assert {"artifact.digest_mismatch", "artifact.size_mismatch"} <= {
        issue.code for issue in report.issues
    }


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission contract")
def test_validator_detects_insecure_file_permissions(tmp_path: Path) -> None:
    recorder = TraceRecorder(_config(tmp_path))
    _record_complete_trace(recorder)
    recorder.finalize()
    (recorder.attempt_dir / "manifest.json").chmod(0o644)

    report = ContractValidator().validate_attempt(recorder.attempt_dir)

    assert not report.valid
    assert "storage.permissions" in {issue.code for issue in report.issues}


def test_validator_detects_span_parent_cycles(tmp_path: Path) -> None:
    recorder = TraceRecorder(_config(tmp_path))
    _record_complete_trace(recorder)
    recorder.finalize()
    events = list(
        read_jsonl(
            recorder.attempt_dir / "events.jsonl",
            allow_torn_final_line=False,
        ).records
    )
    events[0]["parent_span_id"] = "span-shell"
    events[3]["parent_span_id"] = "span-shell"
    records = tuple(events)
    _write_jsonl(recorder.attempt_dir / "events.jsonl", records)
    _write_jsonl(recorder.attempt_dir / "journal.jsonl", records)

    report = ContractValidator().validate_attempt(recorder.attempt_dir)

    assert not report.valid
    assert "spans.parent_cycle" in {issue.code for issue in report.issues}


def test_validator_detects_span_parent_cycles_in_degraded_traces(
    tmp_path: Path,
) -> None:
    recorder = TraceRecorder(_config(tmp_path))
    assert (
        recorder.record_event(
            event_type="invalid",
            event_family="trace",
            phase="instant",
            status="completed",
            span_id="span-invalid",
            origin={
                "component": "example-adapter",
                "capture_method": "derived",
            },
            timing={"fidelity": "not_available"},
        )
        is None
    )
    _record_complete_trace(recorder)
    recorder.finalize()
    events = list(
        read_jsonl(
            recorder.attempt_dir / "events.jsonl",
            allow_torn_final_line=False,
        ).records
    )
    events[0]["parent_span_id"] = "span-shell"
    events[3]["parent_span_id"] = "span-shell"
    records = tuple(events)
    _write_jsonl(recorder.attempt_dir / "events.jsonl", records)
    _write_jsonl(recorder.attempt_dir / "journal.jsonl", records)

    report = ContractValidator().validate_attempt(recorder.attempt_dir)

    assert not report.valid
    assert "spans.parent_cycle" in {issue.code for issue in report.issues}


def test_validator_rejects_unresolved_event_relations(tmp_path: Path) -> None:
    recorder = TraceRecorder(_config(tmp_path))
    assert (
        recorder.record_event(
            event_type="trace.note",
            event_family="trace",
            phase="instant",
            status="completed",
            span_id="span-note",
            origin={
                "component": "example-adapter",
                "capture_method": "derived",
            },
            timing={"fidelity": "not_available"},
            relations=(
                {
                    "type": "derived_from",
                    "event_id": "event-that-does-not-exist",
                },
            ),
        )
        is not None
    )
    _record_complete_trace(recorder)

    result = recorder.finalize()

    assert not result.validation.valid
    assert "events.unresolved_relation" in {
        issue.code for issue in result.validation.issues
    }


def test_timeline_reconstructs_nesting_duration_and_command(
    tmp_path: Path,
) -> None:
    recorder = TraceRecorder(_config(tmp_path))
    _record_complete_trace(recorder)
    recorder.finalize()

    timeline = build_timeline(recorder.attempt_dir)
    rendered = render_timeline(timeline)

    assert [entry.sequence for entry in timeline] == [1, 2, 3, 4]
    assert timeline[1].depth == 1
    assert timeline[2].duration_ms == 12.5
    assert timeline[1].detail == "ls -1"
    assert timeline[-1].relative_ms == 2000
    assert "shell.start start/started actor=agent-root ls -1" in rendered
    assert "shell.end end/completed actor=agent-root duration=12.500ms" in rendered


def test_concurrent_event_submission_preserves_contiguous_sequence(
    tmp_path: Path,
) -> None:
    recorder = TraceRecorder(_config(tmp_path))

    def record_note(index: int) -> None:
        assert (
            recorder.record_event(
                event_type="trace.note",
                event_family="trace",
                phase="instant",
                status="completed",
                span_id=f"span-note-{index}",
                origin={
                    "component": "example-adapter",
                    "capture_method": "derived",
                },
                timing={"fidelity": "not_available"},
                payload={"index": index},
            )
            is not None
        )

    threads = [
        threading.Thread(target=record_note, args=(index,)) for index in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    _record_complete_trace(recorder)

    result = recorder.finalize()
    events = read_jsonl(
        recorder.attempt_dir / "events.jsonl",
        allow_torn_final_line=False,
    ).records

    assert result.validation.valid
    assert [event["sequence"] for event in events] == list(range(1, 25))
    assert len({str(event["event_id"]) for event in events}) == 24


def test_preflight_rejects_incomplete_capability_matrix_without_writing(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    invalid = TraceConfig(
        attempt_dir=config.attempt_dir,
        identity=config.identity,
        producer=config.producer,
        provenance=config.provenance,
        execution=config.execution,
        capabilities=config.capabilities[:-1],
    )

    with pytest.raises(TraceInitializationError):
        TraceRecorder(invalid)

    assert not invalid.attempt_dir.exists()


def test_preflight_rejects_an_existing_attempt_directory(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.attempt_dir.mkdir(parents=True)
    marker = config.attempt_dir / "owned-by-user"
    marker.write_text("preserve")

    with pytest.raises(TraceInitializationError):
        TraceRecorder(config)

    assert marker.read_text() == "preserve"


def test_run_index_uses_contract_digest_and_safe_attempt_path(
    tmp_path: Path,
) -> None:
    recorder = TraceRecorder(_config(tmp_path))
    _record_complete_trace(recorder)
    recorder.finalize()
    validator = ContractValidator()
    run_root = tmp_path / "run"
    document: JsonObject = {
        "schema_version": "benchmark-trace/v1",
        "contract": {
            "name": "benchmark-trace",
            "version": "1.0.0",
            "schema_digest": validator.schema_digest,
        },
        "run_id": recorder.identity.run_id,
        "benchmark": recorder.identity.benchmark,
        "framework": recorder.identity.framework,
        "created_at": "2026-01-02T03:04:05Z",
        "finalized_at": "2026-01-02T03:04:07Z",
        "selection": {
            "strategy": "explicit_ids",
            "requested_count": 1,
            "instance_ids": [recorder.identity.instance_id],
        },
        "attempts": [
            {
                "trace_id": recorder.identity.trace_id,
                "instance_id": recorder.identity.instance_id,
                "attempt": recorder.identity.attempt,
                "path": (
                    "instances/"
                    f"{encode_instance_id(recorder.identity.instance_id)}/attempt-1"
                ),
                "status": "completed",
            }
        ],
    }

    path = write_run_index(run_root, document, validator=validator)

    assert _read_json(path) == document
    assert not validator.validate_document("run.schema.json", _read_json(path))
    assert validator.validate_run(run_root).valid
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
        assert run_root.stat().st_mode & 0o777 == 0o700

    attempts = document["attempts"]
    assert isinstance(attempts, list)
    attempt = attempts[0]
    assert isinstance(attempt, dict)
    attempt["status"] = "failed"
    path.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    path.chmod(0o600)
    assert "run.status_mismatch" in {
        issue.code for issue in validator.validate_run(run_root).issues
    }


def test_finalization_is_idempotent(tmp_path: Path) -> None:
    recorder = TraceRecorder(_config(tmp_path))
    _record_complete_trace(recorder)

    first = recorder.finalize()
    second = recorder.finalize()

    assert second == first
