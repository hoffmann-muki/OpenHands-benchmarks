import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.tracing import (
    DirectTraceHarness,
    TraceConfig,
    TraceIdentity,
    TraceProducer,
    TraceRecorder,
    TraceSelection,
    create_trace_run,
    finalize_trace_run,
)
from benchmarks.tracing.adapters.openhands import openhands_capabilities
from benchmarks.tracing.cli import build_parser, main
from benchmarks.tracing.openhands import (
    OpenHandsTraceSettings,
    create_openhands_attempt_trace,
)


def _run_trace(
    tmp_path: Path,
    *,
    benchmark: str = "custom-benchmark",
    instance_id: str = "owner/project__issue-1",
    attempts: int = 1,
) -> Path:
    run = create_trace_run(
        tmp_path / "traces",
        benchmark=benchmark,
        framework="openhands",
    )
    for attempt in range(1, attempts + 1):
        create_openhands_attempt_trace(
            run=run,
            instance_id=instance_id,
            attempt=attempt,
            session_id=f"session-{benchmark}-{attempt}",
            settings=OpenHandsTraceSettings(
                benchmark_revision="a" * 40,
                framework_revision="b" * 40,
                model="test/model",
                evaluation_workers=1,
                inference_timeout_seconds=30,
                benchmark_retries=0,
                delegation_enabled=False,
                condenser_enabled=False,
            ),
        ).finish("completed")
    finalize_trace_run(
        run,
        DirectTraceHarness(
            TraceSelection(
                instance_ids=(instance_id,),
                strategy="explicit_ids",
            )
        ),
    )
    return run.root


def test_cli_contains_analysis_commands_but_no_collector() -> None:
    parser = build_parser()

    for command in (
        "validate",
        "inspect",
        "summarize",
        "compare",
        "gate",
        "render",
        "recover",
    ):
        assert parser.parse_args([command, "/trace"]).command == command
    with pytest.raises(SystemExit):
        parser.parse_args(["collect", "/trace"])


def test_trace_cli_bootstrap_avoids_benchmark_runtime_side_effects() -> None:
    environment = {**os.environ, "BENCHMARK_TRACE_CLI": "1"}

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from benchmarks.tracing.cli import build_parser; "
            "print(build_parser().prog)",
        ],
        cwd=Path(__file__).parent.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == "benchmark-trace\n"
    assert result.stderr == ""


def test_recover_finalizes_an_interrupted_preflight_attempt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    attempt = tmp_path / "interrupted-attempt"
    recorder = TraceRecorder(
        TraceConfig(
            attempt_dir=attempt,
            identity=TraceIdentity.create(
                run_id="trace-run-recovery",
                benchmark="custom-benchmark",
                framework="openhands",
                instance_id="owner/project__issue-1",
            ),
            producer=TraceProducer(name="test-recorder", version="1.0.0"),
            provenance={
                "benchmark": {"name": "test", "revision": "a" * 40},
                "framework": {"name": "OpenHands", "revision": "b" * 40},
                "adapter": {"name": "test", "revision": "a" * 40},
            },
            execution={
                "model": "test/model",
                "evaluation_workers": 1,
                "inference_timeout_seconds": 30,
                "benchmark_retries": 0,
                "provider_attempts": 1,
            },
            capabilities=openhands_capabilities(
                delegation_enabled=False,
                condenser_enabled=False,
                browser_enabled=False,
            ),
        )
    )
    recorder.close()

    assert main(["recover", str(attempt), "--format", "json"]) == 0
    document = json.loads(capsys.readouterr().out)

    assert document["valid"] is True
    assert document["complete"] is False
    assert document["health"] == "degraded"
    assert document["finalization"] == "recovered"


def test_validate_accepts_a_run_and_discovers_a_trace_base(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = _run_trace(tmp_path, instance_id="instance-a")
    second = _run_trace(tmp_path, instance_id="instance-b")

    assert main(["validate", str(first), "--format", "json"]) == 0
    direct = json.loads(capsys.readouterr().out)
    assert direct["valid"] is True
    assert direct["traces"][0]["kind"] == "run"

    assert main(["validate", str(tmp_path / "traces"), "--format", "json"]) == 0
    discovered = json.loads(capsys.readouterr().out)
    assert discovered["valid"] is True
    assert {item["path"] for item in discovered["traces"]} == {
        str(first),
        str(second),
    }


def test_validate_reports_contract_failures_with_exit_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _run_trace(tmp_path)
    health = next(root.glob("instances/*/attempt-*/health.json"))
    health.unlink()

    assert main(["validate", str(root), "--format", "json"]) == 1
    document = json.loads(capsys.readouterr().out)
    assert document["valid"] is False
    assert any(
        issue["code"] == "file.missing" for issue in document["traces"][0]["issues"]
    )


def test_inspect_reports_identity_configuration_and_capabilities(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _run_trace(tmp_path)
    attempt = next(root.glob("instances/*/attempt-*"))

    assert main(["inspect", str(attempt), "--format", "json"]) == 0
    document = json.loads(capsys.readouterr().out)

    assert document["kind"] == "attempt"
    assert document["identity"]["benchmark"] == "custom-benchmark"
    assert document["execution"]["provider_attempts"] == 1
    assert document["health"]["agent_outcome_affected"] is False
    assert len(document["capabilities"]) == 18

    assert main(["inspect", str(root), "--format", "json"]) == 0
    run = json.loads(capsys.readouterr().out)
    assert run["producer"]["consistent"] is True
    assert run["provenance"]["consistent"] is True
    assert run["execution"]["consistent"] is True
    assert run["execution"]["variants"][0]["provider_attempts"] == 1
    assert run["health"] == {
        "finalization": {"clean": 1},
        "status": {"healthy": 1},
    }
    assert len(run["capabilities"]) == 18


def test_summarize_aggregates_normalized_activity_without_accounting(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _run_trace(tmp_path)

    assert main(["summarize", str(root), "--format", "json"]) == 0
    document = json.loads(capsys.readouterr().out)

    assert document["coverage"] == {
        "attempts": 1,
        "attempts_per_instance": {"owner/project__issue-1": 1},
        "complete_attempts": 1,
        "finalization": {"clean": 1},
        "health": {"healthy": 1},
        "instances": 1,
    }
    assert document["events"]["total"] > 0
    assert document["activity"]["tool_calls"] == 0
    assert document["agent_configuration"] == {
        "consistent": True,
        "variants": [
            {
                "coordination_mode": "framework_native",
                "delegation_enabled": False,
                "delegation_sequence": [],
                "sequence_enforcement": "prompt_guided",
            }
        ],
    }
    assert document["storage"]["dropped_events"] == 0
    assert document["storage"]["native_artifacts"] == 0
    assert document["storage"]["native_chunks"] == 0
    assert document["storage"]["native_artifact_bytes"] == 0
    assert "tokens" not in json.dumps(document).lower()
    assert "cost" not in json.dumps(document).lower()


def test_compare_checks_controlled_run_compatibility(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = _run_trace(tmp_path)
    second = _run_trace(tmp_path)

    assert main(["compare", str(first), str(second), "--format", "json"]) == 0
    comparable = json.loads(capsys.readouterr().out)
    assert comparable["comparable"] is True
    assert comparable["configuration_comparable"] is True
    assert comparable["analysis_ready"] is True
    assert all(comparable["checks"].values())

    different = _run_trace(tmp_path, benchmark="other-benchmark")
    assert main(["compare", str(first), str(different), "--format", "json"]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["comparable"] is False
    assert comparison["checks"]["benchmark"] is False

    extra_attempt = _run_trace(tmp_path, attempts=2)
    assert main(["compare", str(first), str(extra_attempt), "--format", "json"]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["configuration_comparable"] is False
    assert comparison["comparable"] is False
    assert comparison["checks"]["attempt_topology"] is False


def test_gate_combines_contract_validation_and_comparability(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = _run_trace(tmp_path)
    second = _run_trace(tmp_path)

    assert main(["gate", str(first), str(second), "--format", "json"]) == 0
    passing = json.loads(capsys.readouterr().out)
    assert passing["passed"] is True
    assert passing["contract_conformant"] is True

    different = _run_trace(tmp_path, benchmark="other-benchmark")
    assert main(["gate", str(first), str(different), "--format", "json"]) == 1
    failing = json.loads(capsys.readouterr().out)
    assert failing["passed"] is False
    assert failing["comparison"]["checks"]["benchmark"] is False


def test_render_emits_text_and_machine_readable_timelines(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _run_trace(tmp_path)

    assert main(["render", str(root)]) == 0
    text = capsys.readouterr().out
    assert "instance.start" in text
    assert "attempt.end" in text
    assert "actor=session-custom-benchmark" in text

    assert main(["render", str(root), "--format", "json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["timelines"][0]["instance_id"] == "owner/project__issue-1"
    assert document["timelines"][0]["entries"][0]["relative_ms"] == 0


def test_render_includes_artifact_references_and_opt_in_contents(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = (
        Path(__file__).parent.parent
        / "benchmarks"
        / "tracing"
        / "conformance"
        / "v1"
        / "valid"
    )
    attempt = tmp_path / "valid"
    shutil.copytree(source, attempt)
    for path in attempt.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    attempt.chmod(0o700)

    assert main(["render", str(attempt / "manifest.json"), "--format", "json"]) == 0
    references = json.loads(capsys.readouterr().out)
    artifact = next(
        entry["artifacts"][0]
        for entry in references["timelines"][0]["entries"]
        if entry["artifacts"]
    )
    assert artifact["role"] == "tool.stdout"
    assert "content" not in artifact

    assert (
        main(
            [
                "render",
                str(attempt / "manifest.json"),
                "--include-artifacts",
                "--format",
                "json",
            ]
        )
        == 0
    )
    included = json.loads(capsys.readouterr().out)
    artifact = next(
        entry["artifacts"][0]
        for entry in included["timelines"][0]["entries"]
        if entry["artifacts"]
    )
    assert artifact["content"] == "README.md\nsrc\n"


def test_unknown_contract_version_is_rejected_without_collection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _run_trace(tmp_path)
    run_path = root / "run.json"
    document = json.loads(run_path.read_text(encoding="utf-8"))
    document["schema_version"] = "benchmark-trace/v999"
    run_path.write_text(json.dumps(document), encoding="utf-8")

    assert main(["inspect", str(root)]) == 2
    assert "Unsupported trace schema version" in capsys.readouterr().err
