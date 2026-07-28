import json
from collections.abc import Sequence
from pathlib import Path

from benchmarks.tracing import (
    ContractValidator,
    TraceHarnessAdapter,
    TraceRun,
    TraceSelection,
    create_trace_run,
    finalize_trace_run,
)
from benchmarks.tracing.openhands import (
    OpenHandsTraceSettings,
    create_openhands_attempt_trace,
)


class _CustomHarness:
    prepared = False

    def prepare_finalization(self, run: TraceRun) -> None:
        assert run.benchmark == "custom-benchmark"
        self.prepared = True

    def resolve_selection(
        self,
        run: TraceRun,
        observed_instance_ids: Sequence[str],
    ) -> TraceSelection:
        assert run.framework == "openhands"
        assert observed_instance_ids == ("custom-instance",)
        return TraceSelection(
            instance_ids=("custom-instance",),
            strategy="explicit_ids",
        )


def test_generic_coordinator_supports_an_arbitrary_benchmark(
    tmp_path: Path,
) -> None:
    run = create_trace_run(
        tmp_path / "traces",
        benchmark="custom-benchmark",
        framework="openhands",
    )
    adapter = create_openhands_attempt_trace(
        run=run,
        instance_id="custom-instance",
        attempt=1,
        session_id="custom-session",
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
    )
    adapter._recorder.report_issue(
        "trace.synthetic_warning",
        "Synthetic observability warning",
        severity="warning",
    )
    adapter.finish("failed", error_message="Synthetic agent failure")
    custom_harness = _CustomHarness()
    harness: TraceHarnessAdapter = custom_harness

    path = finalize_trace_run(run, harness)

    assert custom_harness.prepared
    assert ContractValidator().validate_run(run.root).valid
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["benchmark"] == "custom-benchmark"
    assert document["framework"] == "openhands"
    assert document["selection"]["instance_ids"] == ["custom-instance"]
    assert document["attempts"][0]["status"] == "failed"


def test_generic_coordinator_recovers_an_interrupted_attempt(
    tmp_path: Path,
) -> None:
    run = create_trace_run(
        tmp_path / "traces",
        benchmark="custom-benchmark",
        framework="openhands",
    )
    adapter = create_openhands_attempt_trace(
        run=run,
        instance_id="custom-instance",
        attempt=1,
        session_id="custom-session",
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
    )
    adapter._recorder.close()

    path = finalize_trace_run(run, _CustomHarness())

    assert ContractValidator().validate_run(run.root).valid
    assert path == run.root / "run.json"
    health = json.loads(
        (
            run.root / "instances" / "custom-instance" / "attempt-1" / "health.json"
        ).read_text(encoding="utf-8")
    )
    assert health["status"] == "degraded"
    assert health["finalization"] == "recovered"
