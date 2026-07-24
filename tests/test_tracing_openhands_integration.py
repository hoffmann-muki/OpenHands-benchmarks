import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from benchmarks.swebench.run_infer import (
    SWEBenchEvaluation,
    _OpenHandsAttemptTrace,
)
from benchmarks.swebenchpro.run_infer import SWEBenchProEvaluation
from benchmarks.tracing import ContractValidator
from benchmarks.tracing.storage import read_jsonl
from benchmarks.utils.args_parser import add_trace_dir_argument, get_parser
from benchmarks.utils.critics import PassCritic
from benchmarks.utils.evaluation import Evaluation, _is_timeout_failure
from benchmarks.utils.models import (
    EvalInstance,
    EvalMetadata,
    EvalOutput,
)
from openhands.sdk import LLM
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.workspace import RemoteWorkspace


def _metadata(
    tmp_path: Path,
    *,
    trace_dir: Path | None = None,
    trace_run_id: str | None = None,
    max_retries: int = 0,
) -> EvalMetadata:
    if trace_dir is not None:
        trace_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    return EvalMetadata(
        llm=LLM(model="test-model"),
        dataset="princeton-nlp/SWE-bench_Verified",
        dataset_split="test",
        max_iterations=24,
        inference_timeout=1800,
        eval_output_dir=str(tmp_path / "outputs"),
        trace_dir=str(trace_dir) if trace_dir is not None else None,
        trace_run_id=trace_run_id,
        trace_created_at=(
            "2026-07-24T00:00:00.000Z"
            if trace_dir is not None or trace_run_id is not None
            else None
        ),
        details={
            "agent_source_commit": "a" * 40,
            "benchmark_source_commit": "b" * 40,
            "evaluation_timeout": 3600,
        },
        eval_limit=1,
        selected_instances_file="selected-instances.txt",
        max_retries=max_retries,
        enable_delegation=True,
        enable_condenser=True,
        critic=PassCritic(),
    )


def _instance() -> EvalInstance:
    return EvalInstance(
        id="django__django-12345",
        data={"repo": "django/django", "base_commit": "a" * 40},
    )


def test_swe_trace_lifecycle_builds_a_valid_run_without_inference(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces" / "trace-run-integration"
    evaluator = SWEBenchEvaluation(
        metadata=_metadata(
            tmp_path,
            trace_dir=trace_root,
            trace_run_id="trace-run-integration",
        ),
        num_workers=1,
        instance_timeout=2400,
    )
    instance = _instance()

    context = evaluator._create_trace_context(instance, 1, 0)

    assert isinstance(context, _OpenHandsAttemptTrace)
    context.adapter.start_session()
    trace_result = evaluator._finish_trace_context(context, "completed", None)
    evaluator._finalize_trace_run([instance])

    assert trace_result is not None
    assert trace_result["valid"] is True
    assert ContractValidator().validate_run(trace_root).valid
    run = json.loads((trace_root / "run.json").read_text(encoding="utf-8"))
    assert run["selection"] == {
        "strategy": "explicit_ids",
        "requested_count": 1,
        "instance_ids": [instance.id],
    }
    assert run["attempts"][0]["status"] == "completed"
    manifest = json.loads(
        (context.attempt_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["execution"]["inference_timeout_seconds"] == 1800
    assert manifest["execution"]["evaluation_timeout_seconds"] == 3600
    events = read_jsonl(
        context.attempt_dir / "events.jsonl",
        allow_torn_final_line=False,
    ).records
    assert [event["event_type"] for event in events] == [
        "instance.start",
        "attempt.start",
        "agent.session_start",
        "agent.session_end",
        "attempt.end",
        "instance.end",
    ]


def test_workspace_failure_does_not_invent_an_agent_session(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces" / "trace-run-workspace-failure"
    evaluator = SWEBenchEvaluation(
        metadata=_metadata(
            tmp_path,
            trace_dir=trace_root,
            trace_run_id="trace-run-workspace-failure",
        )
    )

    context = evaluator._create_trace_context(_instance(), 1, 0)

    assert isinstance(context, _OpenHandsAttemptTrace)
    evaluator._finish_trace_context(context, "failed", RuntimeError("workspace failed"))
    events = read_jsonl(
        context.attempt_dir / "events.jsonl",
        allow_torn_final_line=False,
    ).records
    assert [event["event_type"] for event in events] == [
        "instance.start",
        "attempt.start",
        "attempt.end",
        "instance.end",
    ]


def test_internal_retry_numbers_map_to_distinct_trace_attempts(
    tmp_path: Path,
) -> None:
    evaluator = SWEBenchEvaluation(
        metadata=_metadata(
            tmp_path,
            trace_dir=tmp_path / "traces" / "trace-run-attempts",
            trace_run_id="trace-run-attempts",
            max_retries=2,
        )
    )

    context = evaluator._create_trace_context(_instance(), 2, 1)

    assert isinstance(context, _OpenHandsAttemptTrace)
    assert context.adapter.identity.attempt == 5
    evaluator._finish_trace_context(context, "failed", RuntimeError("synthetic"))


def test_swe_bench_pro_uses_its_own_trace_identity(tmp_path: Path) -> None:
    evaluator = SWEBenchProEvaluation(metadata=_metadata(tmp_path))

    assert evaluator.trace_benchmark_name() == "swe-bench-pro"


def test_trace_preflight_failure_stops_before_workspace_and_provider_work(
    tmp_path: Path,
) -> None:
    instance = _instance()

    class PreflightFailureEvaluation(Evaluation):
        workspace_requested: bool = False

        def prepare_instances(self) -> list[EvalInstance]:
            return [instance]

        def prepare_workspace(
            self,
            instance: EvalInstance,
            resource_factor: int = 1,
            forward_env: list[str] | None = None,
        ) -> RemoteWorkspace:
            self.workspace_requested = True
            raise AssertionError("workspace must not be requested")

        def evaluate_instance(
            self,
            instance: EvalInstance,
            workspace: RemoteWorkspace,
        ) -> EvalOutput:
            raise AssertionError("inference must not begin")

        def _create_trace_context(
            self,
            instance: EvalInstance,
            critic_attempt: int,
            retry_count: int,
        ) -> Any:
            raise RuntimeError("trace preflight failed")

    evaluator = PreflightFailureEvaluation(metadata=_metadata(tmp_path, max_retries=3))

    output, failure_category = evaluator._execute_single_attempt(
        instance=instance,
        eval_span_ctx=None,
        critic_attempt=1,
        resource_factor=1,
        retry_count=0,
        max_retries=3,
        runtime_failure_count=0,
        runtime_runs=[],
    )

    assert output is not None
    assert output.error is not None
    assert output.test_result["trace"] == {
        "status": "initialization_failed",
        "error": "RuntimeError",
    }
    assert failure_category is None
    assert evaluator.workspace_requested is False


def test_wrapped_native_timeout_is_classified_without_message_matching() -> None:
    error = ConversationRunError(
        uuid.uuid4(),
        TimeoutError("synthetic native timeout"),
    )

    assert _is_timeout_failure(error)
    assert not _is_timeout_failure(RuntimeError("text says timeout but is not one"))


def test_trace_metadata_requires_a_complete_run_identity(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="configured together"):
        _metadata(tmp_path, trace_dir=tmp_path / "traces")
    with pytest.raises(ValidationError, match="configured together"):
        _metadata(tmp_path, trace_run_id="trace-run-incomplete")


def test_trace_cli_is_opt_in_only_for_wired_entrypoints(tmp_path: Path) -> None:
    parser = get_parser(add_llm_config=False)
    assert not hasattr(parser.parse_args([]), "trace_dir")

    add_trace_dir_argument(parser)
    args = parser.parse_args(["--trace-dir", str(tmp_path / "traces")])

    assert args.trace_dir == str(tmp_path / "traces")
