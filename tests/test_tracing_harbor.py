from __future__ import annotations

import json
from pathlib import Path

from benchmarks.tracing import (
    CONTRACT_VERSION,
    ContractValidator,
    TraceConfig,
    TraceIdentity,
    TraceProducer,
    TraceRecorder,
    attempt_directory,
    create_trace_run,
)
from benchmarks.tracing.adapters.openhands import (
    OpenHandsTraceAdapter,
    openhands_capabilities,
)
from benchmarks.tracing.harbor import (
    HarborTraceHarness,
    allocate_harbor_trace_attempt,
    attach_harbor_agentsight_profile,
    create_harbor_trace_run,
    finalize_harbor_trace_run,
    promote_harbor_trace_attempt,
    start_harbor_agentsight_profile,
    trace_agent_timeout_from_trial_config,
    trace_container_image_from_trial_config,
    trace_instance_id_from_trial_config,
    trace_instance_ids_from_job,
)


def test_harbor_harness_resolves_any_benchmark_identity(tmp_path: Path) -> None:
    job = tmp_path / "jobs" / "custom"
    job.mkdir(parents=True)
    (job / "lock.json").write_text(
        json.dumps({"trials": [{"task": {"name": "custom/task-a"}}]}),
        encoding="utf-8",
    )
    run = create_trace_run(
        tmp_path / "traces",
        benchmark="custom-harbor-benchmark",
        framework="openhands",
    )
    harness = HarborTraceHarness(
        jobs_dir=tmp_path / "jobs",
        job_name="custom",
        selected_instance_ids=None,
        expected_instance_count=1,
        expected_attempts_per_instance=1,
        selection_strategy="full_dataset",
    )

    selection = harness.resolve_selection(run, ())

    assert selection.instance_ids == ("task-a",)
    assert selection.strategy == "full_dataset"


def test_harbor_attempt_allocator_is_scoped_per_instance(tmp_path: Path) -> None:
    run = create_harbor_trace_run(tmp_path / "traces", "terminal-bench-2.1")

    def allocate(instance_id: str) -> int:
        return allocate_harbor_trace_attempt(
            trace_root=run.root,
            instance_id=instance_id,
            agent_timeout_seconds=900,
            container_image="example/task:latest",
        ).attempt

    assert [allocate("task-a"), allocate("task-a"), allocate("task-b")] == [1, 2, 1]


def test_harbor_job_lock_preserves_resolved_task_order(tmp_path: Path) -> None:
    job = tmp_path / "jobs" / "run"
    job.mkdir(parents=True)
    (job / "lock.json").write_text(
        json.dumps(
            {
                "trials": [
                    {"task": {"name": "terminal-bench/task-b"}},
                    {"task": {"name": "terminal-bench/task-a"}},
                    {"task": {"name": "terminal-bench/task-b"}},
                ]
            }
        )
    )

    assert trace_instance_ids_from_job(tmp_path / "jobs", "run") == [
        "task-b",
        "task-a",
    ]


def test_harbor_bridge_promotes_and_indexes_a_native_attempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        """
[agent]
timeout_sec = 900

[environment]
docker_image = "example/task:latest"
""".strip()
        + "\n"
    )
    logs_dir = tmp_path / "job" / "trial" / "agent"
    logs_dir.mkdir(parents=True)
    (logs_dir.parent / "config.json").write_text(
        json.dumps(
            {
                "task": {
                    "name": "terminal-bench/task-a",
                    "ref": "sha256:" + "a" * 64,
                },
                "timeout_multiplier": 2,
            }
        )
    )
    monkeypatch.setattr(
        "benchmarks.tracing.harbor._resolved_task_path",
        lambda _task: task_dir,
    )

    run = create_harbor_trace_run(tmp_path / "traces", "terminal-bench-2.1")
    attempt = allocate_harbor_trace_attempt(
        trace_root=run.root,
        instance_id=trace_instance_id_from_trial_config(logs_dir),
        agent_timeout_seconds=trace_agent_timeout_from_trial_config(logs_dir),
        container_image=trace_container_image_from_trial_config(logs_dir),
    )
    identity = TraceIdentity.create(
        run_id=run.id,
        benchmark=run.benchmark,
        framework="openhands",
        instance_id=attempt.instance_id,
        attempt=attempt.attempt,
    )
    adapter = OpenHandsTraceAdapter(
        TraceRecorder(
            TraceConfig(
                attempt_dir=attempt_directory(
                    logs_dir / attempt.container_root.name,
                    attempt.instance_id,
                    attempt.attempt,
                ),
                identity=identity,
                producer=TraceProducer(
                    name="benchmarks.tracing.adapters.openhands",
                    version=CONTRACT_VERSION,
                ),
                provenance={
                    "benchmark": {
                        "name": "OpenHands-benchmarks",
                        "revision": "a" * 40,
                    },
                    "framework": {
                        "name": "OpenHands SDK",
                        "revision": "b" * 40,
                    },
                    "adapter": {
                        "name": "benchmarks.tracing.adapters.openhands",
                        "revision": "a" * 40,
                    },
                },
                execution={
                    "model": "openrouter/qwen/qwen3-coder-next",
                    "evaluation_workers": 1,
                    "inference_timeout_seconds": (attempt.agent_timeout_seconds),
                    "benchmark_retries": 0,
                    "provider_attempts": 1,
                },
                capabilities=openhands_capabilities(
                    delegation_enabled=True,
                    condenser_enabled=False,
                    browser_enabled=False,
                    container_enabled=True,
                ),
            )
        ),
        session_id="task-a__trial__agent",
        delegation_enabled=True,
        condenser_enabled=False,
        browser_enabled=False,
        container_enabled=True,
    )
    adapter.start()
    adapter.container_observed({"image": attempt.container_image})
    adapter.start_session()
    adapter.finish("completed")
    profiler = start_harbor_agentsight_profile(
        logs_dir=logs_dir,
        trace_run_id=run.id,
        benchmark=run.benchmark,
        framework="openhands",
        attempt=attempt,
        docker_session_id="task-a__trial",
        tls_python_path="/opt/openhands-sdk-venv/bin/python",
        env={"BENCHMARK_AGENTSIGHT": "off"},
    )
    profiler.finish()
    assert profiler.target.capture_tls is True
    attached_profile = attach_harbor_agentsight_profile(
        logs_dir=logs_dir,
        attempt=attempt,
        profiler=profiler,
    )

    promoted = promote_harbor_trace_attempt(
        logs_dir=logs_dir,
        trace_root=run.root,
        attempt=attempt,
    )
    run_index = finalize_harbor_trace_run(
        trace_root=run.root,
        run_id=run.id,
        benchmark=run.benchmark,
        framework="openhands",
        created_at=run.created_at,
        selected_instance_ids=["task-a"],
        expected_instance_count=1,
        expected_attempts_per_instance=1,
        selection_strategy="explicit_ids",
    )

    assert attempt.agent_timeout_seconds == 1800
    assert attempt.container_image == "example/task:latest"
    assert promoted.is_dir()
    assert attached_profile.name == "agentsight"
    profile = json.loads(
        (promoted / "profiles" / "agentsight" / "profile.json").read_text()
    )
    assert profile["correlation"] == {
        "runId": run.id,
        "benchmark": run.benchmark,
        "framework": "openhands",
        "instanceId": "task-a",
        "attempt": 1,
    }
    assert run_index == run.root / "run.json"
    assert ContractValidator().validate_run(run.root).valid
    events = [
        json.loads(line)
        for line in (promoted / "events.jsonl").read_text().splitlines()
    ]
    session_events = [
        event
        for event in events
        if event["event_type"] in {"agent.session_start", "agent.session_end"}
    ]
    assert len({event["parent_span_id"] for event in session_events}) == 1
    assert session_events[0]["parent_span_id"].startswith("openhands-execution-")
