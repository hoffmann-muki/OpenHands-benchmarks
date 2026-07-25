"""Reusable OpenHands framework-adapter construction for any benchmark."""

from __future__ import annotations

import re
from dataclasses import dataclass

from benchmarks.tracing.adapters.openhands import (
    OpenHandsTraceAdapter,
    openhands_capabilities,
)
from benchmarks.tracing.constants import CONTRACT_VERSION
from benchmarks.tracing.integration import TraceRun
from benchmarks.tracing.models import TraceConfig, TraceIdentity, TraceProducer
from benchmarks.tracing.recorder import TraceRecorder, attempt_directory


@dataclass(frozen=True, slots=True)
class OpenHandsTraceSettings:
    """Framework and execution facts supplied by a benchmark runner."""

    benchmark_revision: str
    framework_revision: str
    model: str
    evaluation_workers: int
    inference_timeout_seconds: int | float
    benchmark_retries: int
    delegation_enabled: bool
    condenser_enabled: bool
    evaluation_timeout_seconds: int | float | None = None
    browser_enabled: bool = False
    completion_logs_enabled: bool = False
    container_enabled: bool = False
    evaluator_enabled: bool = False
    harness_name: str | None = None
    harness_revision: str | None = None
    agent_image: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.benchmark_revision):
            raise ValueError("Tracing requires an exact benchmark revision")
        if not re.fullmatch(r"[0-9a-f]{40}", self.framework_revision):
            raise ValueError("Tracing requires an exact OpenHands revision")
        if not self.model:
            raise ValueError("Tracing requires a model identity")
        if self.evaluation_workers < 1 or self.inference_timeout_seconds <= 0:
            raise ValueError("Tracing requires positive worker and timeout values")
        if self.benchmark_retries < 0:
            raise ValueError("Tracing benchmark retries cannot be negative")
        if (self.harness_name is None) != (self.harness_revision is None):
            raise ValueError("Tracing harness name and revision must be paired")


def create_openhands_attempt_trace(
    *,
    run: TraceRun,
    instance_id: str,
    attempt: int,
    session_id: str,
    settings: OpenHandsTraceSettings,
) -> OpenHandsTraceAdapter:
    """Create and start one native OpenHands adapter before provider work."""

    if run.framework != "openhands":
        raise ValueError("OpenHands trace adapter requires framework='openhands'")
    if (
        not instance_id
        or not session_id
        or isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt < 1
    ):
        raise ValueError("OpenHands trace attempt identity is invalid")
    identity = TraceIdentity.create(
        run_id=run.id,
        benchmark=run.benchmark,
        framework=run.framework,
        instance_id=instance_id,
        attempt=attempt,
    )
    adapter = OpenHandsTraceAdapter(
        TraceRecorder(
            TraceConfig(
                attempt_dir=attempt_directory(run.root, instance_id, attempt),
                identity=identity,
                producer=TraceProducer(
                    name="benchmarks.tracing.adapters.openhands",
                    version=CONTRACT_VERSION,
                ),
                provenance={
                    "benchmark": {
                        "name": "OpenHands-benchmarks",
                        "revision": settings.benchmark_revision,
                    },
                    "framework": {
                        "name": "OpenHands SDK",
                        "revision": settings.framework_revision,
                    },
                    "adapter": {
                        "name": "benchmarks.tracing.adapters.openhands",
                        "revision": settings.benchmark_revision,
                    },
                    **(
                        {
                            "harness": {
                                "name": settings.harness_name,
                                "revision": settings.harness_revision,
                            }
                        }
                        if settings.harness_name and settings.harness_revision
                        else {}
                    ),
                    **(
                        {"agent_image": settings.agent_image}
                        if settings.agent_image
                        else {}
                    ),
                },
                execution={
                    "model": settings.model,
                    "evaluation_workers": settings.evaluation_workers,
                    "inference_timeout_seconds": (settings.inference_timeout_seconds),
                    **(
                        {
                            "evaluation_timeout_seconds": (
                                settings.evaluation_timeout_seconds
                            )
                        }
                        if settings.evaluation_timeout_seconds is not None
                        else {}
                    ),
                    "benchmark_retries": settings.benchmark_retries,
                    "provider_attempts": 1,
                },
                capabilities=openhands_capabilities(
                    delegation_enabled=settings.delegation_enabled,
                    condenser_enabled=settings.condenser_enabled,
                    browser_enabled=settings.browser_enabled,
                    completion_logs_enabled=settings.completion_logs_enabled,
                    container_enabled=settings.container_enabled,
                    evaluator_enabled=settings.evaluator_enabled,
                ),
            )
        ),
        session_id=session_id,
        delegation_enabled=settings.delegation_enabled,
        condenser_enabled=settings.condenser_enabled,
        browser_enabled=settings.browser_enabled,
        completion_logs_enabled=settings.completion_logs_enabled,
        container_enabled=settings.container_enabled,
        evaluator_enabled=settings.evaluator_enabled,
    )
    adapter.start()
    return adapter
