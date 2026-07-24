#!/usr/bin/env python3
"""Bootstrap Harbor's OpenHands SDK runner with native task delegation."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

from openhands.sdk.subagent import (
    AgentDefinition,
    agent_definition_to_factory,
    get_agent_factory,
    register_agent_if_absent,
)
from openhands.tools.task import TaskToolSet


BASE_RUNNER_PATH = Path("/installed-agent/run_agent_base.py")
CONVERSATION_LOG_DIR = Path("/logs/agent/conversation")
BENCHMARK_NAVIGATOR_AGENT = "benchmark-navigator"
BENCHMARK_PATCHER_AGENT = "benchmark-patcher"
BENCHMARK_REVIEWER_AGENT = "benchmark-reviewer"
BENCHMARK_NAVIGATOR_MAX_ITERATIONS = 10
BENCHMARK_PATCHER_MAX_ITERATIONS = 18
BENCHMARK_REVIEWER_MAX_ITERATIONS = 12
ENABLE_DELEGATION = True
TRACE_CONFIG_ENV = "OPENHANDS_BENCHMARK_TRACE_CONFIG"


def terminal_benchmark_agent_definitions() -> tuple[AgentDefinition, ...]:
    """Return the fixed-budget team shared by all Terminal-Bench runners."""
    return (
        AgentDefinition(
            name=BENCHMARK_NAVIGATOR_AGENT,
            description=(
                "Read-only Terminal-Bench navigator for environment inspection "
                "and execution planning."
            ),
            model="inherit",
            tools=["terminal"],
            max_iteration_per_run=BENCHMARK_NAVIGATOR_MAX_ITERATIONS,
            system_prompt=(
                "Investigate the task without changing state. Identify relevant "
                "files, environment constraints, likely failure points, and "
                "feasible verification commands. Return a concise, evidence-backed "
                "handoff for the patcher."
            ),
        ),
        AgentDefinition(
            name=BENCHMARK_PATCHER_AGENT,
            description=(
                "Executes the concrete Terminal-Bench task in the shared environment."
            ),
            model="inherit",
            tools=["terminal", "file_editor", "task_tracker"],
            max_iteration_per_run=BENCHMARK_PATCHER_MAX_ITERATIONS,
            system_prompt=(
                "Use the original task and navigator handoff to perform the smallest "
                "complete set of changes in the shared environment. Run focused "
                "verification and report actions, commands, outcomes, and remaining "
                "risk. Do not delegate."
            ),
        ),
        AgentDefinition(
            name=BENCHMARK_REVIEWER_AGENT,
            description=(
                "Independently verifies the Terminal-Bench result and makes small "
                "corrections."
            ),
            model="inherit",
            tools=["terminal", "file_editor", "task_tracker"],
            max_iteration_per_run=BENCHMARK_REVIEWER_MAX_ITERATIONS,
            system_prompt=(
                "Inspect the original task, current environment, and prior handoffs. "
                "Run feasible checks and make only small, clearly necessary "
                "corrections. Report concrete findings and residual risk. Do not "
                "delegate."
            ),
        ),
    )


def register_terminal_benchmark_agents() -> list[str]:
    """Idempotently register the fixed-budget Terminal-Bench subagents."""
    registered = []
    for definition in terminal_benchmark_agent_definitions():
        if register_agent_if_absent(
            name=definition.name,
            factory_func=agent_definition_to_factory(definition),
            description=definition,
        ):
            registered.append(definition.name)
            continue
        if get_agent_factory(definition.name).definition != definition:
            raise RuntimeError(
                f"Conflicting definition already registered for {definition.name}"
            )
    return registered


def load_base_runner(path: Path = BASE_RUNNER_PATH) -> ModuleType:
    spec = importlib.util.spec_from_file_location("harbor_openhands_sdk_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Harbor OpenHands runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_benchmark_runner(
    module: Any,
    enable_delegation: bool,
    trace_adapter: Any | None = None,
) -> None:
    """Add single-attempt LLM calls, delegation, and aggregate accounting."""
    if enable_delegation:
        register_terminal_benchmark_agents()

    original_llm = module.LLM
    original_agent = module.Agent
    original_conversation = module.Conversation
    original_build_trajectory = module.build_trajectory
    captured: dict[str, Any] = {}

    def single_attempt_llm(*args: Any, **kwargs: Any) -> Any:
        kwargs["num_retries"] = 1
        kwargs["caching_prompt"] = False
        return original_llm(*args, **kwargs)

    def delegating_agent(*args: Any, **kwargs: Any) -> Any:
        tools = list(kwargs.get("tools", []))
        if not any(getattr(tool, "name", None) == TaskToolSet.name for tool in tools):
            tools.append(module.Tool(name=TaskToolSet.name))
        kwargs["tools"] = tools
        return original_agent(*args, **kwargs)

    def persistent_conversation(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("persistence_dir", CONVERSATION_LOG_DIR)
        if trace_adapter is not None:
            callbacks = list(kwargs.get("callbacks", []))
            callbacks.append(trace_adapter.callback)
            kwargs["callbacks"] = callbacks
        conversation = original_conversation(*args, **kwargs)
        captured["conversation"] = conversation
        if trace_adapter is not None:
            trace_adapter.start_session()
        return conversation

    def build_trajectory(
        events: list[dict[str, Any]],
        llm_metrics: dict[str, Any],
        model_name: str,
        system_prompt: str | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        conversation = captured.get("conversation")
        if conversation is not None:
            combined = conversation.conversation_stats.get_combined_metrics()
            usage = combined.accumulated_token_usage
            llm_metrics = {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "cached_tokens": usage.cache_read_tokens if usage else 0,
                "cost_usd": combined.accumulated_cost,
            }
        return original_build_trajectory(
            events,
            llm_metrics,
            model_name,
            system_prompt=system_prompt,
            tool_definitions=tool_definitions,
        )

    setattr(module, "LLM", single_attempt_llm)
    if enable_delegation:
        setattr(module, "Agent", delegating_agent)
    setattr(module, "Conversation", persistent_conversation)
    setattr(module, "build_trajectory", build_trajectory)


def create_trace_adapter() -> Any | None:
    """Create the native adapter before the runner can issue a provider request."""

    raw = os.environ.get(TRACE_CONFIG_ENV)
    if not raw:
        return None
    config = json.loads(raw)
    required = {
        "run_id",
        "benchmark",
        "instance_id",
        "attempt",
        "container_root",
        "benchmark_revision",
        "framework_revision",
        "model",
        "evaluation_workers",
        "inference_timeout_seconds",
        "benchmark_retries",
        "session_id",
    }
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("OpenHands Harbor trace configuration is invalid")

    from benchmarks.tracing import (
        CONTRACT_VERSION,
        TraceConfig,
        TraceIdentity,
        TraceProducer,
        TraceRecorder,
        attempt_directory,
    )
    from benchmarks.tracing.adapters.openhands import (
        OpenHandsTraceAdapter,
        openhands_capabilities,
    )

    root = Path(config["container_root"])
    identity = TraceIdentity.create(
        run_id=config["run_id"],
        benchmark=config["benchmark"],
        framework="openhands",
        instance_id=config["instance_id"],
        attempt=int(config["attempt"]),
    )
    adapter = OpenHandsTraceAdapter(
        TraceRecorder(
            TraceConfig(
                attempt_dir=attempt_directory(
                    root,
                    config["instance_id"],
                    int(config["attempt"]),
                ),
                identity=identity,
                producer=TraceProducer(
                    name="benchmarks.tracing.adapters.openhands",
                    version=CONTRACT_VERSION,
                ),
                provenance={
                    "benchmark": {
                        "name": "OpenHands-benchmarks",
                        "revision": config["benchmark_revision"],
                    },
                    "framework": {
                        "name": "OpenHands SDK",
                        "revision": config["framework_revision"],
                    },
                    "adapter": {
                        "name": "benchmarks.tracing.adapters.openhands",
                        "revision": config["benchmark_revision"],
                    },
                    "harness": {
                        "name": "Harbor",
                        "revision": config.get("harbor_version", "unknown"),
                    },
                    **(
                        {"agent_image": config["container_image"]}
                        if config.get("container_image")
                        else {}
                    ),
                },
                execution={
                    "model": config["model"],
                    "evaluation_workers": int(config["evaluation_workers"]),
                    "inference_timeout_seconds": float(
                        config["inference_timeout_seconds"]
                    ),
                    "benchmark_retries": int(config["benchmark_retries"]),
                    "provider_attempts": 1,
                },
                capabilities=openhands_capabilities(
                    delegation_enabled=ENABLE_DELEGATION,
                    condenser_enabled=False,
                    browser_enabled=False,
                    completion_logs_enabled=False,
                    harness_enabled=True,
                    container_enabled=True,
                    evaluator_enabled=False,
                ),
            )
        ),
        session_id=config["session_id"],
        delegation_enabled=ENABLE_DELEGATION,
        condenser_enabled=False,
        browser_enabled=False,
        completion_logs_enabled=False,
        harness_enabled=True,
        container_enabled=True,
        evaluator_enabled=False,
    )
    adapter.start()
    adapter.start_harness(
        {
            "name": "harbor",
            "version": config.get("harbor_version", "unknown"),
            "phase": "agent",
        }
    )
    adapter.container_observed(
        {
            "session_id": config["session_id"],
            **(
                {"image": config["container_image"]}
                if config.get("container_image")
                else {}
            ),
        }
    )
    return adapter


def main() -> None:
    module = load_base_runner()
    trace_adapter = create_trace_adapter()
    configure_benchmark_runner(
        module,
        enable_delegation=ENABLE_DELEGATION,
        trace_adapter=trace_adapter,
    )
    error = None
    try:
        module.main()
    except BaseException as exception:
        error = exception
        raise
    finally:
        if trace_adapter is not None:
            try:
                trace_adapter.finish(
                    "completed" if error is None else "failed",
                    error_message=str(error) if error is not None else None,
                )
            except Exception:
                # The host bridge reports the missing/partial trace separately.
                # Finalization must not change a completed Harbor agent result.
                pass


if __name__ == "__main__":
    main()
