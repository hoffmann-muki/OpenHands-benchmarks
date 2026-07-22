"""Regression tests for the benchmark supervisor topology."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from benchmark_agents import openhands_harbor_runner
from benchmark_agents.delegation import (
    BENCHMARK_AGENT_TOPOLOGY,
    BENCHMARK_NAVIGATOR_AGENT,
    BENCHMARK_PATCHER_AGENT,
    BENCHMARK_REVIEWER_AGENT,
    append_benchmark_delegation_instructions,
    benchmark_delegation_instructions,
    terminal_benchmark_delegation_instructions,
)
from benchmark_agents.swe_agents import (
    BENCHMARK_NAVIGATOR_MAX_ITERATIONS,
    BENCHMARK_PATCHER_MAX_ITERATIONS,
    BENCHMARK_REVIEWER_MAX_ITERATIONS,
    swe_benchmark_agent_definitions,
    swe_benchmark_hook_config,
)
from benchmark_agents.swe_task import FreshOnlyTaskToolSet
from openhands.sdk.hooks.manager import HookManager


def test_coding_delegation_requires_three_fresh_blocking_phases() -> None:
    instructions = benchmark_delegation_instructions()

    assert BENCHMARK_AGENT_TOPOLOGY == "supervisor-delegation"
    assert f"`{BENCHMARK_NAVIGATOR_AGENT}`" in instructions
    assert f"`{BENCHMARK_PATCHER_AGENT}`" in instructions
    assert f"`{BENCHMARK_REVIEWER_AGENT}`" in instructions
    assert "Do not run delegations in the background" in instructions
    assert "remain responsible for the final repository state" in instructions


def test_swe_agent_iteration_budgets_match_opencode_phases() -> None:
    definitions = {
        definition.name: definition for definition in swe_benchmark_agent_definitions()
    }

    assert definitions[BENCHMARK_NAVIGATOR_AGENT].max_iteration_per_run == (
        BENCHMARK_NAVIGATOR_MAX_ITERATIONS
    )
    assert definitions[BENCHMARK_PATCHER_AGENT].max_iteration_per_run == (
        BENCHMARK_PATCHER_MAX_ITERATIONS
    )
    assert definitions[BENCHMARK_REVIEWER_AGENT].max_iteration_per_run == (
        BENCHMARK_REVIEWER_MAX_ITERATIONS
    )
    assert "file_editor" not in definitions[BENCHMARK_NAVIGATOR_AGENT].tools
    assert "file_editor" in definitions[BENCHMARK_PATCHER_AGENT].tools
    assert "file_editor" in definitions[BENCHMARK_REVIEWER_AGENT].tools


def test_swe_task_resume_is_blocked_without_blocking_fresh_tasks(
    tmp_path: Path,
) -> None:
    manager = HookManager(
        config=swe_benchmark_hook_config(),
        working_dir=str(tmp_path),
    )

    fresh_allowed, fresh_results = manager.run_pre_tool_use(
        "task",
        {"resume": None},
    )
    resume_allowed, resume_results = manager.run_pre_tool_use(
        "task",
        {"resume": "task_00000001"},
    )

    assert fresh_allowed is True
    assert fresh_results[0].exit_code == 0
    assert resume_allowed is False
    assert resume_results[0].exit_code == 2
    assert "resumption is disabled" in resume_results[0].stderr


def test_swe_task_tool_rejects_resume_without_running_a_hook() -> None:
    task_tool = FreshOnlyTaskToolSet.create(conv_state=cast(Any, SimpleNamespace()))[0]
    action = task_tool.action_from_arguments(
        {
            "prompt": "Continue the previous phase",
            "subagent_type": BENCHMARK_NAVIGATOR_AGENT,
            "resume": "task_00000001",
        }
    )

    observation = task_tool(action)

    assert observation.is_error is True
    assert "resumption is disabled" in observation.text
    assert FreshOnlyTaskToolSet.name == "swe_benchmark_task_tool_set"


def test_terminal_delegation_uses_the_same_supervisor_topology() -> None:
    instructions = terminal_benchmark_delegation_instructions()

    assert "`code-explorer`" in instructions
    assert instructions.count("`general-purpose`") == 2
    assert "shared environment" in instructions
    assert "Do not run delegations in the background" in instructions


def test_delegation_instructions_are_only_appended_when_enabled() -> None:
    original = "Solve the task.\n"

    enabled = append_benchmark_delegation_instructions(original, enabled=True)
    disabled = append_benchmark_delegation_instructions(original, enabled=False)

    assert "Required multi-agent workflow" in enabled
    assert disabled == original


def test_harbor_runner_adds_task_tool_persistence_and_combined_metrics(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}
    usage = SimpleNamespace(
        prompt_tokens=120,
        completion_tokens=30,
        cache_read_tokens=15,
    )
    combined_metrics = SimpleNamespace(
        accumulated_token_usage=usage,
        accumulated_cost=0.25,
    )

    def agent(*args, **kwargs):
        captured["agent_kwargs"] = kwargs
        return SimpleNamespace()

    def conversation(*args, **kwargs):
        captured["conversation_kwargs"] = kwargs
        return SimpleNamespace(
            conversation_stats=SimpleNamespace(
                get_combined_metrics=lambda: combined_metrics
            )
        )

    def build_trajectory(events, metrics, model_name, **kwargs):
        captured["trajectory_metrics"] = metrics
        return {"final_metrics": metrics}

    module = SimpleNamespace(
        Agent=agent,
        Conversation=conversation,
        Tool=lambda *, name: SimpleNamespace(name=name),
        build_trajectory=build_trajectory,
    )
    monkeypatch.setattr(
        openhands_harbor_runner,
        "register_builtins_agents",
        lambda *, enable_browser: [],
    )

    openhands_harbor_runner.configure_delegation(module)
    module.Agent(tools=[])
    module.Conversation(workspace="/workspace")
    result = module.build_trajectory([], {}, "test-model")

    tools = captured["agent_kwargs"]["tools"]
    assert [tool.name for tool in tools] == [openhands_harbor_runner.TaskToolSet.name]
    assert captured["conversation_kwargs"]["persistence_dir"] == (
        openhands_harbor_runner.CONVERSATION_LOG_DIR
    )
    assert captured["trajectory_metrics"] == {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "cached_tokens": 15,
        "cost_usd": 0.25,
    }
    assert result["final_metrics"]["cost_usd"] == 0.25
