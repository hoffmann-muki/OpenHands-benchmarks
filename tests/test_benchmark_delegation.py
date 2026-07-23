"""Regression tests for the benchmark supervisor topology."""

from types import SimpleNamespace
from typing import Any

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
)


def test_coding_delegation_uses_native_prompt_directed_phases() -> None:
    instructions = benchmark_delegation_instructions()

    assert BENCHMARK_AGENT_TOPOLOGY == "supervisor-delegation"
    assert f"`{BENCHMARK_NAVIGATOR_AGENT}`" in instructions
    assert f"`{BENCHMARK_PATCHER_AGENT}`" in instructions
    assert f"`{BENCHMARK_REVIEWER_AGENT}`" in instructions
    assert "Do not run delegations in the background" in instructions
    assert "Do not resume or reuse a subagent" in instructions
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


def test_terminal_delegation_uses_the_same_supervisor_topology() -> None:
    instructions = terminal_benchmark_delegation_instructions()

    assert f"`{BENCHMARK_NAVIGATOR_AGENT}`" in instructions
    assert f"`{BENCHMARK_PATCHER_AGENT}`" in instructions
    assert f"`{BENCHMARK_REVIEWER_AGENT}`" in instructions
    assert "shared environment" in instructions
    assert "Do not run delegations in the background" in instructions


def test_terminal_agent_iteration_budgets_match_opencode_phases() -> None:
    definitions = {
        definition.name: definition
        for definition in openhands_harbor_runner.terminal_benchmark_agent_definitions()
    }

    assert definitions[BENCHMARK_NAVIGATOR_AGENT].max_iteration_per_run == 10
    assert definitions[BENCHMARK_PATCHER_AGENT].max_iteration_per_run == 18
    assert definitions[BENCHMARK_REVIEWER_AGENT].max_iteration_per_run == 12
    assert "file_editor" not in definitions[BENCHMARK_NAVIGATOR_AGENT].tools
    assert "file_editor" in definitions[BENCHMARK_PATCHER_AGENT].tools
    assert "file_editor" in definitions[BENCHMARK_REVIEWER_AGENT].tools


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

    def llm(*args, **kwargs):
        captured["llm_kwargs"] = kwargs
        return SimpleNamespace()

    module = SimpleNamespace(
        LLM=llm,
        Agent=agent,
        Conversation=conversation,
        Tool=lambda *, name: SimpleNamespace(name=name),
        build_trajectory=build_trajectory,
    )
    monkeypatch.setattr(
        openhands_harbor_runner,
        "register_terminal_benchmark_agents",
        lambda: [],
    )

    openhands_harbor_runner.configure_benchmark_runner(
        module,
        enable_delegation=True,
    )
    module.LLM(model="test-model", num_retries=9)
    module.Agent(tools=[])
    module.Conversation(workspace="/workspace")
    result = module.build_trajectory([], {}, "test-model")

    tools = captured["agent_kwargs"]["tools"]
    assert [tool.name for tool in tools] == [openhands_harbor_runner.TaskToolSet.name]
    assert captured["llm_kwargs"]["num_retries"] == 1
    assert captured["llm_kwargs"]["caching_prompt"] is False
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


def test_single_agent_harbor_runner_still_limits_provider_attempts(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def llm(*args, **kwargs):
        captured["llm_kwargs"] = kwargs
        return SimpleNamespace()

    def agent(*args, **kwargs):
        return SimpleNamespace()

    def unexpected_registration():
        raise AssertionError("delegation must stay disabled")

    module = SimpleNamespace(
        LLM=llm,
        Agent=agent,
        Conversation=lambda *args, **kwargs: SimpleNamespace(),
        build_trajectory=lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        openhands_harbor_runner,
        "register_terminal_benchmark_agents",
        unexpected_registration,
    )

    openhands_harbor_runner.configure_benchmark_runner(
        module,
        enable_delegation=False,
    )
    module.LLM(model="test-model", num_retries=9)

    assert module.Agent is agent
    assert captured["llm_kwargs"]["num_retries"] == 1
    assert captured["llm_kwargs"]["caching_prompt"] is False
