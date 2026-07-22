"""Fixed-budget OpenHands subagents for SWE benchmark parity."""

from benchmark_agents.delegation import (
    BENCHMARK_NAVIGATOR_AGENT,
    BENCHMARK_PATCHER_AGENT,
    BENCHMARK_REVIEWER_AGENT,
)
from openhands.sdk.hooks import HookConfig, HookDefinition, HookMatcher
from openhands.sdk.subagent import (
    AgentDefinition,
    agent_definition_to_factory,
    get_agent_factory,
    register_agent_if_absent,
)


BENCHMARK_NAVIGATOR_MAX_ITERATIONS = 10
BENCHMARK_PATCHER_MAX_ITERATIONS = 18
BENCHMARK_REVIEWER_MAX_ITERATIONS = 12

_BLOCK_TASK_RESUME_COMMAND = (
    "python -S -c 'import json,sys; event=json.load(sys.stdin); "
    'resume=(event.get("tool_input") or {}).get("resume"); '
    'print("SWE benchmark task resumption is disabled", file=sys.stderr) '
    "if resume else None; raise SystemExit(2 if resume else 0)'"
)


def swe_benchmark_hook_config() -> HookConfig:
    """Block subagent resume calls that bypass fixed per-phase iteration caps."""
    return HookConfig(
        pre_tool_use=[
            HookMatcher(
                matcher="task",
                hooks=[
                    HookDefinition(
                        name="block-swe-task-resume",
                        command=_BLOCK_TASK_RESUME_COMMAND,
                        timeout=5,
                    )
                ],
            )
        ]
    )


def swe_benchmark_agent_definitions() -> tuple[AgentDefinition, ...]:
    """Return the fixed SWE agent team used for cross-framework parity."""
    return (
        AgentDefinition(
            name=BENCHMARK_NAVIGATOR_AGENT,
            description=(
                "Read-only navigator for issue triage, relevant files, constraints, "
                "and verification strategy."
            ),
            model="inherit",
            tools=["terminal"],
            max_iteration_per_run=BENCHMARK_NAVIGATOR_MAX_ITERATIONS,
            system_prompt=(
                "Investigate the task without modifying the workspace. Identify the "
                "likely root cause, relevant source and test files, constraints, "
                "risks, and feasible verification commands. Return a concise handoff "
                "with exact paths and rationale."
            ),
        ),
        AgentDefinition(
            name=BENCHMARK_PATCHER_AGENT,
            description="Implements the concrete fix for the benchmark task.",
            model="inherit",
            tools=["terminal", "file_editor", "task_tracker"],
            max_iteration_per_run=BENCHMARK_PATCHER_MAX_ITERATIONS,
            system_prompt=(
                "Use the original task and navigator handoff to make the smallest "
                "complete fix. Run feasible focused verification, inspect the final "
                "changes, and report changed paths, commands, outcomes, and risks. "
                "Do not use hidden tests, gold patches, or external solution artifacts."
            ),
        ),
        AgentDefinition(
            name=BENCHMARK_REVIEWER_AGENT,
            description=(
                "Reviews the final patch, verification evidence, and residual risk."
            ),
            model="inherit",
            tools=["terminal", "file_editor", "task_tracker"],
            max_iteration_per_run=BENCHMARK_REVIEWER_MAX_ITERATIONS,
            system_prompt=(
                "Inspect the final changes against the original task and verification "
                "evidence. Make only small, clearly necessary corrections and rerun "
                "focused checks when feasible. Report concrete findings and residual "
                "risks without broadening scope."
            ),
        ),
    )


def register_swe_benchmark_agents() -> list[str]:
    """Idempotently register the fixed-budget SWE subagents."""
    definitions = swe_benchmark_agent_definitions()
    registered: list[str] = []
    for definition in definitions:
        was_registered = register_agent_if_absent(
            name=definition.name,
            factory_func=agent_definition_to_factory(definition),
            description=definition,
        )
        if was_registered:
            registered.append(definition.name)
            continue
        if get_agent_factory(definition.name).definition != definition:
            raise RuntimeError(
                f"Conflicting definition already registered for {definition.name}"
            )
    return registered
