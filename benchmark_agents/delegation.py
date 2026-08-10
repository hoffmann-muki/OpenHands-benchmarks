"""Shared multi-agent topology for benchmark inference."""

BENCHMARK_AGENT_TOPOLOGY = "supervisor-delegation"
BENCHMARK_SINGLE_AGENT_TOPOLOGY = "single-agent"
BENCHMARK_NAVIGATOR_AGENT = "benchmark-navigator"
BENCHMARK_PATCHER_AGENT = "benchmark-patcher"
BENCHMARK_REVIEWER_AGENT = "benchmark-reviewer"


def benchmark_delegation_instructions() -> str:
    """Return the required blocking delegation workflow for coding benchmarks."""
    return """

## Required multi-agent workflow

You are the supervising agent and remain responsible for the final repository state.
Use the task tool to perform these blocking delegations sequentially:

1. Delegate investigation to a fresh `benchmark-navigator` subagent. Ask it to locate the relevant code, constraints, likely root cause, and a practical verification strategy without changing files.
2. Delegate implementation to a fresh `benchmark-patcher` subagent. Give it the original task and the investigation result, and require it to make the concrete changes and run focused checks in the shared workspace.
3. Delegate independent review to a fresh `benchmark-reviewer` subagent. Give it the original task and prior results, and require it to inspect the final state, run feasible checks, and correct small clear defects when necessary.

Do not resume or reuse a subagent for a different phase. Do not run delegations in the background. After all three return, reconcile their findings, resolve any remaining issue yourself, and provide the final answer.
""".rstrip()


def terminal_benchmark_delegation_instructions() -> str:
    """Return the required blocking topology for Terminal-Bench tasks."""
    return """

## Required multi-agent workflow

You are the supervising agent and remain responsible for the final environment state.
Use the task tool to perform these blocking delegations sequentially:

1. Delegate investigation to a fresh `benchmark-navigator` subagent. Ask it to inspect the environment, constraints, relevant files, likely root cause, and a practical verification strategy without changing state.
2. Delegate execution to a fresh `benchmark-patcher` subagent. Give it the original task and the investigation result, and require it to perform the concrete work and run focused checks in the shared environment.
3. Delegate independent verification to a fresh `benchmark-reviewer` subagent. Give it the original task and prior results, and require it to inspect the final state, run feasible checks, and correct small clear defects when necessary.

Do not resume or reuse a subagent for a different phase. Do not run delegations in the background. After all three return, reconcile their findings, resolve any remaining issue yourself, and provide the final answer.
""".rstrip()


def terminal_benchmark_single_agent_instructions() -> str:
    """Return the complete workflow contract for one Terminal-Bench agent."""
    return """

## Required single-agent workflow

You are the sole coding agent and remain responsible for the final environment state.
Do not delegate or attempt to create subagents. Work through the task yourself:

1. Inspect the environment, task constraints, relevant files, and a practical verification strategy before changing state.
2. Implement the smallest complete solution directly in the shared environment.
3. Run focused verification, inspect the final state, and correct any clear defect you find.

Conclude with the work performed, verification commands and outcomes, and any residual risk. The environment changes—not prose—are the benchmark answer.
""".rstrip()


def benchmark_single_agent_instructions() -> str:
    """Return the complete workflow contract for a native single agent."""
    return """

## Required single-agent workflow

You are the sole coding agent and remain responsible for the final repository state.
Do not delegate or attempt to create subagents. Work through the task yourself:

1. Investigate the issue, relevant code, constraints, and practical verification strategy before editing.
2. Implement the smallest complete fix directly in the shared workspace.
3. Run focused verification, inspect the final diff, and correct any clear defect you find.

Conclude with the changed paths, verification commands and outcomes, and any residual risk.
""".rstrip()


def append_benchmark_delegation_instructions(
    instruction: str,
    *,
    enabled: bool,
) -> str:
    """Append the benchmark topology only when delegation is enabled."""
    workflow = (
        benchmark_delegation_instructions()
        if enabled
        else benchmark_single_agent_instructions()
    )
    return f"{instruction.rstrip()}\n{workflow}\n"
