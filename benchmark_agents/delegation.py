"""Shared multi-agent topology for benchmark inference."""

BENCHMARK_AGENT_TOPOLOGY = "supervisor-delegation"
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

1. Delegate investigation to a fresh `code-explorer` subagent. Ask it to inspect the environment, constraints, relevant files, likely root cause, and a practical verification strategy without changing state.
2. Delegate execution to a fresh `general-purpose` subagent. Give it the original task and the investigation result, and require it to perform the concrete work and run focused checks in the shared environment.
3. Delegate independent verification to another fresh `general-purpose` subagent. Give it the original task and prior results, and require it to inspect the final state, run feasible checks, and correct small clear defects when necessary.

Do not resume or reuse a subagent for a different phase. Do not run delegations in the background. After all three return, reconcile their findings, resolve any remaining issue yourself, and provide the final answer.
""".rstrip()


def append_benchmark_delegation_instructions(
    instruction: str,
    *,
    enabled: bool,
) -> str:
    """Append the benchmark topology only when delegation is enabled."""
    if not enabled:
        return instruction
    return f"{instruction.rstrip()}\n{benchmark_delegation_instructions()}\n"
