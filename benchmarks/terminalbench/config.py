"""Terminal-Bench configuration defaults."""

from benchmarks.utils.harbor_compat import get_harbor_dataset


TERMINAL_BENCH_TASK_COUNT = 89

# Default inference settings (only include values actually used by argparse)
INFER_DEFAULTS = {
    "dataset": get_harbor_dataset("terminalbench"),
    "output_dir": "./evaluation_outputs",
    "num_workers": 1,
    "n_attempts": 1,
    "n_limit": 1,
    "max_retries": 0,
    "environment": "docker",
    "enable_delegation": True,
}

# Harbor configuration defaults
HARBOR_DEFAULTS = {
    # Harbor executable
    "harbor_executable": "harbor",
    # Default agent name for openhands-sdk
    "agent_name": "openhands-sdk",
    # Repo-local Harbor adapter that enables native OpenHands subagents.
    "delegating_agent_name": "benchmark_agents.openhands_harbor:DelegatingOpenHandsSDK",
    # Match the shared benchmark coordinator budget and sampling configuration.
    "max_iterations": 24,
    "temperature": 0.1,
}
