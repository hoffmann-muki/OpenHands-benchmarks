"""SWE-Bench Pro benchmark configuration."""

from pathlib import Path

from benchmarks.swebenchpro import constants


DEFAULT_SMOKE_INSTANCES_FILE = Path(__file__).with_name("smoke_instances.txt")
DEFAULT_MAX_ITERATIONS = 24
DEFAULT_INFERENCE_TIMEOUT_SECONDS = 30 * 60
# The outer evaluator also covers non-LLM workspace setup and teardown.
DEFAULT_INSTANCE_TIMEOUT_GRACE_SECONDS = 10 * 60


INFER_DEFAULTS = {
    "dataset": "ScaleAI/SWE-bench_Pro",
    "split": "test",
    "workspace": "docker",
    "num_workers": 1,
    "n_limit": 1,
    "select": str(DEFAULT_SMOKE_INSTANCES_FILE),
    "max_iterations": DEFAULT_MAX_ITERATIONS,
    "inference_timeout": DEFAULT_INFERENCE_TIMEOUT_SECONDS,
    "n_critic_runs": 1,
    "max_retries": 0,
    "enable_delegation": True,
    "enable_condenser": True,
    "condenser_max_size": 240,
    "condenser_keep_first": 2,
}

EVAL_DEFAULTS = {
    "dataset": "ScaleAI/SWE-bench_Pro",
    "split": "test",
    "workers": 1,
    "dockerhub_username": constants.DEFAULT_DOCKERHUB_USERNAME,
    "use_local_docker": True,
    "block_network": False,
}
