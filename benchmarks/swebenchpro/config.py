"""SWE-Bench Pro benchmark configuration."""

from pathlib import Path

from benchmarks.swebenchpro import constants


DEFAULT_SMOKE_INSTANCES_FILE = Path(__file__).with_name("smoke_instances.txt")
DATASET_NAME = "ScaleAI/SWE-bench_Pro"
DATASET_SPLIT = "test"
DATASET_REVISION = "7ab5114912baf22bb098818e604c02fe7ad2c11f"
DEFAULT_MAX_ITERATIONS = 24
DEFAULT_INFERENCE_TIMEOUT_SECONDS = 30 * 60
# The outer evaluator also covers non-LLM workspace setup and teardown.
DEFAULT_INSTANCE_TIMEOUT_GRACE_SECONDS = 10 * 60


INFER_DEFAULTS = {
    "dataset": DATASET_NAME,
    "split": DATASET_SPLIT,
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
    "dataset": DATASET_NAME,
    "split": DATASET_SPLIT,
    "workers": 1,
    "dockerhub_username": constants.DEFAULT_DOCKERHUB_USERNAME,
    "use_local_docker": True,
    "block_network": False,
    "timeout": 3600,
}


def resolve_dataset_revision(dataset: str, split: str) -> str | None:
    """Pin the official public split without constraining explicit custom data."""
    if dataset == DATASET_NAME and split == DATASET_SPLIT:
        return DATASET_REVISION
    return None
