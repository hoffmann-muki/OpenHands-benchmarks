"""
SWE-bench benchmark configuration.

Default values aligned with evaluation repository (OpenHands/evaluation).
"""

from pathlib import Path


DEFAULT_SMOKE_INSTANCES_FILE = Path(__file__).with_name("smoke_instances.txt")
DEFAULT_MAX_ITERATIONS = 24
DEFAULT_MAX_FAKE_RESPONSES = 0
DEFAULT_INFERENCE_TIMEOUT_SECONDS = 30 * 60
# The outer evaluator also covers non-LLM workspace setup and teardown.
DEFAULT_INSTANCE_TIMEOUT_GRACE_SECONDS = 10 * 60

# Condenser configuration
# The condenser manages conversation context by automatically truncating history
# when it exceeds max_size and replacing dropped events with an LLM-generated summary.
CONDENSER_DEFAULTS = {
    "enable_condenser": True,
    "condenser_max_size": 240,  # Maximum number of events before condensing
    "condenser_keep_first": 2,  # Number of initial events to always keep
}

# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "dataset": "princeton-nlp/SWE-bench_Verified",
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
    **CONDENSER_DEFAULTS,
}

# Evaluation defaults (used by eval_infer.py)
EVAL_DEFAULTS = {
    "dataset": "princeton-nlp/SWE-bench_Verified",
    "split": "test",
    "workers": 1,
    "modal": False,
    "timeout": 3600,
}
