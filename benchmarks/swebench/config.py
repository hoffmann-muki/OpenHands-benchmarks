"""
SWE-bench benchmark configuration.

Default values aligned with evaluation repository (OpenHands/evaluation).
"""

from dataclasses import dataclass
from pathlib import Path


DEFAULT_SMOKE_INSTANCES_FILE = Path(__file__).with_name("smoke_instances.txt")
DEFAULT_LITE_SMOKE_INSTANCES_FILE = Path(__file__).with_name("smoke_instances_lite.txt")
DEFAULT_MAX_ITERATIONS = 24
DEFAULT_MAX_FAKE_RESPONSES = 0
DEFAULT_INFERENCE_TIMEOUT_SECONDS = 15 * 60
# Large repositories can take longer than the SDK's 30-second command default
# to copy out of /testbed on memory-constrained local Docker installations.
DEFAULT_WORKSPACE_COPY_TIMEOUT_SECONDS = 5 * 60
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


@dataclass(frozen=True, slots=True)
class ClassicSweBenchVariant:
    benchmark: str
    display_name: str
    dataset: str
    smoke_instances_file: Path


SWE_BENCH_VERIFIED = ClassicSweBenchVariant(
    benchmark="swe-bench-verified",
    display_name="SWE-bench Verified",
    dataset="princeton-nlp/SWE-bench_Verified",
    smoke_instances_file=DEFAULT_SMOKE_INSTANCES_FILE,
)

SWE_BENCH_LITE = ClassicSweBenchVariant(
    benchmark="swe-bench-lite",
    display_name="SWE-bench Lite",
    dataset="princeton-nlp/SWE-bench_Lite",
    smoke_instances_file=DEFAULT_LITE_SMOKE_INSTANCES_FILE,
)
