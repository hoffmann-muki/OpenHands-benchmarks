"""Regression tests for one-shot benchmark inference defaults."""

from benchmarks.swebench.config import INFER_DEFAULTS as SWEBENCH_DEFAULTS
from benchmarks.swebenchpro.config import INFER_DEFAULTS as SWEBENCH_PRO_DEFAULTS
from benchmarks.terminalbench.config import INFER_DEFAULTS as TERMINAL_DEFAULTS
from benchmarks.utils.args_parser import get_parser


def _parse_swe_defaults(defaults: dict[str, object]):
    parser = get_parser()
    parser.set_defaults(**defaults)
    return parser.parse_args(["llm-config.json"])


def test_swebench_verified_defaults_to_one_agent_attempt() -> None:
    args = _parse_swe_defaults(SWEBENCH_DEFAULTS)

    assert args.n_critic_runs == 1
    assert args.max_retries == 0


def test_swebench_pro_defaults_to_one_agent_attempt() -> None:
    args = _parse_swe_defaults(SWEBENCH_PRO_DEFAULTS)

    assert args.n_critic_runs == 1
    assert args.max_retries == 0


def test_terminal_bench_defaults_to_one_agent_attempt() -> None:
    assert TERMINAL_DEFAULTS["n_attempts"] == 1
    assert TERMINAL_DEFAULTS["max_retries"] == 0
