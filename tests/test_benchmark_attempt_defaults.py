"""Regression tests for one-shot benchmark inference defaults."""

import pytest

from benchmarks.swebench.config import INFER_DEFAULTS as SWEBENCH_DEFAULTS
from benchmarks.swebenchpro.config import INFER_DEFAULTS as SWEBENCH_PRO_DEFAULTS
from benchmarks.terminalbench.config import INFER_DEFAULTS as TERMINAL_DEFAULTS
from benchmarks.utils.args_parser import get_parser, validate_delegation_agent


def _parse_swe_defaults(defaults: dict[str, object]):
    parser = get_parser()
    parser.set_defaults(**defaults)
    return parser.parse_args(["llm-config.json"])


def test_swebench_verified_defaults_to_one_instance_and_attempt() -> None:
    args = _parse_swe_defaults(SWEBENCH_DEFAULTS)

    assert args.workspace == "docker"
    assert args.n_limit == 1
    assert args.num_workers == 1
    assert args.n_critic_runs == 1
    assert args.max_retries == 0
    assert args.enable_delegation is True


def test_swebench_pro_defaults_to_one_instance_and_attempt() -> None:
    args = _parse_swe_defaults(SWEBENCH_PRO_DEFAULTS)

    assert args.workspace == "docker"
    assert args.n_limit == 1
    assert args.num_workers == 1
    assert args.n_critic_runs == 1
    assert args.max_retries == 0
    assert args.enable_delegation is True


def test_swe_cli_help_reports_benchmark_workspace_default() -> None:
    parser = get_parser()
    parser.set_defaults(**SWEBENCH_DEFAULTS)

    assert "Type of workspace to use (default: docker)" in parser.format_help()


def test_swe_benchmarks_allow_explicit_single_agent_opt_out() -> None:
    parser = get_parser()
    parser.set_defaults(**SWEBENCH_DEFAULTS)

    args = parser.parse_args(["llm-config.json", "--disable-delegation"])

    assert args.enable_delegation is False


def test_native_delegation_rejects_acp_agents() -> None:
    parser = get_parser()
    parser.set_defaults(**SWEBENCH_DEFAULTS)
    args = parser.parse_args(["llm-config.json", "--agent-type", "acp-codex"])

    with pytest.raises(SystemExit):
        validate_delegation_agent(parser, args)

    single_agent_args = parser.parse_args(
        [
            "llm-config.json",
            "--agent-type",
            "acp-codex",
            "--disable-delegation",
        ]
    )
    validate_delegation_agent(parser, single_agent_args)


def test_terminal_bench_defaults_to_one_task_and_attempt() -> None:
    assert TERMINAL_DEFAULTS["n_limit"] == 1
    assert TERMINAL_DEFAULTS["num_workers"] == 1
    assert TERMINAL_DEFAULTS["n_attempts"] == 1
    assert TERMINAL_DEFAULTS["max_retries"] == 0
    assert TERMINAL_DEFAULTS["enable_delegation"] is True
