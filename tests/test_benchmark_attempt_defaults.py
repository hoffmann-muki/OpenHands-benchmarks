"""Regression tests for one-shot benchmark inference defaults."""

import pytest

from benchmarks.swebench.config import (
    DEFAULT_MAX_FAKE_RESPONSES,
    DEFAULT_SMOKE_INSTANCES_FILE as SWEBENCH_SMOKE_INSTANCES_FILE,
    EVAL_DEFAULTS as SWEBENCH_EVAL_DEFAULTS,
    INFER_DEFAULTS as SWEBENCH_DEFAULTS,
)
from benchmarks.swebenchpro.config import (
    DEFAULT_SMOKE_INSTANCES_FILE as SWEBENCH_PRO_SMOKE_INSTANCES_FILE,
    EVAL_DEFAULTS as SWEBENCH_PRO_EVAL_DEFAULTS,
    INFER_DEFAULTS as SWEBENCH_PRO_DEFAULTS,
)
from benchmarks.terminalbench.config import INFER_DEFAULTS as TERMINAL_DEFAULTS
from benchmarks.utils.args_parser import (
    get_parser,
    resolve_selected_instances_file,
    validate_delegation_agent,
)
from benchmarks.utils.llm_config import DEFAULT_LLM_MODEL


def _parse_swe_defaults(defaults: dict[str, object]):
    parser = get_parser()
    parser.set_defaults(**defaults)
    return parser.parse_args(["llm-config.json"])


def test_swebench_verified_defaults_to_one_instance_and_attempt() -> None:
    args = _parse_swe_defaults(SWEBENCH_DEFAULTS)

    assert args.workspace == "docker"
    assert args.n_limit == 1
    assert args.select.endswith("benchmarks/swebench/smoke_instances.txt")
    assert args.max_iterations == 24
    assert DEFAULT_MAX_FAKE_RESPONSES == 0
    assert args.inference_timeout == 15 * 60
    assert args.num_workers == 1
    assert args.n_critic_runs == 1
    assert args.max_retries == 0
    assert args.enable_delegation is True


def test_swebench_pro_defaults_to_one_instance_and_attempt() -> None:
    args = _parse_swe_defaults(SWEBENCH_PRO_DEFAULTS)

    assert args.workspace == "docker"
    assert args.n_limit == 1
    assert args.select.endswith("benchmarks/swebenchpro/smoke_instances.txt")
    assert args.max_iterations == 24
    assert args.inference_timeout == 30 * 60
    assert args.num_workers == 1
    assert args.n_critic_runs == 1
    assert args.max_retries == 0
    assert args.enable_delegation is True


def test_swe_cli_help_reports_benchmark_workspace_default() -> None:
    parser = get_parser()
    parser.set_defaults(**SWEBENCH_DEFAULTS)

    assert "Type of workspace to use (default: docker)" in parser.format_help()


def test_swe_cli_defaults_to_qwen_without_a_config_path() -> None:
    parser = get_parser(default_llm_model=DEFAULT_LLM_MODEL)
    parser.set_defaults(**SWEBENCH_DEFAULTS)

    args = parser.parse_args([])

    assert args.llm_config_path is None
    assert DEFAULT_LLM_MODEL in parser.format_help()


def test_swe_evaluation_defaults_are_single_worker_local_docker() -> None:
    assert SWEBENCH_EVAL_DEFAULTS["workers"] == 1
    assert SWEBENCH_EVAL_DEFAULTS["modal"] is False
    assert SWEBENCH_PRO_EVAL_DEFAULTS["workers"] == 1
    assert SWEBENCH_PRO_EVAL_DEFAULTS["use_local_docker"] is True


def test_swe_smoke_instance_files_are_explicit_and_aligned() -> None:
    assert SWEBENCH_SMOKE_INSTANCES_FILE.read_text().splitlines() == [
        "scikit-learn__scikit-learn-13439"
    ]
    assert SWEBENCH_PRO_SMOKE_INSTANCES_FILE.read_text().splitlines() == [
        "instance_qutebrowser__qutebrowser-5fdc83e5da6222fe61163395baaad7ae57fa2cb4-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d"
    ]


def test_explicit_limit_replaces_only_the_implicit_smoke_selection() -> None:
    smoke_file = str(SWEBENCH_SMOKE_INSTANCES_FILE)

    assert resolve_selected_instances_file(["--n-limit", "5"], smoke_file) is None
    assert resolve_selected_instances_file(["--n-limit=5"], smoke_file) is None
    assert (
        resolve_selected_instances_file(
            ["--n-limit", "2", "--select", "chosen.txt"],
            "chosen.txt",
        )
        == "chosen.txt"
    )
    assert resolve_selected_instances_file([], smoke_file) == smoke_file


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
