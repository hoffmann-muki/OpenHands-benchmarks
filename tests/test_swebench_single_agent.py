import sys

import pytest

from benchmarks.swebench.config import (
    DEFAULT_INFERENCE_TIMEOUT_SECONDS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_WORKSPACE_COPY_TIMEOUT_SECONDS,
    SWE_BENCH_LITE,
    SWE_BENCH_VERIFIED,
    ClassicSweBenchVariant,
)
from benchmarks.swebench.run_infer import (
    parse_args,
    swebench_lite_single_main,
)
from benchmarks.utils.llm_config import benchmark_default_model


def test_classic_swebench_single_agent_variants_are_distinct() -> None:
    assert SWE_BENCH_VERIFIED.benchmark == "swe-bench-verified"
    assert SWE_BENCH_VERIFIED.dataset == "princeton-nlp/SWE-bench_Verified"
    assert SWE_BENCH_VERIFIED.smoke_instances_file.read_text().strip() == (
        "scikit-learn__scikit-learn-13439"
    )

    assert SWE_BENCH_LITE.benchmark == "swe-bench-lite"
    assert SWE_BENCH_LITE.dataset == "princeton-nlp/SWE-bench_Lite"
    assert SWE_BENCH_LITE.smoke_instances_file.read_text().strip() == (
        "astropy__astropy-12907"
    )


@pytest.mark.parametrize("variant", [SWE_BENCH_LITE, SWE_BENCH_VERIFIED])
def test_classic_single_agent_defaults_resolve_the_parity_contract(
    variant: ClassicSweBenchVariant,
) -> None:
    args = parse_args(variant, force_single_agent=True, argv=[])

    assert args.dataset == variant.dataset
    assert args.split == "test"
    assert args.select == str(variant.smoke_instances_file)
    assert args.n_limit == 1
    assert args.num_workers == 1
    assert args.n_critic_runs == 1
    assert args.max_retries == 0
    assert args.max_iterations == DEFAULT_MAX_ITERATIONS == 24
    assert args.inference_timeout == DEFAULT_INFERENCE_TIMEOUT_SECONDS == 900
    assert DEFAULT_WORKSPACE_COPY_TIMEOUT_SECONDS == 300
    assert args.workspace == "docker"
    assert args.agent_type == "default"
    assert args.enable_delegation is False
    assert args.trace_dir.endswith("OpenHands-benchmarks/.benchmark-traces")
    assert benchmark_default_model(single_agent=True) == (
        "openrouter/poolside/laguna-s-2.1:free"
    )
    assert benchmark_default_model(single_agent=False) == (
        "openrouter/qwen/qwen3-coder-next"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--enable-delegation"],
        ["--agent-type", "acp-codex"],
        ["--dataset", "princeton-nlp/SWE-bench_Verified"],
        ["--split", "dev"],
    ],
)
def test_lite_single_agent_entrypoint_rejects_topology_or_dataset_escape(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["swebench-lite-single-infer", *arguments])

    with pytest.raises(SystemExit) as raised:
        swebench_lite_single_main()

    assert raised.value.code == 2
