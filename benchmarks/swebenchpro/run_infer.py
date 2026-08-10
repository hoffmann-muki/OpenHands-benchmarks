import argparse
import json
import sys
from pathlib import Path

from benchmark_agents.delegation import (
    BENCHMARK_AGENT_TOPOLOGY,
    BENCHMARK_SINGLE_AGENT_TOPOLOGY,
)
from benchmark_agents.provenance import (
    openhands_benchmarks_source_commit,
    openhands_sdk_source_commit,
)
from benchmarks.swebench.run_infer import SWEBenchEvaluation
from benchmarks.swebenchpro import constants
from benchmarks.swebenchpro.build_images import (
    extract_custom_tag,
    get_official_docker_image,
)
from benchmarks.swebenchpro.config import (
    DEFAULT_INSTANCE_TIMEOUT_GRACE_SECONDS,
    EVAL_DEFAULTS,
    INFER_DEFAULTS,
    resolve_dataset_revision,
)
from benchmarks.tracing import create_trace_run
from benchmarks.utils.args_parser import (
    add_prompt_path_argument,
    add_trace_dir_argument,
    get_parser,
    resolve_selected_instances_file,
    validate_delegation_agent,
)
from benchmarks.utils.critics import create_critic
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
    retain_failure_predictions,
)
from benchmarks.utils.llm_config import benchmark_default_model, load_llm_config
from benchmarks.utils.models import EvalInstance, EvalMetadata
from openhands.sdk import get_logger


logger = get_logger(__name__)


class SWEBenchProEvaluation(SWEBenchEvaluation):
    def trace_benchmark_name(self) -> str:
        return "swe-bench-pro"

    def get_official_docker_image(self, instance: EvalInstance) -> str:
        return get_official_docker_image(instance.data)

    def extract_custom_tag(self, official_docker_image: str) -> str:
        return extract_custom_tag(official_docker_image)

    def should_wrap_instance(self, instance: EvalInstance) -> bool:
        return False

    def get_source_repo_path(self, instance: EvalInstance) -> str:
        return constants.SOURCE_REPO_PATH


def parse_args(
    *,
    force_single_agent: bool,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = get_parser(
        default_llm_model=benchmark_default_model(single_agent=force_single_agent)
    )
    add_prompt_path_argument(parser, __file__)
    add_trace_dir_argument(parser)
    parser.add_argument(
        "--inference-timeout",
        type=int,
        default=INFER_DEFAULTS["inference_timeout"],
        help="Maximum agent inference time in seconds (default: %(default)s)",
    )
    parser.set_defaults(
        **{
            **INFER_DEFAULTS,
            "enable_delegation": not force_single_agent,
        }
    )
    raw_args = list(argv if argv is not None else sys.argv[1:])
    args = parser.parse_args(raw_args)
    if force_single_agent and args.enable_delegation:
        parser.error("This entrypoint enforces native single-agent execution")
    if force_single_agent and args.agent_type != "default":
        parser.error(
            "Single-agent execution requires the native OpenHands default agent"
        )
    if force_single_agent and args.dataset != INFER_DEFAULTS["dataset"]:
        parser.error(
            f"This entrypoint requires the official dataset {INFER_DEFAULTS['dataset']}"
        )
    if force_single_agent and args.split != INFER_DEFAULTS["split"]:
        parser.error("This entrypoint requires the official test split")
    args.select = resolve_selected_instances_file(raw_args, args.select)
    validate_delegation_agent(parser, args)

    if args.n_critic_runs < 1:
        raise ValueError(f"n_critic_runs must be >= 1, got {args.n_critic_runs}")
    if args.inference_timeout < 1:
        parser.error("--inference-timeout must be a positive integer")
    return args


def _main(*, force_single_agent: bool) -> None:
    args = parse_args(force_single_agent=force_single_agent)

    llm = load_llm_config(
        args.llm_config_path,
        default_model=benchmark_default_model(single_agent=force_single_agent),
        num_retries=1,
        caching_prompt=False,
    )
    sdk_commit = openhands_sdk_source_commit()
    benchmark_commit = openhands_benchmarks_source_commit(
        require_clean=bool(args.trace_dir)
    )
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    dataset_description = (
        args.dataset.replace("/", "__") + "-" + args.split.replace("/", "__")
    )
    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=dataset_description,
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note=args.note,
    )

    critic = create_critic(args)
    logger.info("Using critic: %s", type(critic).__name__)
    logger.info("Using tool preset: %s", args.tool_preset)

    enable_condenser = args.enable_condenser
    if args.disable_condenser:
        enable_condenser = False
    trace_run = (
        create_trace_run(
            Path(args.trace_dir),
            benchmark="swe-bench-pro",
            framework="openhands",
        )
        if args.trace_dir
        else None
    )

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        inference_timeout=args.inference_timeout,
        eval_output_dir=structured_output_dir,
        trace_dir=str(trace_run.root) if trace_run is not None else None,
        trace_run_id=trace_run.id if trace_run is not None else None,
        trace_created_at=trace_run.created_at if trace_run is not None else None,
        details={
            "benchmark": "swe-bench-pro",
            "benchmark_display_name": "SWE-bench Pro",
            "dataset_revision": resolve_dataset_revision(args.dataset, args.split),
            "agent_topology": (
                BENCHMARK_AGENT_TOPOLOGY
                if args.enable_delegation
                else BENCHMARK_SINGLE_AGENT_TOPOLOGY
            ),
            "primary_agent": "coordinator" if args.enable_delegation else "agent",
            "agent_sequence": (
                ["coordinator", "navigator", "patcher", "reviewer"]
                if args.enable_delegation
                else ["agent"]
            ),
            "delegation_enabled": args.enable_delegation,
            "inference_timeout": args.inference_timeout,
            "instance_timeout_grace": DEFAULT_INSTANCE_TIMEOUT_GRACE_SECONDS,
            "evaluation_timeout": EVAL_DEFAULTS["timeout"],
            "agent_source_commit": sdk_commit,
            "benchmark_source_commit": benchmark_commit,
            "provider_attempts_per_turn": 1,
        },
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
        n_critic_runs=args.n_critic_runs,
        critic=critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        tool_preset=args.tool_preset,
        enable_delegation=args.enable_delegation,
        agent_type=args.agent_type,
        enable_condenser=enable_condenser,
        condenser_max_size=args.condenser_max_size,
        condenser_keep_first=args.condenser_keep_first,
    )

    evaluator = SWEBenchProEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
        instance_timeout=(
            args.inference_timeout + DEFAULT_INSTANCE_TIMEOUT_GRACE_SECONDS
        ),
    )
    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))
    recovered = retain_failure_predictions(
        Path(evaluator.output_path),
        args.n_critic_runs,
    )
    if recovered:
        logger.info(
            "Restored %d SWE-bench Pro predictions from failed attempt records",
            recovered,
        )

    logger.info("Evaluation completed!")
    result = {"output_json": str(evaluator.output_path)}
    if trace_run is not None:
        result["trace_dir"] = str(trace_run.root)
    print(json.dumps(result))


def main() -> None:
    _main(force_single_agent=False)


def swebenchpro_single_main() -> None:
    _main(force_single_agent=True)


if __name__ == "__main__":
    main()
