import json
import sys
import uuid
from pathlib import Path

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
    INFER_DEFAULTS,
)
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
)
from benchmarks.utils.llm_config import DEFAULT_LLM_MODEL, load_llm_config
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


def main() -> None:
    parser = get_parser(default_llm_model=DEFAULT_LLM_MODEL)
    add_prompt_path_argument(parser, __file__)
    add_trace_dir_argument(parser)
    parser.add_argument(
        "--inference-timeout",
        type=int,
        default=INFER_DEFAULTS["inference_timeout"],
        help="Maximum agent inference time in seconds (default: %(default)s)",
    )
    parser.set_defaults(**INFER_DEFAULTS)
    raw_args = sys.argv[1:]
    args = parser.parse_args(raw_args)
    args.select = resolve_selected_instances_file(raw_args, args.select)
    validate_delegation_agent(parser, args)

    if args.n_critic_runs < 1:
        raise ValueError(f"n_critic_runs must be >= 1, got {args.n_critic_runs}")
    if args.inference_timeout < 1:
        parser.error("--inference-timeout must be a positive integer")

    llm = load_llm_config(
        args.llm_config_path,
        default_model=DEFAULT_LLM_MODEL,
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
    trace_run_id = f"trace-run-{uuid.uuid4().hex}" if args.trace_dir else None
    trace_dir = (
        str(Path(args.trace_dir).resolve() / trace_run_id)
        if args.trace_dir and trace_run_id
        else None
    )

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        inference_timeout=args.inference_timeout,
        eval_output_dir=structured_output_dir,
        trace_dir=trace_dir,
        trace_run_id=trace_run_id,
        details={
            "inference_timeout": args.inference_timeout,
            "instance_timeout_grace": DEFAULT_INSTANCE_TIMEOUT_GRACE_SECONDS,
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

    logger.info("Evaluation completed!")
    result = {"output_json": str(evaluator.output_path)}
    if trace_dir is not None:
        result["trace_dir"] = trace_dir
    print(json.dumps(result))


if __name__ == "__main__":
    main()
