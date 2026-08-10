import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from benchmarks.swebenchpro.build_images import (
    collect_unique_base_images,
    extract_custom_tag,
    get_official_docker_image,
)
from benchmarks.swebenchpro.config import (
    DATASET_REVISION,
    DEFAULT_INFERENCE_TIMEOUT_SECONDS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_SMOKE_INSTANCES_FILE,
    INFER_DEFAULTS,
    resolve_dataset_revision,
)
from benchmarks.swebenchpro.constants import SOURCE_REPO_PATH
from benchmarks.swebenchpro.eval_infer import (
    convert_to_swebenchpro_format,
    get_parser,
    run_swebenchpro_evaluation,
    write_report,
)
from benchmarks.swebenchpro.run_infer import parse_args, swebenchpro_single_main
from benchmarks.utils.llm_config import benchmark_default_model


def test_get_official_docker_image_uses_dataset_dockerhub_tag():
    image = get_official_docker_image(
        {
            "instance_id": "instance_demo",
            "dockerhub_tag": "nodebb.nodebb-instance_demo",
        }
    )

    assert image == "docker.io/jefzda/sweap-images:nodebb.nodebb-instance_demo"
    assert extract_custom_tag(image) == "nodebb.nodebb-instance_demo"


def test_single_agent_defaults_resolve_the_parity_contract() -> None:
    args = parse_args(force_single_agent=True, argv=[])

    assert args.dataset == "ScaleAI/SWE-bench_Pro"
    assert args.split == "test"
    assert args.select == str(DEFAULT_SMOKE_INSTANCES_FILE)
    assert DEFAULT_SMOKE_INSTANCES_FILE.read_text().strip() == (
        "instance_qutebrowser__qutebrowser-5fdc83e5da6222fe61163395baaad7ae57fa2cb4-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d"
    )
    assert args.n_limit == 1
    assert args.num_workers == 1
    assert args.n_critic_runs == 1
    assert args.max_retries == 0
    assert args.max_iterations == DEFAULT_MAX_ITERATIONS == 24
    assert args.inference_timeout == DEFAULT_INFERENCE_TIMEOUT_SECONDS == 1800
    assert args.workspace == "docker"
    assert args.agent_type == "default"
    assert args.enable_delegation is False
    assert args.trace_dir.endswith("OpenHands-benchmarks/.benchmark-traces")
    assert INFER_DEFAULTS["dataset"] == "ScaleAI/SWE-bench_Pro"
    assert DATASET_REVISION == "7ab5114912baf22bb098818e604c02fe7ad2c11f"
    assert resolve_dataset_revision(args.dataset, args.split) == DATASET_REVISION
    assert resolve_dataset_revision("custom/dataset", "test") is None
    assert benchmark_default_model(single_agent=True) == (
        "openrouter/poolside/laguna-s-2.1:free"
    )
    assert benchmark_default_model(single_agent=False) == (
        "openrouter/qwen/qwen3-coder-next"
    )


def test_collect_unique_base_images_deduplicates(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "instance_id": "instance_a",
                "dockerhub_tag": "repo-image-a",
            },
            {
                "instance_id": "instance_b",
                "dockerhub_tag": "repo-image-a",
            },
            {
                "instance_id": "instance_c",
                "dockerhub_tag": "repo-image-c",
            },
        ]
    )
    monkeypatch.setattr(
        "benchmarks.swebenchpro.build_images.get_dataset",
        lambda dataset_name,
        split,
        eval_limit,
        selected_instances_file,
        selection_mode,
        revision: (
            df
            if selection_mode == "ordered" and revision == DATASET_REVISION
            else pytest.fail("expected ordered mode at the pinned dataset revision")
        ),
    )

    images = collect_unique_base_images("ScaleAI/SWE-bench_Pro", "test", 0)

    assert images == [
        "docker.io/jefzda/sweap-images:repo-image-a",
        "docker.io/jefzda/sweap-images:repo-image-c",
    ]


def test_extract_custom_tag_shortens_long_tags():
    image = (
        "docker.io/jefzda/sweap-images:"
        "qutebrowser.qutebrowser-qutebrowser__qutebrowser-"
        "5fdc83e5da6222fe61163395baaad7ae57fa2cb4-v363c8a7e5ccdf6968fc7ab84a2053ac780366"
    )

    custom_tag = extract_custom_tag(image)

    assert len(custom_tag) <= 96
    assert custom_tag.startswith("qutebrowser.qutebrowser-qutebrowser__qutebrowser-")
    assert custom_tag != image.rsplit(":", 1)[1]


def test_build_cli_defaults_to_smoke_instance_file():
    from benchmarks.swebenchpro.build_images import get_parser as get_build_parser
    from benchmarks.swebenchpro.config import DEFAULT_SMOKE_INSTANCES_FILE

    parser = get_build_parser()

    assert parser.parse_args([]).select == str(DEFAULT_SMOKE_INSTANCES_FILE)
    assert parser.parse_args(["--select", ""]).select == ""


def test_default_prompt_includes_all_public_pro_issue_fields():
    prompt_path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "swebenchpro"
        / "prompts"
        / "default.j2"
    )
    template = Environment(
        loader=FileSystemLoader(str(prompt_path.parent)),
        undefined=StrictUndefined,
    ).get_template(prompt_path.name)

    prompt = template.render(
        instance={
            "repo_path": "/workspace/demo",
            "repo": "owner/demo",
            "base_commit": "abc123",
            "instance_id": "owner__demo-1",
            "problem_statement": "Fix the public issue.",
            "requirements": "Preserve the documented behavior.",
            "interface": "Add Widget.render().",
            "repo_language": "Python",
            "gold_patch": "DO NOT RENDER GOLD PATCH",
            "test_patch": "DO NOT RENDER TEST PATCH",
        }
    )

    assert "Fix the public issue." in prompt
    assert "Requirements:\nPreserve the documented behavior." in prompt
    assert "New interfaces introduced:\nAdd Widget.render()." in prompt
    assert "Repository language: Python" in prompt
    assert "DO NOT RENDER GOLD PATCH" not in prompt
    assert "DO NOT RENDER TEST PATCH" not in prompt
    assert "Phase 1" not in prompt
    assert "Completion requirements" in prompt


def test_evaluation_cli_defaults_to_local_docker():
    args = get_parser().parse_args(["output.jsonl"])

    assert args.use_local_docker is True
    assert args.workers == 1


def test_evaluation_cli_allows_modal_opt_in():
    args = get_parser().parse_args(["output.jsonl", "--no-use-local-docker"])

    assert args.use_local_docker is False


def test_convert_to_swebenchpro_format_writes_patch_array(tmp_path):
    input_path = tmp_path / "output.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "instance_id": "instance_a",
                        "test_result": {"git_patch": "diff --git a/a b/a"},
                    }
                ),
                json.dumps(
                    {
                        "instance_id": "instance_b",
                        "test_result": {},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "output.swebenchpro.json"

    convert_to_swebenchpro_format(
        str(input_path), str(output_path), prefix="demo-model"
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == [
        {
            "instance_id": "instance_a",
            "patch": "diff --git a/a b/a",
            "prefix": "demo-model",
        },
        {
            "instance_id": "instance_b",
            "patch": "",
            "prefix": "demo-model",
        },
    ]


def test_convert_to_swebenchpro_format_raises_on_malformed_input(tmp_path):
    input_path = tmp_path / "broken_output.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "instance_id": "instance_a",
                        "test_result": {"git_patch": "diff --git a/a b/a"},
                    }
                ),
                "{not-json}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "output.swebenchpro.json"

    with pytest.raises(ValueError, match="malformed input"):
        convert_to_swebenchpro_format(str(input_path), str(output_path))

    assert not output_path.exists()


def test_run_swebenchpro_evaluation_requires_harness_script(tmp_path):
    raw_sample_path = tmp_path / "raw_samples.jsonl"
    raw_sample_path.write_text('{"instance_id": "instance_a"}\n', encoding="utf-8")
    patch_path = tmp_path / "output.swebenchpro.json"
    patch_path.write_text("[]", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="swe_bench_pro_eval.py"):
        run_swebenchpro_evaluation(
            harness_dir=tmp_path,
            raw_sample_path=raw_sample_path,
            patch_path=patch_path,
            output_dir=tmp_path / "eval_output",
            workers=1,
            dockerhub_username="anonymous",
            use_local_docker=True,
            block_network=False,
        )


def test_write_report_records_resolved_ids(tmp_path):
    eval_results_path = tmp_path / "eval_results.json"
    eval_results_path.write_text(
        json.dumps(
            {
                "instance_a": True,
                "instance_b": False,
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "output.report.json"

    report = write_report(eval_results_path, report_path)

    assert report["resolved_instances"] == 1
    assert report["unresolved_instances"] == 1
    assert report["resolved_ids"] == ["instance_a"]
    assert report["unresolved_ids"] == ["instance_b"]
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_source_repo_path_constant_matches_swebench_pro_layout():
    assert SOURCE_REPO_PATH == "/app"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--enable-delegation"],
        ["--agent-type", "acp-codex"],
        ["--dataset", "ScaleAI/SWE-bench_Pro_private"],
        ["--split", "dev"],
    ],
)
def test_single_agent_entrypoint_rejects_topology_or_dataset_escape(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["swebenchpro-single-infer", *arguments])

    with pytest.raises(SystemExit) as raised:
        swebenchpro_single_main()

    assert raised.value.code == 2
