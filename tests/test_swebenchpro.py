import json
from pathlib import Path

import pandas as pd
import pytest
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from benchmarks.swebenchpro.build_images import (
    collect_unique_base_images,
    extract_custom_tag,
    get_official_docker_image,
)
from benchmarks.swebenchpro.constants import SOURCE_REPO_PATH
from benchmarks.swebenchpro.eval_infer import (
    convert_to_swebenchpro_format,
    get_parser,
    run_swebenchpro_evaluation,
    write_report,
)


def test_get_official_docker_image_uses_dataset_dockerhub_tag():
    image = get_official_docker_image(
        {
            "instance_id": "instance_demo",
            "dockerhub_tag": "nodebb.nodebb-instance_demo",
        }
    )

    assert image == "docker.io/jefzda/sweap-images:nodebb.nodebb-instance_demo"
    assert extract_custom_tag(image) == "nodebb.nodebb-instance_demo"


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
        lambda dataset_name, split, eval_limit, selected_instances_file: df,
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
