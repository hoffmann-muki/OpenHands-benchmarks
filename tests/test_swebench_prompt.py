"""Parity checks for the concise SWE-bench Verified prompt."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined


def test_default_prompt_contains_only_public_issue_context() -> None:
    prompt_path = (
        Path(__file__).parents[1] / "benchmarks" / "swebench" / "prompts" / "default.j2"
    )
    prompt = (
        Environment(
            loader=FileSystemLoader(str(prompt_path.parent)),
            undefined=StrictUndefined,
        )
        .get_template(prompt_path.name)
        .render(
            benchmark_display_name="SWE-bench Verified",
            instance={
                "repo_path": "/workspace/demo",
                "repo": "owner/demo",
                "base_commit": "abc123",
                "instance_id": "owner__demo-1",
                "difficulty": "medium",
                "problem_statement": "Fix the public issue.",
                "patch": "DO NOT RENDER GOLD PATCH",
                "test_patch": "DO NOT RENDER TEST PATCH",
            },
        )
    )

    assert "Fix the public issue." in prompt
    assert "DO NOT RENDER GOLD PATCH" not in prompt
    assert "DO NOT RENDER TEST PATCH" not in prompt
    assert "Phase 1" not in prompt
    assert "Completion requirements" in prompt


def test_prompt_uses_the_selected_classic_swebench_variant() -> None:
    prompt_path = (
        Path(__file__).parents[1] / "benchmarks" / "swebench" / "prompts" / "default.j2"
    )
    prompt = (
        Environment(
            loader=FileSystemLoader(str(prompt_path.parent)),
            undefined=StrictUndefined,
        )
        .get_template(prompt_path.name)
        .render(
            benchmark_display_name="SWE-bench Lite",
            instance={
                "repo_path": "/workspace/demo",
                "repo": "owner/demo",
                "base_commit": "abc123",
                "instance_id": "owner__demo-1",
                "difficulty": None,
                "problem_statement": "Fix the public issue.",
            },
        )
    )

    assert prompt.startswith("Resolve this SWE-bench Lite issue using OpenHands.")
