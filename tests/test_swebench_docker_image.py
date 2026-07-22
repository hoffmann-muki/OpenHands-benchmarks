"""Regression tests for local SWE-bench agent-server image selection."""

from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.swebench import run_infer as swebench_run_infer
from benchmarks.swebenchpro.run_infer import SWEBenchProEvaluation
from benchmarks.utils.models import EvalInstance


class FakeDockerWorkspace:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


@pytest.mark.parametrize(
    ("evaluation_type", "instance", "task_tag"),
    [
        (
            swebench_run_infer.SWEBenchEvaluation,
            EvalInstance(id="django__django-11333", data={}),
            "sweb.eval.x86_64.django_1776_django-11333",
        ),
        (
            SWEBenchProEvaluation,
            EvalInstance(
                id="instance_demo",
                data={"dockerhub_tag": "nodebb.nodebb-instance_demo"},
            ),
            "nodebb.nodebb-instance_demo",
        ),
    ],
)
def test_docker_workspace_builds_and_uses_same_phased_image_tag(
    monkeypatch,
    evaluation_type,
    instance,
    task_tag,
):
    metadata = SimpleNamespace(
        workspace_type="docker",
        agent_type="default",
        enable_delegation=True,
        env_setup_commands=[],
    )
    evaluation = object.__new__(evaluation_type)
    object.__setattr__(evaluation, "metadata", metadata)

    build_args = {}

    def ensure_local_phased_image(**kwargs):
        build_args.update(kwargs)
        return False

    monkeypatch.setattr(
        swebench_run_infer,
        "get_phased_image_tag_prefix",
        lambda: "sdk1234-hash567",
    )
    monkeypatch.setattr(
        swebench_run_infer,
        "ensure_local_phased_image",
        ensure_local_phased_image,
    )
    monkeypatch.setattr(
        swebench_run_infer,
        "DockerWorkspace",
        FakeDockerWorkspace,
    )

    workspace = evaluation.prepare_workspace(instance)

    expected_image = (
        f"ghcr.io/openhands/eval-agent-server:sdk1234-hash567-{task_tag}-source-minimal"
    )
    assert build_args["agent_server_image"] == expected_image
    assert build_args["custom_tag"] == task_tag
    assert workspace.kwargs["server_image"] == expected_image
    assert workspace.kwargs["volumes"] == [evaluation.get_benchmark_agents_bind_mount()]
