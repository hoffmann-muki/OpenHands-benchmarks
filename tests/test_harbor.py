"""Tests for shared Harbor benchmark execution utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmarks.utils.harbor import (
    HarborCredentialMode,
    build_harbor_command,
    get_supported_agent_name,
    get_supported_task_filter_flag,
    run_harbor_evaluation,
)
from openhands.sdk import LLM


def _fake_run(returncode: int = 0, stderr: str = "") -> Any:
    """Return a fake subprocess.run callable capturing calls."""
    captured: dict[str, Any] = {}

    def fake(cmd: list[str], *, capture_output: bool, text: bool, env=None) -> Any:
        captured.setdefault("cmds", []).append(list(cmd))
        captured["env"] = env
        return type(
            "R", (), {"returncode": returncode, "stdout": "ok", "stderr": stderr}
        )()

    fake.captured = captured  # type: ignore[attr-defined]
    return fake


class TestGetSupportedTaskFilterFlag:
    """Tests for get_supported_task_filter_flag parsing logic."""

    def test_returns_include_task_name_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "benchmarks.utils.harbor._probe_harbor_run_help",
            lambda _: "--include-task-name\n--other-flag",
        )
        assert get_supported_task_filter_flag("harbor") == "--include-task-name"

    def test_returns_task_name_when_include_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "benchmarks.utils.harbor._probe_harbor_run_help",
            lambda _: "--task-name  Filter by task",
        )
        assert get_supported_task_filter_flag("harbor") == "--task-name"

    def test_include_task_name_takes_priority_over_task_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "benchmarks.utils.harbor._probe_harbor_run_help",
            lambda _: "--task-name ...\n--include-task-name ...",
        )
        assert get_supported_task_filter_flag("harbor") == "--include-task-name"

    def test_falls_back_to_include_task_name_when_neither_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "benchmarks.utils.harbor._probe_harbor_run_help",
            lambda _: "Options: --output-dir --help",
        )
        assert get_supported_task_filter_flag("harbor") == "--include-task-name"

    def test_falls_back_when_help_text_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty string is returned by _probe_harbor_run_help on timeout/not-found."""
        monkeypatch.setattr(
            "benchmarks.utils.harbor._probe_harbor_run_help",
            lambda _: "",
        )
        assert get_supported_task_filter_flag("harbor") == "--include-task-name"

    def test_regex_boundary_rejects_partial_matches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The lookbehind prevents flags that are embedded inside a word."""
        monkeypatch.setattr(
            "benchmarks.utils.harbor._probe_harbor_run_help",
            lambda _: "prefix--task-name  (no standalone flag here)",
        )
        assert get_supported_task_filter_flag("harbor") == "--include-task-name"


class TestGetSupportedAgentName:
    """Tests for get_supported_agent_name parsing logic."""

    def test_returns_openhands_sdk_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "benchmarks.utils.harbor._probe_harbor_run_help",
            lambda _: "Agents: openhands-sdk, other-agent",
        )
        assert get_supported_agent_name("harbor") == "openhands-sdk"

    def test_returns_openhands_when_sdk_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "benchmarks.utils.harbor._probe_harbor_run_help",
            lambda _: "Agents: openhands, other-agent",
        )
        assert get_supported_agent_name("harbor") == "openhands"

    def test_openhands_sdk_takes_priority_over_openhands(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "benchmarks.utils.harbor._probe_harbor_run_help",
            lambda _: "openhands openhands-sdk",
        )
        assert get_supported_agent_name("harbor") == "openhands-sdk"

    def test_falls_back_to_default_when_neither_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "benchmarks.utils.harbor._probe_harbor_run_help",
            lambda _: "Agents: some-other-agent",
        )
        assert (
            get_supported_agent_name("harbor", default_agent_name="custom-agent")
            == "custom-agent"
        )

    def test_falls_back_when_help_text_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "benchmarks.utils.harbor._probe_harbor_run_help",
            lambda _: "",
        )
        assert get_supported_agent_name("harbor") == "openhands-sdk"


class TestRunHarborEvaluationCredentialModes:
    """Tests for run_harbor_evaluation credential injection modes."""

    def test_agent_env_flags_mode_injects_ae_flags(self, tmp_path: Path) -> None:
        """AGENT_ENV_FLAGS mode passes credentials via --ae flags; env is None."""
        run = _fake_run()
        run_harbor_evaluation(
            llm=LLM(
                model="test/model",
                api_key="my-key",
                base_url="https://proxy.example.com",
            ),
            dataset="my-dataset",
            output_dir=str(tmp_path),
            credential_mode=HarborCredentialMode.AGENT_ENV_FLAGS,
            subprocess_run=run,
        )
        cmd = run.captured["cmds"][0]
        assert "--ae" in cmd
        assert any("LLM_API_KEY=my-key" in part for part in cmd)
        assert any("LLM_BASE_URL=https://proxy.example.com" in part for part in cmd)
        assert run.captured["env"] is None

    def test_process_env_mode_sets_env_vars(self, tmp_path: Path) -> None:
        """PROCESS_ENV mode passes credentials via env dict; no --ae flags."""
        run = _fake_run()
        run_harbor_evaluation(
            llm=LLM(
                model="test/model",
                api_key="my-key",
                base_url="https://proxy.example.com",
            ),
            dataset="my-dataset",
            output_dir=str(tmp_path),
            credential_mode=HarborCredentialMode.PROCESS_ENV,
            subprocess_run=run,
        )
        cmd = run.captured["cmds"][0]
        assert "--ae" not in cmd
        env = run.captured["env"]
        assert env is not None
        assert env["LLM_API_KEY"] == "my-key"
        assert env["LLM_BASE_URL"] == "https://proxy.example.com"


class TestRunHarborEvaluationTaskFiltering:
    """Tests for task_ids, n_limit, and fallback-retry in run_harbor_evaluation."""

    def test_task_ids_and_n_limit_included_in_command(self, tmp_path: Path) -> None:
        run = _fake_run()
        run_harbor_evaluation(
            llm=LLM(model="test/model"),
            dataset="my-dataset",
            output_dir=str(tmp_path),
            task_ids=["task-a", "task-b"],
            n_limit=5,
            task_filter_flag="--task-name",
            subprocess_run=run,
        )
        cmd = run.captured["cmds"][0]
        assert cmd.count("--task-name") == 2
        assert "task-a" in cmd
        assert "task-b" in cmd
        assert cmd[cmd.index("--n-tasks") + 1] == "5"

    def test_fallback_retry_switches_to_include_task_name(self, tmp_path: Path) -> None:
        """When --task-name fails with 'No such option', retries with --include-task-name."""
        cmds: list[list[str]] = []

        def fake_run(
            cmd: list[str], *, capture_output: bool, text: bool, env=None
        ) -> Any:
            cmds.append(list(cmd))
            if "--task-name" in cmd:
                return type(
                    "R",
                    (),
                    {
                        "returncode": 2,
                        "stdout": "",
                        "stderr": "No such option: --task-name",
                    },
                )()
            return type("R", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

        run_harbor_evaluation(
            llm=LLM(model="test/model"),
            dataset="my-dataset",
            output_dir=str(tmp_path),
            task_ids=["task-a"],
            task_filter_flag="--task-name",
            retry_legacy_task_flag=True,
            subprocess_run=fake_run,
        )
        assert len(cmds) == 2
        assert "--task-name" in cmds[0]
        assert "--include-task-name" in cmds[1]
        assert "--task-name" not in cmds[1]


class TestBuildHarborCommand:
    def test_builds_reproducible_execution_controls(self, tmp_path: Path) -> None:
        cmd = build_harbor_command(
            model="test/model",
            dataset="org/dataset",
            harbor_output_dir=tmp_path,
            environment="docker",
            num_workers=3,
            n_attempts=5,
            max_retries=2,
            agent_kwargs=["version=1.2.3"],
            job_name="official-run",
            upload=True,
            public=True,
        )

        assert cmd[cmd.index("--env") + 1] == "docker"
        assert cmd[cmd.index("--n-concurrent") + 1] == "3"
        assert cmd[cmd.index("--n-attempts") + 1] == "5"
        assert cmd[cmd.index("--max-retries") + 1] == "2"
        assert cmd[cmd.index("--agent-kwarg") + 1] == "version=1.2.3"
        assert cmd[cmd.index("--job-name") + 1] == "official-run"
        assert cmd[-2:] == ["--upload", "--public"]

    def test_rejects_invalid_controls(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="num_workers"):
            build_harbor_command(
                model="test/model",
                dataset="org/dataset",
                harbor_output_dir=tmp_path,
                num_workers=0,
            )
        with pytest.raises(ValueError, match="n_attempts"):
            build_harbor_command(
                model="test/model",
                dataset="org/dataset",
                harbor_output_dir=tmp_path,
                n_attempts=0,
            )
        with pytest.raises(ValueError, match="max_retries"):
            build_harbor_command(
                model="test/model",
                dataset="org/dataset",
                harbor_output_dir=tmp_path,
                max_retries=-1,
            )
        with pytest.raises(ValueError, match="require upload"):
            build_harbor_command(
                model="test/model",
                dataset="org/dataset",
                harbor_output_dir=tmp_path,
                public=True,
            )
