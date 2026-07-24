"""Tests for Terminal-Bench benchmark module."""

import json
import os
import sys
from pathlib import Path

import pytest

from benchmarks.terminalbench.config import HARBOR_DEFAULTS, INFER_DEFAULTS
from benchmarks.terminalbench.eval_infer import process_terminalbench_results
from benchmarks.terminalbench.run_infer import (
    DEFAULT_TRACE_DIR,
    benchmark_process_env,
    build_output_dir,
    build_terminal_bench_command,
    convert_harbor_to_eval_output,
    load_task_ids_from_file,
    make_streaming_runner,
    parse_args,
    resolve_harbor_agent,
    run_harbor_evaluation,
)
from benchmarks.tracing.harbor import HarborTraceRun
from openhands.sdk import LLM


class TestProcessTerminalbenchResults:
    """Tests for the process_terminalbench_results function."""

    def test_empty_input(self, tmp_path: Path) -> None:
        """Test processing empty input file."""
        input_file = tmp_path / "empty.jsonl"
        output_file = tmp_path / "empty.report.json"
        input_file.write_text("")

        result = process_terminalbench_results(str(input_file), str(output_file))

        assert result["total_instances"] == 0
        assert result["completed_instances"] == 0
        assert result["resolved_instances"] == 0

    def test_single_completed_instance(self, tmp_path: Path) -> None:
        """Test processing a single completed instance."""
        input_file = tmp_path / "single.jsonl"
        output_file = tmp_path / "single.report.json"

        entry = {
            "instance_id": "hello-world",
            "test_result": {
                "trajectory_path": "/path/to/trajectory.json",
                "total_steps": 5,
                "final_metrics": {
                    "total_prompt_tokens": 1000,
                    "total_completion_tokens": 200,
                    "total_cost_usd": 0.01,
                },
            },
            "instruction": "Create hello.txt",
            "error": None,
            "history": [],
        }
        input_file.write_text(json.dumps(entry) + "\n")

        result = process_terminalbench_results(str(input_file), str(output_file))

        assert result["total_instances"] == 1
        assert result["completed_instances"] == 1
        # Without explicit passed=True, instance is unresolved
        assert result["unresolved_instances"] == 1
        assert result["resolved_instances"] == 0
        assert "hello-world" in result["completed_ids"]

    def test_resolved_instance(self, tmp_path: Path) -> None:
        """Test processing a resolved (passed=True) instance."""
        input_file = tmp_path / "resolved.jsonl"
        output_file = tmp_path / "resolved.report.json"

        entry = {
            "instance_id": "test-task",
            "test_result": {
                "passed": True,
                "total_steps": 10,
            },
            "instruction": "Do something",
            "error": None,
            "history": [],
        }
        input_file.write_text(json.dumps(entry) + "\n")

        result = process_terminalbench_results(str(input_file), str(output_file))

        assert result["resolved_instances"] == 1
        assert result["unresolved_instances"] == 0
        assert "test-task" in result["resolved_ids"]

    def test_instance_with_error(self, tmp_path: Path) -> None:
        """Test processing an instance with error."""
        input_file = tmp_path / "error.jsonl"
        output_file = tmp_path / "error.report.json"

        entry = {
            "instance_id": "error-task",
            "test_result": {},
            "instruction": "Do something",
            "error": "Runtime timeout",
            "history": [],
        }
        input_file.write_text(json.dumps(entry) + "\n")

        result = process_terminalbench_results(str(input_file), str(output_file))

        assert result["error_instances"] == 1
        assert result["incomplete_instances"] == 1
        assert result["completed_instances"] == 0
        assert "error-task" in result["error_ids"]

    def test_multiple_instances(self, tmp_path: Path) -> None:
        """Test processing multiple instances."""
        input_file = tmp_path / "multi.jsonl"
        output_file = tmp_path / "multi.report.json"

        entries = [
            {
                "instance_id": "task-1",
                "test_result": {"passed": True},
                "error": None,
            },
            {
                "instance_id": "task-2",
                "test_result": {"passed": False},
                "error": None,
            },
            {
                "instance_id": "task-3",
                "test_result": {},
                "error": "Failed",
            },
        ]
        input_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        result = process_terminalbench_results(str(input_file), str(output_file))

        assert result["total_instances"] == 3
        assert result["completed_instances"] == 2
        assert result["resolved_instances"] == 1
        assert result["unresolved_instances"] == 1
        assert result["error_instances"] == 1

    def test_aggregate_metrics(self, tmp_path: Path) -> None:
        """Test that metrics are aggregated correctly."""
        input_file = tmp_path / "metrics.jsonl"
        output_file = tmp_path / "metrics.report.json"

        entries = [
            {
                "instance_id": "task-1",
                "test_result": {
                    "final_metrics": {
                        "total_prompt_tokens": 1000,
                        "total_completion_tokens": 200,
                        "total_cost_usd": 0.01,
                    }
                },
                "metrics": {},
                "error": None,
            },
            {
                "instance_id": "task-2",
                "test_result": {},
                "metrics": {
                    "total_prompt_tokens": 2000,
                    "total_completion_tokens": 400,
                    "total_cost_usd": 0.02,
                },
                "error": None,
            },
        ]
        input_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        result = process_terminalbench_results(str(input_file), str(output_file))

        aggregate = result["aggregate_metrics"]
        assert aggregate["total_prompt_tokens"] == 3000
        assert aggregate["total_completion_tokens"] == 600
        assert abs(aggregate["total_cost_usd"] - 0.03) < 0.001

    def test_duplicate_instance_ids_ignored(self, tmp_path: Path) -> None:
        """Test that duplicate instance IDs are handled."""
        input_file = tmp_path / "dup.jsonl"
        output_file = tmp_path / "dup.report.json"

        entries = [
            {"instance_id": "task-1", "test_result": {}, "error": None},
            {"instance_id": "task-1", "test_result": {}, "error": None},  # Duplicate
        ]
        input_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        result = process_terminalbench_results(str(input_file), str(output_file))

        # Only first occurrence should be counted
        assert result["completed_instances"] == 1

    def test_report_file_written(self, tmp_path: Path) -> None:
        """Test that report file is written correctly."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.report.json"

        entry = {
            "instance_id": "task-1",
            "test_result": {"passed": True},
            "error": None,
        }
        input_file.write_text(json.dumps(entry) + "\n")

        process_terminalbench_results(str(input_file), str(output_file))

        assert output_file.exists()
        with open(output_file) as f:
            report = json.load(f)
        assert "total_instances" in report
        assert "resolved_ids" in report


class TestRunHarborEvaluation:
    """Tests for building Harbor invocation arguments."""

    def test_default_dataset_matches_harbor_registry(self) -> None:
        """Test that the default dataset name matches Harbor's published registry."""
        assert INFER_DEFAULTS["dataset"] == "terminal-bench/terminal-bench-2-1"

    def test_safe_smoke_defaults(self) -> None:
        args = parse_args(
            ["config.json"],
            default_agent_version="1.27.0",
        )

        assert args.n_limit == 1
        assert args.n_attempts == 1
        assert args.num_workers == 1
        assert args.max_retries == 0
        assert args.environment == "docker"
        assert args.all_tasks is False
        assert args.upload is False
        assert args.public is False
        assert args.leaderboard is False
        assert args.enable_delegation is True
        assert args.trace_dir == str(DEFAULT_TRACE_DIR)

    def test_default_model_does_not_require_a_config_path(self) -> None:
        args = parse_args([], default_agent_version="1.27.0")

        assert args.llm_config_path is None

    def test_delegation_can_be_disabled_explicitly(self) -> None:
        args = parse_args(
            ["config.json", "--disable-delegation"],
            default_agent_version="1.27.0",
        )

        assert args.enable_delegation is False
        assert (
            resolve_harbor_agent(args.enable_delegation)
            == HARBOR_DEFAULTS["agent_name"]
        )

    def test_leaderboard_mode_enforces_official_protocol(self) -> None:
        args = parse_args(
            ["config.json", "--leaderboard", "--num-workers", "4"],
            default_agent_version="1.27.0",
        )

        assert args.n_limit is None
        assert args.n_attempts == 5
        assert args.num_workers == 4
        assert args.upload is True
        assert args.public is True

    def test_all_tasks_removes_only_the_smoke_limit(self) -> None:
        args = parse_args(
            ["config.json", "--all-tasks"],
            default_agent_version="1.27.0",
        )

        assert args.n_limit is None
        assert args.n_attempts == 1
        assert args.upload is False
        assert args.public is False

        with pytest.raises(SystemExit):
            parse_args(
                ["config.json", "--all-tasks", "--n-limit", "5"],
                default_agent_version="1.27.0",
            )

    def test_output_directory_has_stable_benchmark_run_shape(
        self, tmp_path: Path
    ) -> None:
        args = parse_args(
            [
                "config.json",
                "--output-dir",
                str(tmp_path),
                "--run-id",
                "terminal-smoke",
            ],
            default_agent_version="1.27.0",
        )

        assert build_output_dir(args) == (
            tmp_path / "terminal-bench-2.1" / "runs" / "terminal-smoke"
        )

    def test_empty_task_selection_file_is_rejected(self, tmp_path: Path) -> None:
        selection = tmp_path / "tasks.txt"
        selection.write_text("\n# no tasks selected\n")

        with pytest.raises(ValueError, match="contains no task IDs"):
            load_task_ids_from_file(str(selection))

    @pytest.mark.parametrize(
        "extra_args",
        [
            ["--n-limit", "5"],
            ["--task-id", "task-a"],
            ["--n-attempts", "4"],
            ["--dataset", "terminal-bench@2.0"],
        ],
    )
    def test_leaderboard_mode_rejects_partial_or_nonofficial_runs(
        self, extra_args: list[str]
    ) -> None:
        with pytest.raises(SystemExit):
            parse_args(
                ["config.json", "--leaderboard", *extra_args],
                default_agent_version="1.27.0",
            )

    def test_build_command_is_reproducible_and_credential_free(
        self, tmp_path: Path
    ) -> None:
        args = parse_args(
            [
                "config.json",
                "--run-id",
                "terminal-smoke",
                "--task-id",
                "task-a",
                "--n-attempts",
                "3",
                "--num-workers",
                "2",
                "--max-retries",
                "1",
            ],
            default_agent_version="1.27.0",
        )

        cmd = build_terminal_bench_command(
            args=args,
            model="litellm_proxy/test-model",
            harbor_output_dir=tmp_path / "harbor_output",
            task_ids=["task-a"],
            sdk_commit="a" * 40,
        )

        assert cmd[:8] == [
            "harbor",
            "run",
            "-d",
            "terminal-bench/terminal-bench-2-1",
            "-a",
            HARBOR_DEFAULTS["delegating_agent_name"],
            "-m",
            "litellm_proxy/test-model",
        ]
        assert cmd[cmd.index("--agent-kwarg") + 1] == "version=1.27.0"
        agent_kwargs = [
            cmd[index + 1]
            for index, value in enumerate(cmd)
            if value == "--agent-kwarg"
        ]
        assert agent_kwargs == [
            "version=1.27.0",
            f"sdk_commit={'a' * 40}",
            "max_iterations=24",
            "temperature=0.1",
        ]
        assert cmd[cmd.index("--env") + 1] == "docker"
        assert cmd[cmd.index("--n-attempts") + 1] == "3"
        assert cmd[cmd.index("--n-concurrent") + 1] == "2"
        assert cmd[cmd.index("--max-retries") + 1] == "1"
        assert cmd[cmd.index("--job-name") + 1] == "terminal-smoke"
        assert "LLM_API_KEY" not in " ".join(cmd)

    def test_trace_command_passes_only_nonsecret_normalized_metadata(
        self, tmp_path: Path
    ) -> None:
        args = parse_args(
            [
                "config.json",
                "--trace-dir",
                str(tmp_path / "traces"),
            ],
            default_agent_version="1.27.0",
        )
        trace_run = HarborTraceRun(
            id="trace-run-test",
            root=tmp_path / "traces" / "trace-run-test",
            created_at="2026-07-20T12:34:56+00:00",
            benchmark="terminal-bench-2.1",
        )

        cmd = build_terminal_bench_command(
            args=args,
            model="litellm_proxy/test-model",
            harbor_output_dir=tmp_path / "harbor_output",
            task_ids=None,
            sdk_commit="a" * 40,
            benchmark_commit="b" * 40,
            trace_run=trace_run,
            harbor_version="0.20.0",
        )
        agent_kwargs = [
            cmd[index + 1]
            for index, value in enumerate(cmd)
            if value == "--agent-kwarg"
        ]

        assert args.trace_dir == str(tmp_path / "traces")
        assert f"trace_root={trace_run.root}" in agent_kwargs
        assert "trace_run_id=trace-run-test" in agent_kwargs
        assert "trace_benchmark=terminal-bench-2.1" in agent_kwargs
        assert "benchmark_commit=" + "b" * 40 in agent_kwargs
        assert "evaluation_workers=1" in agent_kwargs
        assert "benchmark_retries=0" in agent_kwargs
        assert "harbor_version=0.20.0" in agent_kwargs
        assert "API_KEY" not in " ".join(agent_kwargs)

    def test_skip_harbor_rejects_new_tracing(self) -> None:
        args = parse_args(
            ["config.json", "--skip-harbor", "--run-id", "existing"],
            default_agent_version="1.27.0",
        )
        assert args.trace_dir is None

        with pytest.raises(SystemExit):
            parse_args(
                [
                    "config.json",
                    "--skip-harbor",
                    "--run-id",
                    "existing",
                    "--trace-dir",
                    "/tmp/traces",
                ],
                default_agent_version="1.27.0",
            )

    def test_single_agent_opt_out_keeps_the_reproducible_adapter(
        self, tmp_path: Path
    ) -> None:
        args = parse_args(
            ["config.json", "--disable-delegation"],
            default_agent_version="1.27.0",
        )

        cmd = build_terminal_bench_command(
            args=args,
            model="litellm_proxy/test-model",
            harbor_output_dir=tmp_path / "harbor_output",
            task_ids=None,
            sdk_commit="a" * 40,
        )

        assert cmd[cmd.index("-a") + 1] == HARBOR_DEFAULTS["agent_name"]

    def test_run_harbor_evaluation_passes_filters_and_limits(
        self, tmp_path: Path
    ) -> None:
        """Test Harbor command includes task filters and n-limit for CI runs."""
        captured: dict[str, object] = {}

        def fake_run(
            cmd: list[str], capture_output: bool, text: bool, env=None, timeout=None
        ):
            captured["cmd"] = cmd
            captured["env"] = env
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "ok", "stderr": ""},
            )()

        harbor_output_dir = run_harbor_evaluation(
            llm=LLM(
                model="litellm_proxy/test-model",
                api_key="test-key",
                base_url="https://proxy.example.com",
            ),
            dataset=INFER_DEFAULTS["dataset"],
            output_dir=str(tmp_path),
            num_workers=3,
            n_attempts=5,
            max_retries=2,
            environment="docker",
            openhands_version="1.27.0",
            sdk_commit="a" * 40,
            job_name="terminal-test",
            task_ids=["task-a", "task-b"],
            n_limit=5,
            subprocess_run=fake_run,
        )

        expected_output_dir = tmp_path / "harbor_output"
        assert harbor_output_dir == expected_output_dir

        cmd = captured["cmd"]
        assert isinstance(cmd, list)
        assert cmd[:8] == [
            "harbor",
            "run",
            "-d",
            "terminal-bench/terminal-bench-2-1",
            "-a",
            HARBOR_DEFAULTS["delegating_agent_name"],
            "-m",
            "litellm_proxy/test-model",
        ]
        assert "--jobs-dir" in cmd
        assert str(expected_output_dir.resolve()) in cmd
        assert cmd.count("--include-task-name") == 2
        assert "task-a" in cmd
        assert "task-b" in cmd
        assert cmd[cmd.index("--n-concurrent") + 1] == "3"
        assert cmd[cmd.index("--n-tasks") + 1] == "5"
        assert cmd[cmd.index("--n-attempts") + 1] == "5"
        assert cmd[cmd.index("--max-retries") + 1] == "2"
        assert cmd[cmd.index("--env") + 1] == "docker"
        assert cmd[cmd.index("--agent-kwarg") + 1] == "version=1.27.0"
        agent_kwargs = [
            cmd[index + 1]
            for index, value in enumerate(cmd)
            if value == "--agent-kwarg"
        ]
        assert agent_kwargs == [
            "version=1.27.0",
            f"sdk_commit={'a' * 40}",
            "max_iterations=24",
            "temperature=0.1",
        ]
        assert cmd[cmd.index("--job-name") + 1] == "terminal-test"
        assert "--ae" not in cmd
        env = captured["env"]
        assert isinstance(env, dict)
        assert env["LLM_API_KEY"] == "test-key"
        assert env["LLM_BASE_URL"] == "https://proxy.example.com"
        assert str(Path(__file__).resolve().parents[1]) in env["PYTHONPATH"].split(
            os.pathsep
        )

    def test_benchmark_process_env_preserves_existing_pythonpath(self) -> None:
        result = benchmark_process_env({"PYTHONPATH": "/existing/path"})

        paths = result["PYTHONPATH"].split(os.pathsep)
        assert str(Path(__file__).resolve().parents[1]) in paths
        assert "/existing/path" in paths

    def test_streaming_runner_persists_both_output_streams(
        self, tmp_path: Path
    ) -> None:
        stdout_path = tmp_path / "stdout.log"
        stderr_path = tmp_path / "stderr.log"
        runner = make_streaming_runner(stdout_path, stderr_path)

        result = runner(
            [
                sys.executable,
                "-c",
                "import sys; print('out'); print('err', file=sys.stderr)",
            ],
            capture_output=True,
            text=True,
            env=None,
        )

        assert result.returncode == 0
        assert "out" in result.stdout
        assert "err" in result.stderr
        assert stdout_path.read_text() == "out\n"
        assert stderr_path.read_text().endswith("err\n")


class TestConvertHarborToEvalOutput:
    """Tests for convert_harbor_to_eval_output function."""

    def _create_harbor_structure(
        self, tmp_path: Path, trials: list[tuple[str, dict]]
    ) -> Path:
        """Create a mock Harbor output structure.

        Harbor stores results as:
            harbor_output/TIMESTAMP/TRIAL_NAME/result.json
        with a job-level result.json at harbor_output/TIMESTAMP/result.json
        """
        harbor_dir = tmp_path / "harbor_output"
        job_dir = harbor_dir / "2026-01-01__00-00-00"
        job_dir.mkdir(parents=True)

        # Create job-level result.json
        (job_dir / "result.json").write_text(json.dumps({"id": "test-job"}))

        for trial_name, trial_result in trials:
            trial_dir = job_dir / trial_name
            trial_dir.mkdir()
            (trial_dir / "result.json").write_text(json.dumps(trial_result))

        return harbor_dir

    def test_successful_trial_parsing(self, tmp_path: Path) -> None:
        """Test successful parsing of harbor trial result."""
        trial_result = {
            "task_name": "hello-world",
            "trial_name": "hello-world__abc123",
            "trial_uri": "file:///path/to/trial",
            "agent_result": {
                "n_input_tokens": 500,
                "n_output_tokens": 100,
                "cost_usd": 0.01,
            },
            "verifier_result": {"rewards": {"reward": 1.0}},
            "exception_info": None,
        }

        harbor_dir = self._create_harbor_structure(
            tmp_path, [("hello-world__abc123", trial_result)]
        )
        output_file = tmp_path / "output.jsonl"

        convert_harbor_to_eval_output(harbor_dir, output_file)

        assert output_file.exists()
        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) == 1
        assert entries[0]["instance_id"] == "hello-world"
        assert entries[0]["metrics"]["total_cost_usd"] == 0.01
        assert entries[0]["test_result"]["passed"] is True

    def test_failed_trial(self, tmp_path: Path) -> None:
        """Test parsing of a trial with reward 0."""
        trial_result = {
            "task_name": "test-task",
            "trial_name": "test-task__xyz",
            "agent_result": {
                "n_input_tokens": None,
                "n_output_tokens": None,
                "cost_usd": None,
            },
            "verifier_result": {"rewards": {"reward": 0.0}},
            "exception_info": None,
        }

        harbor_dir = self._create_harbor_structure(
            tmp_path, [("test-task__xyz", trial_result)]
        )
        output_file = tmp_path / "output.jsonl"

        convert_harbor_to_eval_output(harbor_dir, output_file)

        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) == 1
        assert entries[0]["test_result"]["passed"] is False
        assert entries[0]["metrics"]["total_cost_usd"] == 0.0

    def test_trial_with_exception(self, tmp_path: Path) -> None:
        """Test exception-only Harbor output is preserved for downstream reporting."""
        trial_result = {
            "task_name": "error-task",
            "trial_name": "error-task__err",
            "agent_result": {},
            "verifier_result": {},
            "exception_info": {"type": "TimeoutError", "message": "Agent timed out"},
        }

        harbor_dir = self._create_harbor_structure(
            tmp_path, [("error-task__err", trial_result)]
        )
        output_file = tmp_path / "output.jsonl"
        report_file = tmp_path / "report.json"

        convert_harbor_to_eval_output(harbor_dir, output_file)

        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert entries == [
            {
                "instance_id": "error-task",
                "error": "{'type': 'TimeoutError', 'message': 'Agent timed out'}",
                "test_result": {},
            }
        ]

        report = process_terminalbench_results(str(output_file), str(report_file))
        assert report["total_instances"] == 1
        assert report["completed_instances"] == 0
        assert report["error_instances"] == 1
        assert report["incomplete_ids"] == ["error-task"]

    def test_mixed_valid_and_exception_trials(self, tmp_path: Path) -> None:
        """Test handling mix of successful and exception trials."""
        trials = [
            (
                "good-task__abc",
                {
                    "task_name": "good-task",
                    "trial_name": "good-task__abc",
                    "agent_result": {},
                    "verifier_result": {"rewards": {"reward": 1.0}},
                    "exception_info": None,
                },
            ),
            (
                "bad-task__def",
                {
                    "task_name": "bad-task",
                    "trial_name": "bad-task__def",
                    "agent_result": {},
                    "verifier_result": {},
                    "exception_info": {"type": "Error", "message": "Failed"},
                },
            ),
        ]

        harbor_dir = self._create_harbor_structure(tmp_path, trials)
        output_file = tmp_path / "output.jsonl"
        convert_harbor_to_eval_output(harbor_dir, output_file)

        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) == 2
        success = [e for e in entries if e.get("error") is None]
        errors = [e for e in entries if e.get("error") is not None]
        assert len(success) == 1
        assert len(errors) == 1

    def test_empty_job_directory(self, tmp_path: Path) -> None:
        """Test handling of empty harbor job directory."""
        harbor_dir = tmp_path / "harbor_output"
        job_dir = harbor_dir / "2026-01-01__00-00-00"
        job_dir.mkdir(parents=True)
        (job_dir / "result.json").write_text(json.dumps({"id": "test"}))

        output_file = tmp_path / "output.jsonl"

        with pytest.raises(RuntimeError, match="No trial result files found"):
            convert_harbor_to_eval_output(harbor_dir, output_file)

    def test_missing_job_directory(self, tmp_path: Path) -> None:
        """Test handling when no job directory exists."""
        harbor_dir = tmp_path / "harbor_output"
        harbor_dir.mkdir()

        output_file = tmp_path / "output.jsonl"

        with pytest.raises(RuntimeError, match="No harbor job directory found"):
            convert_harbor_to_eval_output(harbor_dir, output_file)

    def test_discovery_finds_all_trials(self, tmp_path: Path) -> None:
        """Test that discovery finds all trial subdirectories."""
        trials = [
            (
                f"task-{i}__trial{i}",
                {
                    "task_name": f"task-{i}",
                    "trial_name": f"task-{i}__trial{i}",
                    "agent_result": {},
                    "verifier_result": {"rewards": {"reward": 0.0}},
                    "exception_info": None,
                },
            )
            for i in range(5)
        ]

        harbor_dir = self._create_harbor_structure(tmp_path, trials)
        output_file = tmp_path / "output.jsonl"

        convert_harbor_to_eval_output(harbor_dir, output_file)

        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) == 5
        instance_ids = {e["instance_id"] for e in entries}
        assert instance_ids == {f"task-{i}" for i in range(5)}
