"""Run OpenHands on Terminal-Bench 2.1 through the official Harbor harness."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import signal
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import IO, Any, Callable, Sequence

from benchmarks.terminalbench.config import (
    HARBOR_DEFAULTS,
    INFER_DEFAULTS,
    TERMINAL_BENCH_TASK_COUNT,
)
from benchmarks.terminalbench.eval_infer import process_terminalbench_results
from benchmarks.utils.harbor import (
    HarborCredentialMode,
    build_harbor_command,
    check_harbor_installed as _check_harbor_installed,
    convert_harbor_to_eval_output,
    run_harbor_evaluation as _run_harbor_evaluation,
)
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import LLM, get_logger


logger = get_logger(__name__)

OUTPUT_FILENAME = "output.jsonl"
REPORT_FILENAME = "output.report.json"
MANIFEST_FILENAME = "manifest.json"
METADATA_FILENAME = "metadata.json"
STDOUT_FILENAME = "harbor.stdout.log"
STDERR_FILENAME = "harbor.stderr.log"
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
OUTPUT_TAIL_LINES = 200


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _default_run_id(now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(timezone.utc)).isoformat()
    return f"terminal-bench-2.1-{re.sub(r'[:+.]', '-', timestamp)}"


def installed_openhands_sdk_version() -> str:
    """Return the SDK version Harbor should install inside each task."""
    return version("openhands-sdk")


def build_parser(default_agent_version: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenHands on Terminal-Bench 2.1 with the official Harbor harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run one task as a safe smoke evaluation
    uv run terminalbench-infer .llm_config/claude.json

    # Select tasks or increase the smoke sample
    uv run terminalbench-infer .llm_config/claude.json --task-id task-name
    uv run terminalbench-infer .llm_config/claude.json --n-limit 5 --num-workers 2

    # Enforce the official full-dataset leaderboard protocol
    uv run terminalbench-infer .llm_config/claude.json --leaderboard --num-workers 4
        """,
    )
    parser.add_argument(
        "llm_config_path",
        help="Path to a JSON OpenHands LLM configuration file",
    )
    parser.add_argument(
        "--dataset",
        default=INFER_DEFAULTS["dataset"],
        help="Harbor dataset name",
    )
    parser.add_argument(
        "--output-dir",
        default=INFER_DEFAULTS["output_dir"],
        help="Base output directory for evaluation results",
    )
    parser.add_argument(
        "--num-workers",
        type=_positive_int,
        default=INFER_DEFAULTS["num_workers"],
        help="Number of concurrent Harbor trials",
    )
    parser.add_argument(
        "--n-attempts",
        type=_positive_int,
        default=INFER_DEFAULTS["n_attempts"],
        help="Number of independent Harbor attempts per task",
    )
    parser.add_argument(
        "--n-limit",
        type=_positive_int,
        default=INFER_DEFAULTS["n_limit"],
        help="Maximum tasks after filtering; defaults to one for smoke runs",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Remove the smoke task limit without enabling leaderboard upload",
    )
    parser.add_argument(
        "--max-retries",
        type=_non_negative_int,
        default=INFER_DEFAULTS["max_retries"],
        help="Infrastructure retries per Harbor trial",
    )
    parser.add_argument(
        "--environment",
        default=INFER_DEFAULTS["environment"],
        help="Harbor environment implementation",
    )
    parser.add_argument(
        "--openhands-version",
        default=default_agent_version,
        help="OpenHands SDK version installed by Harbor",
    )
    parser.add_argument(
        "--harbor-bin",
        default=HARBOR_DEFAULTS["harbor_executable"],
        help="Harbor executable",
    )
    parser.add_argument(
        "--run-id",
        default=_default_run_id(),
        help="Stable Harbor job and local artifact identifier",
    )
    parser.add_argument(
        "--select",
        help="Text file containing task IDs to run, one per line",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        help="Specific task ID to run; repeatable",
    )
    parser.add_argument("--note", help="Optional run note recorded in metadata")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the completed Harbor job",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make an uploaded Harbor job public; requires --upload",
    )
    parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Run all 89 tasks with at least five attempts and public upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the credential-free Harbor command without executing it",
    )
    parser.add_argument(
        "--skip-harbor",
        action="store_true",
        help="Only regenerate output.jsonl and its report for an existing --run-id",
    )
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    default_agent_version: str | None = None,
) -> argparse.Namespace:
    raw_args = list(argv if argv is not None else sys.argv[1:])
    parser = build_parser(default_agent_version or installed_openhands_sdk_version())
    args = parser.parse_args(raw_args)

    if args.select and args.task_id:
        parser.error("--select and --task-id cannot be combined")
    if not SAFE_RUN_ID.fullmatch(args.run_id) or args.run_id in {".", ".."}:
        parser.error(
            "--run-id may contain only letters, numbers, dots, underscores, and hyphens"
        )
    if not args.environment.strip():
        parser.error("--environment cannot be empty")
    if not args.openhands_version.strip():
        parser.error("--openhands-version cannot be empty")
    if not args.harbor_bin.strip():
        parser.error("--harbor-bin cannot be empty")
    if args.all_tasks and "--n-limit" in raw_args:
        parser.error("--all-tasks cannot be combined with --n-limit")
    if args.all_tasks:
        args.n_limit = None

    if args.leaderboard:
        if args.dataset != INFER_DEFAULTS["dataset"]:
            parser.error(
                "--leaderboard requires the official Terminal-Bench 2.1 dataset"
            )
        if args.select or args.task_id or "--n-limit" in raw_args:
            parser.error(
                "--leaderboard must run the complete Terminal-Bench 2.1 dataset"
            )
        if "--n-attempts" in raw_args and args.n_attempts < 5:
            parser.error("--leaderboard requires at least five attempts per task")
        args.n_limit = None
        args.n_attempts = max(args.n_attempts, 5)
        args.upload = True
        args.public = True

    if args.public and not args.upload:
        parser.error("--public requires --upload")

    if args.skip_harbor:
        if "--run-id" not in raw_args:
            parser.error("--skip-harbor requires the existing --run-id")
        if args.dry_run or args.leaderboard or args.upload or args.public:
            parser.error(
                "--skip-harbor cannot be combined with execution or upload flags"
            )

    return args


def check_harbor_installed(harbor_executable: str = "harbor") -> bool:
    """Return whether the Harbor CLI is installed and executable."""
    return _check_harbor_installed(harbor_executable, probe_arg="--version")


def load_task_ids_from_file(filepath: str) -> list[str]:
    """Load task IDs from a text file, ignoring comments and blank lines."""
    task_ids: list[str] = []
    with open(filepath) as task_file:
        for line in task_file:
            value = line.strip()
            if value and not value.startswith("#"):
                task_ids.append(value)
    if not task_ids:
        raise ValueError(f"Task selection file contains no task IDs: {filepath}")
    return task_ids


def resolve_task_ids(args: argparse.Namespace) -> list[str] | None:
    if args.select:
        task_ids = load_task_ids_from_file(args.select)
        logger.info(f"Loaded {len(task_ids)} task IDs from {args.select}")
        return task_ids
    if args.task_id:
        task_ids = list(args.task_id)
        logger.info(f"Running {len(task_ids)} specified task IDs")
        return task_ids
    return None


def build_terminal_bench_command(
    *,
    args: argparse.Namespace,
    model: str,
    harbor_output_dir: Path,
    task_ids: list[str] | None,
) -> list[str]:
    """Build the exact credential-free Harbor command recorded in the manifest."""
    return build_harbor_command(
        model=model,
        dataset=args.dataset,
        harbor_output_dir=harbor_output_dir,
        harbor_executable=args.harbor_bin,
        agent_name=HARBOR_DEFAULTS["agent_name"],
        environment=args.environment,
        num_workers=args.num_workers,
        n_attempts=args.n_attempts,
        max_retries=args.max_retries,
        task_ids=task_ids,
        n_limit=args.n_limit,
        task_filter_flag="--include-task-name",
        agent_kwargs=[f"version={args.openhands_version}"],
        job_name=args.run_id,
        upload=args.upload,
        public=args.public,
    )


def run_harbor_evaluation(
    llm: LLM,
    dataset: str,
    output_dir: str,
    num_workers: int = 1,
    n_attempts: int = 1,
    max_retries: int = 0,
    environment: str = "docker",
    openhands_version: str | None = None,
    job_name: str | None = None,
    task_ids: list[str] | None = None,
    n_limit: int | None = None,
    upload: bool = False,
    public: bool = False,
    harbor_executable: str = "harbor",
    subprocess_run: Callable[..., Any] = subprocess.run,
) -> Path:
    """Run Harbor with secrets inherited through the process environment."""
    agent_kwargs = (
        [f"version={openhands_version}"] if openhands_version is not None else None
    )
    return _run_harbor_evaluation(
        llm=llm,
        dataset=dataset,
        output_dir=output_dir,
        harbor_executable=harbor_executable,
        agent_name=HARBOR_DEFAULTS["agent_name"],
        environment=environment,
        num_workers=num_workers,
        n_attempts=n_attempts,
        max_retries=max_retries,
        task_ids=task_ids,
        n_limit=n_limit,
        task_filter_flag="--include-task-name",
        agent_kwargs=agent_kwargs,
        job_name=job_name,
        upload=upload,
        public=public,
        credential_mode=HarborCredentialMode.PROCESS_ENV,
        subprocess_run=subprocess_run,
    )


def _stream_pipe(
    source: IO[str],
    console: IO[str],
    log_file: IO[str],
    tail: deque[str],
) -> None:
    for chunk in iter(source.readline, ""):
        console.write(chunk)
        console.flush()
        log_file.write(chunk)
        log_file.flush()
        tail.append(chunk)


def make_streaming_runner(
    stdout_path: Path,
    stderr_path: Path,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Create a subprocess.run-compatible Harbor runner with durable live logs."""

    def run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text
        stdout_tail: deque[str] = deque(maxlen=OUTPUT_TAIL_LINES)
        stderr_tail: deque[str] = deque(maxlen=OUTPUT_TAIL_LINES)

        with (
            open(stdout_path, "w") as stdout_log,
            open(stderr_path, "w") as stderr_log,
        ):
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
            if process.stdout is None or process.stderr is None:
                process.kill()
                raise RuntimeError("Harbor subprocess did not expose output streams")

            stdout_thread = threading.Thread(
                target=_stream_pipe,
                args=(process.stdout, sys.stdout, stdout_log, stdout_tail),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=_stream_pipe,
                args=(process.stderr, sys.stderr, stderr_log, stderr_tail),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()

            try:
                return_code = process.wait()
            except BaseException:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
            finally:
                stdout_thread.join()
                stderr_thread.join()

        return subprocess.CompletedProcess(
            args=cmd,
            returncode=return_code,
            stdout="".join(stdout_tail),
            stderr="".join(stderr_tail),
        )

    return run


def harbor_version(harbor_executable: str) -> str:
    try:
        result = subprocess.run(
            [harbor_executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"Harbor is not executable: {harbor_executable}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Harbor version check timed out") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Could not execute Harbor: {detail}")
    return result.stdout.strip() or result.stderr.strip() or "unknown"


def preflight(args: argparse.Namespace, llm: LLM) -> str:
    """Fail before a paid run when required local infrastructure is unavailable."""
    resolved_harbor_version = harbor_version(args.harbor_bin)

    if args.environment == "docker":
        docker = shutil.which("docker")
        if docker is None:
            raise RuntimeError("Docker is required for the default Harbor environment")
        result = subprocess.run(
            [docker, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Docker is installed but not running: {detail}")

    if not llm.api_key:
        raise RuntimeError("The OpenHands Harbor agent requires an LLM API key")

    return resolved_harbor_version


def write_json(path: Path, data: dict[str, object]) -> None:
    """Atomically write a JSON artifact."""
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(data, indent=2) + "\n")
    temporary_path.replace(path)


def load_llm(path: str) -> LLM:
    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"LLM config file does not exist: {config_path}")
    return LLM.model_validate_json(config_path.read_text())


def build_output_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir).resolve() / "terminal-bench-2.1" / "runs" / args.run_id


def postprocess_harbor_results(output_dir: Path) -> tuple[Path, Path]:
    harbor_output_dir = output_dir / "harbor_output"
    output_path = output_dir / OUTPUT_FILENAME
    report_path = output_dir / REPORT_FILENAME
    convert_harbor_to_eval_output(
        harbor_output_dir=harbor_output_dir,
        eval_output_path=output_path,
    )
    process_terminalbench_results(str(output_path), str(report_path))
    generate_cost_report(str(output_path))
    return output_path, report_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        llm = load_llm(args.llm_config_path)
        task_ids = resolve_task_ids(args)
    except Exception as error:
        logger.error(str(error))
        raise SystemExit(1) from error

    logger.info(f"Using LLM: {llm.model}")
    output_dir = build_output_dir(args)
    harbor_output_dir = output_dir / "harbor_output"
    command = build_terminal_bench_command(
        args=args,
        model=llm.model,
        harbor_output_dir=harbor_output_dir,
        task_ids=task_ids,
    )

    if args.dry_run:
        print(json.dumps(command))
        return

    if args.skip_harbor:
        try:
            output_path, report_path = postprocess_harbor_results(output_dir)
        except Exception as error:
            logger.error(f"Post-processing failed: {error}")
            raise SystemExit(1) from error
        print(
            json.dumps(
                {
                    "output_json": str(output_path),
                    "report_json": str(report_path),
                    "harbor_jobs": str(harbor_output_dir),
                }
            )
        )
        return

    try:
        resolved_harbor_version = preflight(args, llm)
    except Exception as error:
        logger.error(f"Preflight failed: {error}")
        raise SystemExit(1) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / STDOUT_FILENAME
    stderr_path = output_dir / STDERR_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    started_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "benchmark": "terminal-bench",
        "dataset": args.dataset,
        "official_task_count": TERMINAL_BENCH_TASK_COUNT,
        "official_runner": "harbor",
        "harbor_version": resolved_harbor_version,
        "model": llm.model,
        "agent": HARBOR_DEFAULTS["agent_name"],
        "agent_version": args.openhands_version,
        "environment": args.environment,
        "task_ids": task_ids or [],
        "max_tasks": args.n_limit,
        "all_tasks": args.n_limit is None,
        "attempts": args.n_attempts,
        "concurrency": args.num_workers,
        "max_retries": args.max_retries,
        "upload": args.upload,
        "public": args.public,
        "leaderboard": args.leaderboard,
        "command": command,
        "harbor_jobs": str(harbor_output_dir),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "started_at": started_at,
        "status": "running",
    }
    write_json(manifest_path, manifest)
    write_json(
        metadata_path,
        {
            "model": llm.model,
            "dataset": args.dataset,
            "timestamp": started_at,
            "harbor_agent": HARBOR_DEFAULTS["agent_name"],
            "agent_version": args.openhands_version,
            "run_id": args.run_id,
            "note": args.note,
        },
    )

    try:
        run_harbor_evaluation(
            llm=llm,
            dataset=args.dataset,
            output_dir=str(output_dir),
            num_workers=args.num_workers,
            n_attempts=args.n_attempts,
            max_retries=args.max_retries,
            environment=args.environment,
            openhands_version=args.openhands_version,
            job_name=args.run_id,
            task_ids=task_ids,
            n_limit=args.n_limit,
            upload=args.upload,
            public=args.public,
            harbor_executable=args.harbor_bin,
            subprocess_run=make_streaming_runner(stdout_path, stderr_path),
        )
        output_path, report_path = postprocess_harbor_results(output_dir)
    except BaseException as error:
        write_json(
            manifest_path,
            {
                **manifest,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "error": str(error),
            },
        )
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        logger.error(f"Evaluation failed: {error}")
        raise SystemExit(1) from error

    write_json(
        manifest_path,
        {
            **manifest,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "exit_code": 0,
            "status": "completed",
            "output_json": str(output_path),
            "report_json": str(report_path),
        },
    )
    logger.info("Terminal-Bench evaluation completed")
    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "report_json": str(report_path),
                "manifest_json": str(manifest_path),
                "harbor_jobs": str(harbor_output_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
