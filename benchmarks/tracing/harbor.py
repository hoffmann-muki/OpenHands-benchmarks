"""Harbor bridge utilities for normalized benchmark traces."""

from __future__ import annotations

import json
import os
import shutil
import tomllib
import uuid
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from benchmarks.tracing import (
    CONTRACT_VERSION,
    ContractValidator,
    attempt_directory,
    write_run_index,
)
from benchmarks.tracing.models import JsonObject
from benchmarks.tracing.storage import read_jsonl, utc_now


_ALLOCATION_FILENAME = ".harbor-attempts.json"
_LOCK_FILENAME = ".harbor-attempts.lock"


@dataclass(frozen=True, slots=True)
class HarborTraceAttempt:
    """Identity and paths assigned to one Harbor agent execution."""

    instance_id: str
    attempt: int
    agent_timeout_seconds: float
    container_image: str
    container_root: Path


@dataclass(frozen=True, slots=True)
class HarborTraceRun:
    """Private normalized trace root owned by one benchmark invocation."""

    id: str
    root: Path
    created_at: str
    benchmark: str


def create_harbor_trace_run(
    base_directory: Path,
    benchmark: str,
) -> HarborTraceRun:
    """Preflight and create a private trace root before Harbor starts."""

    expanded = base_directory.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"Trace base cannot be a symbolic link: {expanded}")
    base = expanded.resolve()
    existed = base.exists()
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not base.is_dir() or base.is_symlink():
        raise ValueError(f"Trace base must be a real directory: {base}")
    if not existed and os.name != "nt":
        base.chmod(0o700)
    run_id = f"trace-run-{uuid.uuid4().hex}"
    root = base / run_id
    root.mkdir(mode=0o700)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Trace root must be a real directory: {root}")
    if os.name != "nt":
        root.chmod(0o700)
    return HarborTraceRun(
        id=run_id,
        root=root,
        created_at=utc_now(),
        benchmark=benchmark,
    )


def allocate_harbor_trace_attempt(
    *,
    trace_root: Path,
    instance_id: str,
    agent_timeout_seconds: float,
    container_image: str,
) -> HarborTraceAttempt:
    """Atomically assign an attempt ordinal before Harbor starts the agent."""

    if not instance_id:
        raise ValueError("Harbor trace instance_id cannot be empty")
    if agent_timeout_seconds <= 0:
        raise ValueError("Harbor trace agent timeout must be positive")
    if not container_image:
        raise ValueError("Harbor trace container image cannot be empty")
    resolved_root = trace_root.resolve()
    if not resolved_root.is_dir() or trace_root.is_symlink():
        raise ValueError(f"Harbor trace root must be a real directory: {trace_root}")

    with _allocation_lock(resolved_root):
        allocation_path = resolved_root / _ALLOCATION_FILENAME
        allocations = _read_allocations(allocation_path)
        attempt = allocations.get(instance_id, 0) + 1
        allocations[instance_id] = attempt
        _atomic_write_json(allocation_path, allocations)

    return HarborTraceAttempt(
        instance_id=instance_id,
        attempt=attempt,
        agent_timeout_seconds=agent_timeout_seconds,
        container_image=container_image,
        container_root=Path("/logs/agent/benchmark-trace"),
    )


def promote_harbor_trace_attempt(
    *,
    logs_dir: Path,
    trace_root: Path,
    attempt: HarborTraceAttempt,
) -> Path:
    """Move one sanitized in-container trace into the canonical run layout."""

    source = attempt_directory(
        logs_dir / attempt.container_root.name,
        attempt.instance_id,
        attempt.attempt,
    )
    destination = attempt_directory(
        trace_root.resolve(),
        attempt.instance_id,
        attempt.attempt,
    )
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"Harbor agent trace is missing: {source}")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Harbor agent trace contains a symbolic link")
    if destination.exists():
        raise FileExistsError(f"Harbor trace attempt already exists: {destination}")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    if os.name != "nt":
        destination.chmod(0o700)
    shutil.rmtree(logs_dir / attempt.container_root.name, ignore_errors=True)
    return destination


def finalize_harbor_trace_run(
    *,
    trace_root: Path,
    run_id: str,
    benchmark: str,
    framework: str,
    created_at: str,
    selected_instance_ids: Sequence[str] | None,
    expected_instance_count: int,
    expected_attempts_per_instance: int,
    selection_strategy: str,
) -> Path:
    """Validate promoted attempts and write one deterministic run index."""

    root = trace_root.resolve()
    _remove_allocator_files(root)
    validator = ContractValidator()
    attempts: list[JsonObject] = []
    observed: dict[str, list[int]] = {}

    instances_root = root / "instances"
    for manifest_path in sorted(instances_root.glob("*/attempt-*/manifest.json")):
        attempt_dir = manifest_path.parent
        report = validator.validate_attempt(attempt_dir)
        if not report.valid:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            continue
        instance_id = manifest.get("instance_id")
        attempt_number = manifest.get("attempt")
        if (
            manifest.get("run_id") != run_id
            or manifest.get("benchmark") != benchmark
            or manifest.get("framework") != framework
            or not isinstance(instance_id, str)
            or not isinstance(attempt_number, int)
        ):
            continue
        events = read_jsonl(
            attempt_dir / "events.jsonl",
            allow_torn_final_line=False,
        ).records
        terminal_status = next(
            (
                event["status"]
                for event in reversed(events)
                if event["event_type"] == "attempt.end"
            ),
            "degraded",
        )
        health = json.loads((attempt_dir / "health.json").read_text(encoding="utf-8"))
        status = terminal_status if health.get("status") == "healthy" else "degraded"
        observed.setdefault(instance_id, []).append(attempt_number)
        attempts.append(
            {
                "trace_id": str(manifest["trace_id"]),
                "instance_id": instance_id,
                "attempt": attempt_number,
                "path": attempt_dir.relative_to(root).as_posix(),
                "status": str(status),
            }
        )

    instance_ids = (
        list(selected_instance_ids)
        if selected_instance_ids is not None
        else sorted(observed)
    )
    if len(instance_ids) != expected_instance_count:
        raise ValueError(
            "Trace run index omitted because the observed instance count "
            f"{len(instance_ids)} does not match {expected_instance_count}"
        )
    if set(instance_ids) != set(observed):
        raise ValueError(
            "Trace run index omitted because selected instances lack finalized traces"
        )
    if any(
        len(attempt_numbers) < expected_attempts_per_instance
        for attempt_numbers in observed.values()
    ):
        raise ValueError(
            "Trace run index omitted because an instance lacks a requested attempt"
        )

    order = {instance_id: index for index, instance_id in enumerate(instance_ids)}

    def attempt_order(value: JsonObject) -> tuple[int, int]:
        instance_id = value.get("instance_id")
        attempt_number = value.get("attempt")
        if not isinstance(instance_id, str) or not isinstance(attempt_number, int):
            raise ValueError("Validated trace attempt identity is invalid")
        return order[instance_id], attempt_number

    attempts.sort(key=attempt_order)
    selection: JsonObject = {
        "strategy": selection_strategy,
        "requested_count": len(instance_ids),
        "instance_ids": [value for value in instance_ids],
    }
    document: JsonObject = {
        "schema_version": "benchmark-trace/v1",
        "contract": {
            "name": "benchmark-trace",
            "version": CONTRACT_VERSION,
            "schema_digest": validator.schema_digest,
        },
        "run_id": run_id,
        "benchmark": benchmark,
        "framework": framework,
        "created_at": created_at,
        "finalized_at": utc_now(),
        "selection": selection,
        "attempts": [value for value in attempts],
    }
    return write_run_index(root, document, validator=validator)


def trace_instance_id_from_trial_config(logs_dir: Path) -> str:
    """Read Harbor's resolved package task and return its canonical short name."""

    config = _trial_config(logs_dir)
    task = config.get("task")
    if not isinstance(task, dict):
        raise ValueError("Harbor trial config has no task object")
    name = task.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Harbor trial config has no task name")
    return name.split("/", 1)[-1]


def trace_instance_ids_from_job(jobs_dir: Path, job_name: str) -> list[str]:
    """Return Harbor's resolved task order with repeated attempts collapsed."""

    value = json.loads((jobs_dir / job_name / "lock.json").read_text(encoding="utf-8"))
    trials = value.get("trials") if isinstance(value, dict) else None
    if not isinstance(trials, list):
        raise ValueError("Harbor job lock has no resolved trials")

    instance_ids: list[str] = []
    for trial in trials:
        task = trial.get("task") if isinstance(trial, dict) else None
        name = task.get("name") if isinstance(task, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError("Harbor job lock has a trial without a task name")
        instance_id = name.split("/", 1)[-1]
        if instance_id not in instance_ids:
            instance_ids.append(instance_id)
    if not instance_ids:
        raise ValueError("Harbor job lock selected no tasks")
    return instance_ids


def trace_agent_timeout_from_trial_config(logs_dir: Path) -> float:
    """Resolve the effective Harbor agent deadline from cached task metadata."""

    config = _trial_config(logs_dir)
    task = config.get("task")
    agent = config.get("agent")
    if not isinstance(task, dict):
        raise ValueError("Harbor trial config has no task object")
    override = agent.get("override_timeout_sec") if isinstance(agent, dict) else None
    if isinstance(override, int | float) and override > 0:
        timeout = float(override)
    else:
        task_path = _resolved_task_path(task)
        with (task_path / "task.toml").open("rb") as file:
            task_document = tomllib.load(file)
        task_agent = task_document.get("agent")
        timeout_value = (
            task_agent.get("timeout_sec") if isinstance(task_agent, dict) else None
        )
        if not isinstance(timeout_value, int | float) or timeout_value <= 0:
            raise ValueError("Harbor task has no positive agent timeout")
        timeout = float(timeout_value)

    multiplier = config.get("agent_timeout_multiplier")
    if multiplier is None:
        multiplier = config.get("timeout_multiplier", 1)
    if not isinstance(multiplier, int | float) or multiplier <= 0:
        raise ValueError("Harbor trial timeout multiplier must be positive")
    return timeout * float(multiplier)


def trace_container_image_from_trial_config(logs_dir: Path) -> str:
    """Resolve Harbor's task-container image from cached task metadata."""

    task = _trial_config(logs_dir).get("task")
    if not isinstance(task, dict):
        raise ValueError("Harbor trial config has no task object")
    with (_resolved_task_path(task) / "task.toml").open("rb") as file:
        task_document = tomllib.load(file)
    environment = task_document.get("environment")
    image = environment.get("docker_image") if isinstance(environment, dict) else None
    if not isinstance(image, str) or not image:
        raise ValueError("Harbor task has no Docker image")
    return image


def _trial_config(logs_dir: Path) -> dict[str, Any]:
    path = logs_dir.parent / "config.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Harbor trial config must be an object: {path}")
    return value


def _resolved_task_path(task: dict[str, Any]) -> Path:
    name = task.get("name")
    reference = task.get("ref")
    if (
        not isinstance(name, str)
        or "/" not in name
        or not isinstance(reference, str)
        or not reference.startswith("sha256:")
    ):
        raise ValueError("Harbor package task is not pinned to a digest")
    # Harbor is an optional dependency outside Terminal-Bench execution.
    from harbor.models.task.id import (  # pyright: ignore[reportMissingImports]
        PackageTaskId,
    )

    organization, task_name = name.split("/", 1)
    return PackageTaskId(
        org=organization,
        name=task_name,
        ref=reference,
    ).get_local_path()


@contextmanager
def _allocation_lock(root: Path) -> Iterator[None]:
    import fcntl

    path = root / _LOCK_FILENAME
    with path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _read_allocations(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, int) and item >= 0
        for key, item in value.items()
    ):
        raise ValueError("Harbor trace attempt allocation state is invalid")
    return value


def _atomic_write_json(path: Path, value: dict[str, int]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if os.name != "nt":
        temporary.chmod(0o600)
    temporary.replace(path)


def _remove_allocator_files(root: Path) -> None:
    for name in (_ALLOCATION_FILENAME, _LOCK_FILENAME):
        (root / name).unlink(missing_ok=True)
