"""Immutable source identities shared by benchmark agent adapters."""

import re
import subprocess
from pathlib import Path


OPENHANDS_SDK_SOURCE = (
    Path(__file__).resolve().parents[1] / "vendor" / "software-agent-sdk"
)
BENCHMARKS_SOURCE = Path(__file__).resolve().parents[1]
FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def openhands_sdk_source_commit() -> str:
    """Return the exact clean vendored SDK revision used by benchmark tasks."""
    commit = subprocess.run(
        ["git", "-C", str(OPENHANDS_SDK_SOURCE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not FULL_GIT_SHA.fullmatch(commit):
        raise RuntimeError(f"Invalid vendored OpenHands SDK revision: {commit}")

    status = subprocess.run(
        [
            "git",
            "-C",
            str(OPENHANDS_SDK_SOURCE),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError(
            "The vendored OpenHands SDK has uncommitted changes; benchmark "
            "runtime provenance would not be reproducible"
        )
    return commit


def openhands_benchmarks_source_commit(*, require_clean: bool = False) -> str:
    """Return the benchmark revision, optionally requiring an exact clean tree."""

    commit = subprocess.run(
        ["git", "-C", str(BENCHMARKS_SOURCE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not FULL_GIT_SHA.fullmatch(commit):
        raise RuntimeError(f"Invalid OpenHands-benchmarks revision: {commit}")
    if not require_clean:
        return commit
    status = subprocess.run(
        [
            "git",
            "-C",
            str(BENCHMARKS_SOURCE),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            ".",
            ":(exclude).benchmark-traces/**",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError(
            "OpenHands-benchmarks has uncommitted changes; tracing provenance "
            "would not identify the executed adapter exactly"
        )
    return commit
