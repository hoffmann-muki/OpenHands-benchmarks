import subprocess
from pathlib import Path

import pytest

from benchmark_agents import provenance


@pytest.fixture
def benchmark_repo(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Path:
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    subprocess.run(
        ["git", "-C", tmp_path, "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", tmp_path, "config", "user.name", "Test"], check=True)
    (tmp_path / "source.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", tmp_path, "add", "source.py"], check=True)
    subprocess.run(["git", "-C", tmp_path, "commit", "-qm", "initial"], check=True)
    monkeypatch.setattr(provenance, "BENCHMARKS_SOURCE", tmp_path)
    return tmp_path


def test_benchmark_provenance_ignores_generated_traces(
    benchmark_repo: Path,
) -> None:
    tmp_path = benchmark_repo

    trace = tmp_path / ".benchmark-traces" / "trace-run-test"
    trace.mkdir(parents=True)
    (trace / "events.jsonl").write_text("{}\n", encoding="utf-8")

    assert provenance.openhands_benchmarks_source_commit(require_clean=True) == (
        subprocess.run(
            ["git", "-C", tmp_path, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def test_benchmark_provenance_rejects_other_untracked_files(
    benchmark_repo: Path,
) -> None:
    tmp_path = benchmark_repo
    (tmp_path / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="uncommitted changes"):
        provenance.openhands_benchmarks_source_commit(require_clean=True)
