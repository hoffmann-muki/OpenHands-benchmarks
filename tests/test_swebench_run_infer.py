import json

from benchmarks.swebench.run_infer import retain_swebench_failure_predictions


def _record(instance_id: str, patch: str, error: str | None) -> str:
    return json.dumps(
        {
            "instance_id": instance_id,
            "attempt": 1,
            "test_result": {"git_patch": patch},
            "error": error,
        }
    )


def test_retains_success_and_restores_failed_swebench_predictions(tmp_path):
    output_path = tmp_path / "output.jsonl"
    success = _record("repo__success-1", "success patch", None)
    failure = _record("repo__failure-2", "failure patch", "timed out")
    empty_failure = _record("repo__empty-3", "", "iteration limit")
    output_path.write_text(f"{success}\n")
    (tmp_path / "output.critic_attempt_1.jsonl").write_text(
        f"{success}\n{failure}\n{empty_failure}\n"
    )

    recovered = retain_swebench_failure_predictions(output_path, 1)

    assert recovered == 2
    results = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [result["instance_id"] for result in results] == [
        "repo__success-1",
        "repo__failure-2",
        "repo__empty-3",
    ]
    assert [result["test_result"]["git_patch"] for result in results] == [
        "success patch",
        "failure patch",
        "",
    ]


def test_prefers_latest_failed_attempt_and_is_idempotent(tmp_path):
    output_path = tmp_path / "output.jsonl"
    output_path.write_text("")
    first = _record("repo__failure-1", "first patch", "first failure")
    second = _record("repo__failure-1", "second patch", "second failure")
    (tmp_path / "output.critic_attempt_1.jsonl").write_text(f"{first}\n")
    (tmp_path / "output.critic_attempt_2.jsonl").write_text(f"{second}\n")

    assert retain_swebench_failure_predictions(output_path, 2) == 1
    result = json.loads(output_path.read_text())
    assert result["test_result"]["git_patch"] == "second patch"
    assert retain_swebench_failure_predictions(output_path, 2) == 0
