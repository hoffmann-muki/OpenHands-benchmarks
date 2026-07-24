import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from benchmarks.tracing import CONTRACT_NAME, CONTRACT_VERSION, SCHEMA_VERSION


TRACE_ROOT = Path(__file__).parents[1] / "benchmarks" / "tracing"
SCHEMA_ROOT = TRACE_ROOT / "spec" / "v1"
FIXTURE_ROOT = TRACE_ROOT / "conformance" / "v1"
VALID_ROOT = FIXTURE_ROOT / "valid"

SCHEMA_FILES = {path.name: path for path in sorted(SCHEMA_ROOT.glob("*.schema.json"))}
CAPABILITY_CATEGORIES = {
    "agent.session",
    "model.turn",
    "provider.exchange",
    "tool.invocation",
    "tool.result",
    "tool.timing",
    "shell",
    "file",
    "search",
    "browser",
    "delegation",
    "context.compaction",
    "memory",
    "harness.lifecycle",
    "container.lifecycle",
    "evaluator.lifecycle",
    "patch",
    "native.evidence",
}
DISALLOWED_STRUCTURED_FIELDS = {
    "api_key",
    "authorization",
    "cached_tokens",
    "completion_tokens",
    "cookie",
    "cost",
    "estimated_cost",
    "input_tokens",
    "output_tokens",
    "password",
    "prompt_tokens",
    "token_count",
    "total_cost",
    "total_tokens",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    assert path.read_bytes().endswith(b"\n")
    return [json.loads(line) for line in path.read_text().splitlines()]


def build_registry() -> Registry:
    resources = [
        (schema["$id"], Resource.from_contents(schema))
        for schema in map(load_json, SCHEMA_FILES.values())
    ]
    return Registry().with_resources(resources)


def validator(schema_name: str) -> Draft202012Validator:
    return Draft202012Validator(
        load_json(SCHEMA_FILES[schema_name]),
        registry=build_registry(),
        format_checker=FormatChecker(),
    )


def schema_digest() -> str:
    digest = hashlib.sha256()
    for path in SCHEMA_FILES.values():
        canonical = json.dumps(
            load_json(path),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        digest.update(path.name.encode())
        digest.update(b"\n")
        digest.update(canonical)
        digest.update(b"\n")
    return digest.hexdigest()


def artifact_refs(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if {
            "sha256",
            "path",
            "size_bytes",
            "media_type",
            "encoding",
            "role",
            "redaction",
        } <= value.keys():
            yield value
        for item in value.values():
            yield from artifact_refs(item)
    if isinstance(value, list):
        for item in value:
            yield from artifact_refs(item)


def structured_field_names(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key.lower()
            yield from structured_field_names(item)
    if isinstance(value, list):
        for item in value:
            yield from structured_field_names(item)


def test_contract_constants_match_schema() -> None:
    definitions = load_json(SCHEMA_FILES["definitions.schema.json"])

    assert (
        CONTRACT_NAME
        == definitions["$defs"]["contract_header"]["properties"]["name"]["const"]
    )
    assert (
        CONTRACT_VERSION
        == definitions["$defs"]["contract_header"]["properties"]["version"]["const"]
    )
    assert SCHEMA_VERSION == definitions["$defs"]["schema_version"]["const"]


def test_schemas_are_valid_draft_2020_12() -> None:
    for path in SCHEMA_FILES.values():
        Draft202012Validator.check_schema(load_json(path))


@pytest.mark.parametrize(
    ("schema_name", "fixture_name"),
    [
        ("run.schema.json", "run.json"),
        ("manifest.schema.json", "manifest.json"),
        ("capabilities.schema.json", "capabilities.json"),
        ("health.schema.json", "health.json"),
    ],
)
def test_valid_json_fixtures(schema_name: str, fixture_name: str) -> None:
    validator(schema_name).validate(load_json(VALID_ROOT / fixture_name))


@pytest.mark.parametrize("fixture_name", ["events.jsonl", "journal.jsonl"])
def test_valid_event_streams(fixture_name: str) -> None:
    records = load_jsonl(VALID_ROOT / fixture_name)
    event_validator = validator("event.schema.json")

    for record in records:
        event_validator.validate(record)


def test_valid_native_index() -> None:
    native_validator = validator("native-index.schema.json")

    for record in load_jsonl(VALID_ROOT / "native" / "index.jsonl"):
        native_validator.validate(record)


def test_invalid_conformance_fixtures_are_rejected() -> None:
    cases = load_json(FIXTURE_ROOT / "invalid" / "index.json")

    for case in cases:
        errors = list(
            validator(case["schema"]).iter_errors(
                load_json(FIXTURE_ROOT / "invalid" / case["fixture"])
            )
        )
        assert errors, case["name"]


def test_fixture_schema_digests_match_bundle() -> None:
    expected = schema_digest()
    documents = [
        load_json(VALID_ROOT / "run.json"),
        load_json(VALID_ROOT / "manifest.json"),
        load_json(VALID_ROOT / "capabilities.json"),
        load_json(VALID_ROOT / "health.json"),
        *load_jsonl(VALID_ROOT / "events.jsonl"),
        *load_jsonl(VALID_ROOT / "journal.jsonl"),
        *load_jsonl(VALID_ROOT / "native" / "index.jsonl"),
    ]

    for document in documents:
        assert (
            document.get(
                "schema_digest", document.get("contract", {}).get("schema_digest")
            )
            == expected
        )


def test_event_and_native_sequences_are_contiguous() -> None:
    events = load_jsonl(VALID_ROOT / "events.jsonl")
    native_records = load_jsonl(VALID_ROOT / "native" / "index.jsonl")

    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert [record["sequence"] for record in native_records] == list(
        range(1, len(native_records) + 1)
    )
    assert len({event["event_id"] for event in events}) == len(events)


def test_final_event_stream_matches_journal() -> None:
    assert (VALID_ROOT / "events.jsonl").read_bytes() == (
        VALID_ROOT / "journal.jsonl"
    ).read_bytes()


def test_capability_matrix_is_unique_and_exhaustive() -> None:
    report = load_json(VALID_ROOT / "capabilities.json")
    categories = [capability["category"] for capability in report["capabilities"]]

    assert len(categories) == len(set(categories))
    assert set(categories) == CAPABILITY_CATEGORIES


def test_capability_evidence_resolves_to_observed_event_types() -> None:
    report = load_json(VALID_ROOT / "capabilities.json")
    event_types = {
        event["event_type"] for event in load_jsonl(VALID_ROOT / "events.jsonl")
    }

    for capability in report["capabilities"]:
        assert set(capability["evidence"]) <= event_types


def test_artifact_references_match_retained_bytes() -> None:
    documents = [
        *load_jsonl(VALID_ROOT / "events.jsonl"),
        *load_jsonl(VALID_ROOT / "native" / "index.jsonl"),
    ]
    references = {
        reference["path"]: reference
        for document in documents
        for reference in artifact_refs(document)
    }

    assert references
    for relative_path, reference in references.items():
        artifact = VALID_ROOT / relative_path
        content = artifact.read_bytes()

        assert len(content) == reference["size_bytes"]
        assert hashlib.sha256(content).hexdigest() == reference["sha256"]

    artifacts = {
        path.relative_to(VALID_ROOT).as_posix()
        for path in (VALID_ROOT / "artifacts").rglob("*")
        if path.is_file()
    }
    assert artifacts == references.keys()


def test_health_counters_match_finalized_trace() -> None:
    health = load_json(VALID_ROOT / "health.json")
    events = load_jsonl(VALID_ROOT / "events.jsonl")
    native_records = load_jsonl(VALID_ROOT / "native" / "index.jsonl")
    references = {
        reference["path"]: reference
        for document in [*events, *native_records]
        for reference in artifact_refs(document)
    }

    assert health["counters"]["events_written"] == len(events)
    assert health["counters"]["artifacts_written"] == len(references)
    assert health["counters"]["artifact_bytes_written"] == sum(
        reference["size_bytes"] for reference in references.values()
    )


def test_completed_spans_have_matching_start_boundaries() -> None:
    events = load_jsonl(VALID_ROOT / "events.jsonl")
    starts = {event["span_id"]: event for event in events if event["phase"] == "start"}
    ends = {event["span_id"]: event for event in events if event["phase"] == "end"}

    assert starts.keys() == ends.keys()
    for span_id, end in ends.items():
        assert starts[span_id]["sequence"] < end["sequence"]


def test_cross_file_identity_is_consistent() -> None:
    run = load_json(VALID_ROOT / "run.json")
    manifest = load_json(VALID_ROOT / "manifest.json")
    capabilities = load_json(VALID_ROOT / "capabilities.json")
    health = load_json(VALID_ROOT / "health.json")
    events = load_jsonl(VALID_ROOT / "events.jsonl")
    native_records = load_jsonl(VALID_ROOT / "native" / "index.jsonl")
    attempt = run["attempts"][0]

    assert attempt["trace_id"] == manifest["trace_id"]
    assert attempt["instance_id"] == manifest["instance_id"]
    assert attempt["attempt"] == manifest["attempt"]
    assert run["run_id"] == manifest["run_id"]
    assert run["benchmark"] == manifest["benchmark"]
    assert run["framework"] == manifest["framework"]
    assert capabilities["trace_id"] == manifest["trace_id"]
    assert capabilities["framework"] == manifest["framework"]
    assert health["trace_id"] == manifest["trace_id"]
    for event in events:
        assert event["trace_id"] == manifest["trace_id"]
        assert event["run_id"] == manifest["run_id"]
        assert event["framework"] == manifest["framework"]
        assert event["instance_id"] == manifest["instance_id"]
        assert event["attempt"] == manifest["attempt"]
    for record in native_records:
        assert record["trace_id"] == manifest["trace_id"]
        assert record["framework"] == manifest["framework"]


def test_native_event_references_resolve() -> None:
    event_ids = {event["event_id"] for event in load_jsonl(VALID_ROOT / "events.jsonl")}

    for record in load_jsonl(VALID_ROOT / "native" / "index.jsonl"):
        assert set(record["event_ids"]) <= event_ids


def test_normalized_fixtures_exclude_sensitive_accounting_fields() -> None:
    documents = [
        load_json(VALID_ROOT / "run.json"),
        load_json(VALID_ROOT / "manifest.json"),
        load_json(VALID_ROOT / "capabilities.json"),
        load_json(VALID_ROOT / "health.json"),
        *load_jsonl(VALID_ROOT / "events.jsonl"),
        *load_jsonl(VALID_ROOT / "native" / "index.jsonl"),
    ]

    fields = {
        field for document in documents for field in structured_field_names(document)
    }
    assert not fields & DISALLOWED_STRUCTURED_FIELDS
