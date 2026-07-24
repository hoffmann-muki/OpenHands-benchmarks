"""Schema and cross-file validation for finalized benchmark traces."""

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from benchmarks.tracing.errors import TraceStorageError, TraceValidationError
from benchmarks.tracing.models import JsonObject, JsonValue
from benchmarks.tracing.redaction import Redactor
from benchmarks.tracing.storage import (
    DIRECTORY_MODE,
    FILE_MODE,
    JournalRead,
    read_jsonl,
)


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
_ARTIFACT_FIELDS = {
    "sha256",
    "path",
    "size_bytes",
    "media_type",
    "encoding",
    "role",
    "redaction",
}


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: Literal["warning", "error"]
    code: str
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def require_valid(self) -> None:
        if self.valid:
            return
        codes = ", ".join(
            sorted({issue.code for issue in self.issues if issue.severity == "error"})
        )
        raise TraceValidationError(f"Trace validation failed: {codes}")


class ContractValidator:
    """Validate individual contract documents and complete attempt directories."""

    def __init__(self, redactor: Redactor | None = None) -> None:
        schema_root = files("benchmarks.tracing").joinpath("spec", "v1")
        schema_resources = sorted(
            (
                resource
                for resource in schema_root.iterdir()
                if resource.name.endswith(".schema.json")
            ),
            key=lambda resource: resource.name,
        )
        self._schemas: dict[str, Any] = {
            resource.name: json.loads(resource.read_text())
            for resource in schema_resources
        }
        resources = [
            (schema["$id"], Resource.from_contents(schema))
            for schema in self._schemas.values()
        ]
        registry = Registry().with_resources(resources)
        self._validators = {
            name: Draft202012Validator(
                schema,
                registry=registry,
                format_checker=FormatChecker(),
            )
            for name, schema in self._schemas.items()
        }
        self._redactor = redactor or Redactor()
        self.schema_digest = self._calculate_schema_digest()

    def validate_document(
        self, schema_name: str, document: JsonValue, *, path: str = "$"
    ) -> tuple[ValidationIssue, ...]:
        validator = self._validators.get(schema_name)
        if validator is None:
            raise ValueError(f"Unknown trace schema: {schema_name}")
        issues: list[ValidationIssue] = []
        for error in sorted(
            validator.iter_errors(document),
            key=lambda item: tuple(str(part) for part in item.path),
        ):
            location = ".".join(str(part) for part in error.path)
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="schema.invalid",
                    path=f"{path}.{location}" if location else path,
                    message=f"Document does not satisfy {schema_name}",
                )
            )
        return tuple(issues)

    def require_document(
        self, schema_name: str, document: JsonValue, *, path: str = "$"
    ) -> None:
        report = ValidationReport(
            issues=self.validate_document(schema_name, document, path=path)
        )
        report.require_valid()

    def validate_capability_semantics(
        self, document: JsonObject
    ) -> tuple[ValidationIssue, ...]:
        capabilities = document.get("capabilities")
        if not isinstance(capabilities, list):
            return ()
        categories = [
            item.get("category")
            for item in capabilities
            if isinstance(item, dict) and isinstance(item.get("category"), str)
        ]
        issues: list[ValidationIssue] = []
        if len(categories) != len(set(categories)):
            issues.append(
                _issue(
                    "capabilities.duplicate",
                    "$.capabilities",
                    "Capability categories must be unique",
                )
            )
        if set(categories) != CAPABILITY_CATEGORIES:
            issues.append(
                _issue(
                    "capabilities.incomplete",
                    "$.capabilities",
                    "Capability matrix must contain every v1 category",
                )
            )
        return tuple(issues)

    def validate_attempt(self, attempt_dir: Path) -> ValidationReport:
        issues: list[ValidationIssue] = []
        manifest = _load_json(attempt_dir / "manifest.json", issues)
        capabilities = _load_json(attempt_dir / "capabilities.json", issues)
        health = _load_json(attempt_dir / "health.json", issues)
        events = _load_jsonl(attempt_dir / "events.jsonl", issues)
        journal = _load_jsonl(
            attempt_dir / "journal.jsonl",
            issues,
            allow_torn_final_line=True,
        )
        native = _load_jsonl(attempt_dir / "native" / "index.jsonl", issues)
        self._validate_permissions(attempt_dir, issues)
        if any(value is None for value in (manifest, capabilities, health)):
            return ValidationReport(tuple(issues))
        if any(value is None for value in (events, journal, native)):
            return ValidationReport(tuple(issues))
        assert manifest is not None
        assert capabilities is not None
        assert health is not None
        assert events is not None
        assert journal is not None
        assert native is not None

        schema_issues = [
            *self.validate_document(
                "manifest.schema.json", manifest, path="manifest.json"
            ),
            *self.validate_document(
                "capabilities.schema.json",
                capabilities,
                path="capabilities.json",
            ),
            *self.validate_document("health.schema.json", health, path="health.json"),
        ]
        for index, event in enumerate(events.records, start=1):
            schema_issues.extend(
                self.validate_document(
                    "event.schema.json",
                    event,
                    path=f"events.jsonl:{index}",
                )
            )
        for index, event in enumerate(journal.records, start=1):
            schema_issues.extend(
                self.validate_document(
                    "event.schema.json",
                    event,
                    path=f"journal.jsonl:{index}",
                )
            )
        for index, record in enumerate(native.records, start=1):
            schema_issues.extend(
                self.validate_document(
                    "native-index.schema.json",
                    record,
                    path=f"native/index.jsonl:{index}",
                )
            )
        issues.extend(schema_issues)
        if schema_issues:
            return ValidationReport(tuple(issues))

        issues.extend(self.validate_capability_semantics(capabilities))
        self._validate_schema_digests(
            manifest, capabilities, health, events.records, native.records, issues
        )
        self._validate_sequences(events.records, native.records, issues)
        self._validate_identity(
            manifest, capabilities, health, events.records, native.records, issues
        )
        self._validate_journal(events, journal, health, issues)
        self._validate_capability_evidence(capabilities, events.records, issues)
        self._validate_spans(manifest, events.records, issues)
        self._validate_event_relations(events.records, issues)
        self._validate_native_links(events.records, native.records, issues)
        references = self._validate_artifacts(
            attempt_dir, events.records, native.records, issues
        )
        self._validate_health(
            manifest, health, events.records, references, journal, issues
        )
        self._validate_sensitive_content(
            attempt_dir,
            (manifest, capabilities, health, *events.records, *native.records),
            references,
            issues,
        )
        return ValidationReport(tuple(issues))

    def validate_run(self, run_root: Path) -> ValidationReport:
        issues: list[ValidationIssue] = []
        document = _load_json(run_root / "run.json", issues)
        self._validate_run_permissions(run_root, issues)
        if document is None:
            return ValidationReport(tuple(issues))
        schema_issues = self.validate_document(
            "run.schema.json",
            document,
            path="run.json",
        )
        issues.extend(schema_issues)
        if schema_issues:
            return ValidationReport(tuple(issues))

        contract = document["contract"]
        assert isinstance(contract, dict)
        if contract["schema_digest"] != self.schema_digest:
            issues.append(
                _issue(
                    "schema.digest_mismatch",
                    "run.json",
                    "Run schema digest does not match the installed contract",
                )
            )
        if self._redactor.detect_json(document):
            issues.append(
                _issue(
                    "redaction.structured_content",
                    "run.json",
                    "Run index contains credentials or accounting fields",
                )
            )

        selection = document["selection"]
        attempts = document["attempts"]
        assert isinstance(selection, dict)
        assert isinstance(attempts, list)
        instance_ids = selection["instance_ids"]
        assert isinstance(instance_ids, list)
        requested_count = selection.get("requested_count")
        if requested_count is not None and requested_count != len(instance_ids):
            issues.append(
                _issue(
                    "run.selection_count",
                    "run.json.selection",
                    "Requested count does not match selected instance identifiers",
                )
            )

        trace_ids: set[str] = set()
        attempt_keys: set[tuple[str, int]] = set()
        attempted_instances: set[str] = set()
        for index, attempt in enumerate(attempts):
            assert isinstance(attempt, dict)
            trace_id = str(attempt["trace_id"])
            instance_id = str(attempt["instance_id"])
            attempt_number = attempt["attempt"]
            assert isinstance(attempt_number, int)
            key = (instance_id, attempt_number)
            if trace_id in trace_ids or key in attempt_keys:
                issues.append(
                    _issue(
                        "run.duplicate_attempt",
                        f"run.json.attempts.{index}",
                        "Run attempts must have unique trace and attempt identity",
                    )
                )
            trace_ids.add(trace_id)
            attempt_keys.add(key)
            attempted_instances.add(instance_id)

            expected_path = (
                "instances/"
                f"{quote(instance_id, safe='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._~-')}/"
                f"attempt-{attempt_number}"
            )
            relative_path = str(attempt["path"])
            if relative_path != expected_path:
                issues.append(
                    _issue(
                        "run.attempt_path",
                        f"run.json.attempts.{index}.path",
                        "Attempt path does not match its canonical instance identity",
                    )
                )
                continue
            attempt_dir = run_root / relative_path
            report = self.validate_attempt(attempt_dir)
            issues.extend(
                ValidationIssue(
                    severity=issue.severity,
                    code=issue.code,
                    path=f"{relative_path}/{issue.path}",
                    message=issue.message,
                )
                for issue in report.issues
            )
            manifest = _load_json(attempt_dir / "manifest.json", issues)
            events = _load_jsonl(attempt_dir / "events.jsonl", issues)
            if manifest is None or events is None:
                continue
            expected_identity = {
                "trace_id": trace_id,
                "run_id": document["run_id"],
                "benchmark": document["benchmark"],
                "framework": document["framework"],
                "instance_id": instance_id,
                "attempt": attempt_number,
            }
            if any(manifest[key] != value for key, value in expected_identity.items()):
                issues.append(
                    _issue(
                        "run.identity_mismatch",
                        f"run.json.attempts.{index}",
                        "Attempt manifest identity does not match the run index",
                    )
                )
            terminal_status = next(
                (
                    event["status"]
                    for event in reversed(events.records)
                    if event["event_type"] == "instance.end"
                ),
                None,
            )
            if terminal_status is not None and terminal_status != attempt["status"]:
                issues.append(
                    _issue(
                        "run.status_mismatch",
                        f"run.json.attempts.{index}.status",
                        "Attempt status does not match its terminal instance event",
                    )
                )

        if attempted_instances != {str(value) for value in instance_ids}:
            issues.append(
                _issue(
                    "run.selection_mismatch",
                    "run.json",
                    "Finalized attempts do not match the selected instance set",
                )
            )
        return ValidationReport(tuple(issues))

    def _calculate_schema_digest(self) -> str:
        digest = hashlib.sha256()
        for name, schema in sorted(self._schemas.items()):
            canonical = json.dumps(
                schema,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            digest.update(name.encode())
            digest.update(b"\n")
            digest.update(canonical)
            digest.update(b"\n")
        return digest.hexdigest()

    def _validate_schema_digests(
        self,
        manifest: JsonObject,
        capabilities: JsonObject,
        health: JsonObject,
        events: tuple[JsonObject, ...],
        native: tuple[JsonObject, ...],
        issues: list[ValidationIssue],
    ) -> None:
        contract = manifest["contract"]
        assert isinstance(contract, dict)
        values = [
            ("manifest.json", contract["schema_digest"]),
            ("capabilities.json", capabilities["schema_digest"]),
            ("health.json", health["schema_digest"]),
            *[
                (f"events.jsonl:{index}", event["schema_digest"])
                for index, event in enumerate(events, start=1)
            ],
            *[
                (f"native/index.jsonl:{index}", record["schema_digest"])
                for index, record in enumerate(native, start=1)
            ],
        ]
        for path, value in values:
            if value != self.schema_digest:
                issues.append(
                    _issue(
                        "schema.digest_mismatch",
                        path,
                        "Document schema digest does not match the installed contract",
                    )
                )

    def _validate_sequences(
        self,
        events: tuple[JsonObject, ...],
        native: tuple[JsonObject, ...],
        issues: list[ValidationIssue],
    ) -> None:
        if [event["sequence"] for event in events] != list(range(1, len(events) + 1)):
            issues.append(
                _issue(
                    "events.sequence",
                    "events.jsonl",
                    "Event sequence must be contiguous and start at one",
                )
            )
        if [record["sequence"] for record in native] != list(range(1, len(native) + 1)):
            issues.append(
                _issue(
                    "native.sequence",
                    "native/index.jsonl",
                    "Native record sequence must be contiguous and start at one",
                )
            )
        event_ids = [event["event_id"] for event in events]
        if len(event_ids) != len(set(event_ids)):
            issues.append(
                _issue(
                    "events.duplicate_id",
                    "events.jsonl",
                    "Event identifiers must be unique within a trace",
                )
            )
        native_ids = [str(record["native_record_id"]) for record in native]
        if len(native_ids) != len(set(native_ids)):
            issues.append(
                _issue(
                    "native.duplicate_id",
                    "native/index.jsonl",
                    "Native record identifiers must be unique within a trace",
                )
            )

    def _validate_identity(
        self,
        manifest: JsonObject,
        capabilities: JsonObject,
        health: JsonObject,
        events: tuple[JsonObject, ...],
        native: tuple[JsonObject, ...],
        issues: list[ValidationIssue],
    ) -> None:
        expected = {
            key: manifest[key]
            for key in (
                "trace_id",
                "run_id",
                "benchmark",
                "framework",
                "instance_id",
                "attempt",
            )
        }
        if capabilities["trace_id"] != expected["trace_id"]:
            issues.append(
                _issue(
                    "identity.mismatch",
                    "capabilities.json",
                    "Capability trace identity does not match the manifest",
                )
            )
        if capabilities["framework"] != expected["framework"]:
            issues.append(
                _issue(
                    "identity.mismatch",
                    "capabilities.json",
                    "Capability framework does not match the manifest",
                )
            )
        if health["trace_id"] != expected["trace_id"]:
            issues.append(
                _issue(
                    "identity.mismatch",
                    "health.json",
                    "Health trace identity does not match the manifest",
                )
            )
        for index, event in enumerate(events, start=1):
            if any(event[key] != value for key, value in expected.items()):
                issues.append(
                    _issue(
                        "identity.mismatch",
                        f"events.jsonl:{index}",
                        "Event identity does not match the manifest",
                    )
                )
        for index, record in enumerate(native, start=1):
            if (
                record["trace_id"] != expected["trace_id"]
                or record["framework"] != expected["framework"]
            ):
                issues.append(
                    _issue(
                        "identity.mismatch",
                        f"native/index.jsonl:{index}",
                        "Native record identity does not match the manifest",
                    )
                )

    def _validate_journal(
        self,
        events: JournalRead,
        journal: JournalRead,
        health: JsonObject,
        issues: list[ValidationIssue],
    ) -> None:
        if events.records != journal.records:
            issues.append(
                _issue(
                    "journal.mismatch",
                    "journal.jsonl",
                    "Complete journal records do not match finalized events",
                )
            )
        if journal.torn_final_line and health["finalization"] not in {
            "recovered",
            "partial",
        }:
            issues.append(
                _issue(
                    "journal.unreported_recovery",
                    "health.json",
                    "Torn journal recovery is not reported by trace health",
                )
            )

    def _validate_capability_evidence(
        self,
        capabilities: JsonObject,
        events: tuple[JsonObject, ...],
        issues: list[ValidationIssue],
    ) -> None:
        event_types = {str(event["event_type"]) for event in events}
        values = capabilities["capabilities"]
        assert isinstance(values, list)
        for index, capability in enumerate(values):
            assert isinstance(capability, dict)
            evidence = capability["evidence"]
            assert isinstance(evidence, list)
            if not {str(item) for item in evidence} <= event_types:
                issues.append(
                    _issue(
                        "capabilities.unresolved_evidence",
                        f"capabilities.json.capabilities.{index}",
                        "Capability evidence does not resolve to an observed event type",
                    )
                )

    def _validate_spans(
        self,
        manifest: JsonObject,
        events: tuple[JsonObject, ...],
        issues: list[ValidationIssue],
    ) -> None:
        spans = {str(event["span_id"]) for event in events}
        for index, event in enumerate(events, start=1):
            parent = event.get("parent_span_id")
            if parent is not None and str(parent) not in spans:
                issues.append(
                    _issue(
                        "spans.missing_parent",
                        f"events.jsonl:{index}",
                        "Parent span does not exist in the trace",
                    )
                )
            timing = event["timing"]
            assert isinstance(timing, dict)
            if (
                timing["fidelity"] == "native_monotonic"
                and "ended_monotonic_ns" in timing
            ):
                started = timing["started_monotonic_ns"]
                ended = timing["ended_monotonic_ns"]
                duration = timing["duration_ms"]
                assert isinstance(started, int)
                assert isinstance(ended, int)
                assert isinstance(duration, int | float)
                measured = (ended - started) / 1_000_000
                if abs(measured - duration) > 0.001:
                    issues.append(
                        _issue(
                            "timing.inconsistent",
                            f"events.jsonl:{index}.timing",
                            "Monotonic boundaries do not match reported duration",
                        )
                    )

        parents = {
            str(event["span_id"]): (
                str(event["parent_span_id"])
                if event.get("parent_span_id") is not None
                else None
            )
            for event in events
        }
        for span_id in parents:
            if _has_parent_cycle(span_id, parents):
                issues.append(
                    _issue(
                        "spans.parent_cycle",
                        f"events.jsonl#{span_id}",
                        "Span parent relationships must be acyclic",
                    )
                )
                break

        if manifest["complete"] is not True:
            return
        boundaries: dict[str, list[JsonObject]] = {}
        for event in events:
            phase = event["phase"]
            if phase == "instant":
                continue
            span_id = str(event["span_id"])
            boundaries.setdefault(span_id, []).append(event)
        for span_id, span_events in boundaries.items():
            phases = [str(event["phase"]) for event in span_events]
            if phases != ["start", "end"]:
                issues.append(
                    _issue(
                        "spans.unbalanced",
                        f"events.jsonl#{span_id}",
                        "Complete traces require one ordered start and end per span",
                    )
                )
                continue
            start, end = span_events
            if start["event_family"] != end["event_family"] or _activity_name(
                str(start["event_type"])
            ) != _activity_name(str(end["event_type"])):
                issues.append(
                    _issue(
                        "spans.boundary_mismatch",
                        f"events.jsonl#{span_id}",
                        "Span start and end boundaries describe different activities",
                    )
                )

    def _validate_event_relations(
        self,
        events: tuple[JsonObject, ...],
        issues: list[ValidationIssue],
    ) -> None:
        event_ids = {str(event["event_id"]) for event in events}
        for index, event in enumerate(events, start=1):
            relations = event.get("relations", [])
            assert isinstance(relations, list)
            for relation in relations:
                assert isinstance(relation, dict)
                if str(relation["event_id"]) not in event_ids:
                    issues.append(
                        _issue(
                            "events.unresolved_relation",
                            f"events.jsonl:{index}.relations",
                            "Event relation references an unknown event",
                        )
                    )

    def _validate_native_links(
        self,
        events: tuple[JsonObject, ...],
        native: tuple[JsonObject, ...],
        issues: list[ValidationIssue],
    ) -> None:
        event_ids = {str(event["event_id"]) for event in events}
        for index, record in enumerate(native, start=1):
            values = record["event_ids"]
            assert isinstance(values, list)
            if not {str(value) for value in values} <= event_ids:
                issues.append(
                    _issue(
                        "native.unresolved_event",
                        f"native/index.jsonl:{index}",
                        "Native evidence references an unknown event",
                    )
                )

    def _validate_artifacts(
        self,
        attempt_dir: Path,
        events: tuple[JsonObject, ...],
        native: tuple[JsonObject, ...],
        issues: list[ValidationIssue],
    ) -> dict[str, JsonObject]:
        references: dict[str, JsonObject] = {}
        for document in (*events, *native):
            for reference in _artifact_references(document):
                relative_path = str(reference["path"])
                existing = references.get(relative_path)
                if existing is not None and existing != reference:
                    issues.append(
                        _issue(
                            "artifact.conflicting_reference",
                            relative_path,
                            "One artifact path has conflicting metadata",
                        )
                    )
                references[relative_path] = reference
        for relative_path, reference in references.items():
            digest = str(reference["sha256"])
            if relative_path != f"artifacts/sha256/{digest[:2]}/{digest}":
                issues.append(
                    _issue(
                        "artifact.noncanonical_path",
                        relative_path,
                        "Artifact path does not match its digest",
                    )
                )
            path = attempt_dir / relative_path
            if path.is_symlink() or not path.is_file():
                issues.append(
                    _issue(
                        "artifact.missing",
                        relative_path,
                        "Referenced artifact is missing or unsafe",
                    )
                )
                continue
            try:
                content = path.read_bytes()
            except OSError:
                issues.append(
                    _issue(
                        "artifact.unreadable",
                        relative_path,
                        "Referenced artifact cannot be read",
                    )
                )
                continue
            if len(content) != reference["size_bytes"]:
                issues.append(
                    _issue(
                        "artifact.size_mismatch",
                        relative_path,
                        "Artifact byte length does not match its reference",
                    )
                )
            if hashlib.sha256(content).hexdigest() != reference["sha256"]:
                issues.append(
                    _issue(
                        "artifact.digest_mismatch",
                        relative_path,
                        "Artifact digest does not match its reference",
                    )
                )

        artifact_root = attempt_dir / "artifacts" / "sha256"
        stored = (
            {
                path.relative_to(attempt_dir).as_posix()
                for path in artifact_root.rglob("*")
                if path.is_file()
            }
            if artifact_root.is_dir()
            else set()
        )
        orphaned = stored - references.keys()
        if orphaned:
            issues.append(
                _issue(
                    "artifact.orphaned",
                    "artifacts/sha256",
                    "Trace contains unreferenced artifacts",
                )
            )
        return references

    def _validate_health(
        self,
        manifest: JsonObject,
        health: JsonObject,
        events: tuple[JsonObject, ...],
        references: dict[str, JsonObject],
        journal: JournalRead,
        issues: list[ValidationIssue],
    ) -> None:
        counters = health["counters"]
        assert isinstance(counters, dict)
        expected_bytes = 0
        for reference in references.values():
            size = reference["size_bytes"]
            assert isinstance(size, int)
            expected_bytes += size
        expected = {
            "events_written": len(events),
            "artifacts_written": len(references),
            "artifact_bytes_written": expected_bytes,
        }
        for key, value in expected.items():
            if counters[key] != value:
                issues.append(
                    _issue(
                        "health.counter_mismatch",
                        f"health.json.counters.{key}",
                        "Health counter does not match retained trace data",
                    )
                )
        referenced_redactions = 0
        for reference in references.values():
            redaction = reference["redaction"]
            assert isinstance(redaction, dict)
            matches = redaction["matches"]
            assert isinstance(matches, int)
            referenced_redactions += matches
        recorded_redactions = counters["redactions_applied"]
        assert isinstance(recorded_redactions, int)
        if recorded_redactions < referenced_redactions:
            issues.append(
                _issue(
                    "health.counter_mismatch",
                    "health.json.counters.redactions_applied",
                    "Health redaction count is lower than retained artifact metadata",
                )
            )
        dropped_events = counters["dropped_events"]
        assert isinstance(dropped_events, int)
        if journal.torn_final_line and dropped_events < 1:
            issues.append(
                _issue(
                    "health.counter_mismatch",
                    "health.json.counters.dropped_events",
                    "Torn event recovery must increment dropped events",
                )
            )
        health_issues = health["issues"]
        assert isinstance(health_issues, list)
        if health["status"] == "healthy" and (
            health_issues
            or health["finalization"] != "clean"
            or counters["dropped_events"] != 0
            or counters["sequence_gaps"] != 0
        ):
            issues.append(
                _issue(
                    "health.inconsistent",
                    "health.json",
                    "Healthy trace status conflicts with reported degradation",
                )
            )
        if manifest["complete"] is True and health["status"] != "healthy":
            issues.append(
                _issue(
                    "manifest.inconsistent",
                    "manifest.json.complete",
                    "A complete manifest requires healthy trace finalization",
                )
            )
        if manifest["complete"] is False and health["status"] == "healthy":
            issues.append(
                _issue(
                    "manifest.inconsistent",
                    "manifest.json.complete",
                    "A healthy finalized trace must be marked complete",
                )
            )

    def _validate_sensitive_content(
        self,
        attempt_dir: Path,
        documents: tuple[JsonObject, ...],
        references: dict[str, JsonObject],
        issues: list[ValidationIssue],
    ) -> None:
        if any(self._redactor.detect_json(document) for document in documents):
            issues.append(
                _issue(
                    "redaction.structured_content",
                    "$",
                    "Trace documents contain credentials or accounting fields",
                )
            )
        for relative_path, reference in references.items():
            path = attempt_dir / relative_path
            if not path.is_file() or path.is_symlink():
                continue
            try:
                content = path.read_bytes()
            except OSError:
                continue
            detected = self._redactor.detect_bytes(content)
            if str(reference["media_type"]) == "application/json":
                try:
                    parsed = json.loads(content)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    parsed = None
                if parsed is not None:
                    detected = tuple(
                        dict.fromkeys((*detected, *self._redactor.detect_json(parsed)))
                    )
            if detected:
                issues.append(
                    _issue(
                        "redaction.sensitive_artifact",
                        relative_path,
                        "Artifact contains credential or accounting material",
                    )
                )

    def _validate_permissions(
        self, attempt_dir: Path, issues: list[ValidationIssue]
    ) -> None:
        if os.name == "nt":
            return
        paths = [
            attempt_dir,
            attempt_dir / "native",
            attempt_dir / "artifacts",
            attempt_dir / "artifacts" / "sha256",
        ]
        paths.extend(
            path
            for root in (
                attempt_dir / "native",
                attempt_dir / "artifacts",
            )
            if root.exists()
            for path in root.rglob("*")
        )
        paths.extend(
            attempt_dir / name
            for name in (
                "manifest.json",
                "journal.jsonl",
                "events.jsonl",
                "capabilities.json",
                "health.json",
            )
        )
        for path in paths:
            if not path.exists() and not path.is_symlink():
                continue
            if path.is_symlink():
                issues.append(
                    _issue(
                        "storage.symlink",
                        path.relative_to(attempt_dir).as_posix(),
                        "Trace storage must not contain symbolic links",
                    )
                )
                continue
            expected = DIRECTORY_MODE if path.is_dir() else FILE_MODE
            try:
                actual = path.stat().st_mode & 0o777
            except OSError:
                continue
            if actual != expected:
                issues.append(
                    _issue(
                        "storage.permissions",
                        path.relative_to(attempt_dir).as_posix() or ".",
                        f"Trace path must use mode {expected:04o}",
                    )
                )

    def _validate_run_permissions(
        self, run_root: Path, issues: list[ValidationIssue]
    ) -> None:
        if os.name == "nt":
            return
        for path, expected in (
            (run_root, DIRECTORY_MODE),
            (run_root / "run.json", FILE_MODE),
        ):
            if path.is_symlink():
                issues.append(
                    _issue(
                        "storage.symlink",
                        path.name,
                        "Trace storage must not contain symbolic links",
                    )
                )
                continue
            if not path.exists():
                continue
            try:
                actual = path.stat().st_mode & 0o777
            except OSError:
                continue
            if actual != expected:
                issues.append(
                    _issue(
                        "storage.permissions",
                        path.name,
                        f"Trace path must use mode {expected:04o}",
                    )
                )


def _load_json(path: Path, issues: list[ValidationIssue]) -> JsonObject | None:
    if path.is_symlink() or not path.is_file():
        issues.append(
            _issue("file.missing", path.name, "Required trace file is missing")
        )
        return None
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        issues.append(
            _issue("file.invalid_json", path.name, "Trace file is invalid JSON")
        )
        return None
    if not isinstance(value, dict):
        issues.append(
            _issue("file.invalid_type", path.name, "Trace file is not an object")
        )
        return None
    return value


def _load_jsonl(
    path: Path,
    issues: list[ValidationIssue],
    *,
    allow_torn_final_line: bool = False,
) -> JournalRead | None:
    if path.is_symlink() or not path.is_file():
        issues.append(
            _issue(
                "file.missing",
                path.as_posix(),
                "Required trace JSONL file is missing",
            )
        )
        return None
    try:
        return read_jsonl(path, allow_torn_final_line=allow_torn_final_line)
    except TraceStorageError:
        issues.append(
            _issue(
                "file.invalid_jsonl",
                path.as_posix(),
                "Trace JSONL file is malformed",
            )
        )
        return None


def _artifact_references(value: JsonValue) -> Iterator[JsonObject]:
    if isinstance(value, dict):
        if _ARTIFACT_FIELDS <= value.keys():
            yield value
        for item in value.values():
            yield from _artifact_references(item)
    if isinstance(value, list):
        for item in value:
            yield from _artifact_references(item)


def _activity_name(event_type: str) -> str:
    for suffix in (".start", ".end", "_start", "_end"):
        if event_type.endswith(suffix):
            return event_type[: -len(suffix)]
    return event_type


def _has_parent_cycle(
    span_id: str,
    parents: dict[str, str | None],
) -> bool:
    visited: set[str] = set()
    current: str | None = span_id
    while current is not None:
        if current in visited:
            return True
        visited.add(current)
        current = parents.get(current)
    return False


def _issue(code: str, path: str, message: str) -> ValidationIssue:
    return ValidationIssue(
        severity="error",
        code=code,
        path=path,
        message=message,
    )
