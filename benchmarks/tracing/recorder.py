"""Framework-neutral Python reference recorder for benchmark traces."""

import json
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import quote
from uuid import uuid4

from benchmarks.tracing.artifacts import ArtifactStore, ArtifactWrite
from benchmarks.tracing.errors import (
    TraceFinalizationError,
    TraceInitializationError,
    TraceStorageError,
    TraceValidationError,
)
from benchmarks.tracing.models import (
    Capability,
    JsonObject,
    JsonValue,
    TraceConfig,
    TraceIdentity,
    TraceIssue,
)
from benchmarks.tracing.redaction import Redactor
from benchmarks.tracing.storage import (
    JsonlJournal,
    atomic_write,
    canonical_json_bytes,
    create_private_directory,
    ensure_private_directory,
    format_timestamp,
    read_jsonl,
    utc_now,
)
from benchmarks.tracing.validation import (
    ContractValidator,
    ValidationReport,
)


@dataclass(frozen=True, slots=True)
class FinalizationResult:
    manifest: JsonObject
    health: JsonObject
    validation: ValidationReport


def encode_instance_id(instance_id: str) -> str:
    if not instance_id:
        raise ValueError("Instance identifier must not be empty")
    return quote(
        instance_id,
        safe="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._~-",
    )


def attempt_directory(run_root: Path, instance_id: str, attempt: int) -> Path:
    if attempt < 1:
        raise ValueError("Trace attempt must be positive")
    return (
        run_root / "instances" / encode_instance_id(instance_id) / f"attempt-{attempt}"
    )


def write_run_index(
    run_root: Path,
    document: JsonObject,
    *,
    redactor: Redactor | None = None,
    validator: ContractValidator | None = None,
) -> Path:
    """Validate and atomically write a finalized run index."""

    active_redactor = redactor or Redactor()
    active_validator = validator or ContractValidator(active_redactor)
    result = active_redactor.sanitize_object(document)
    if result.matches:
        raise TraceValidationError(
            "Run index contains material prohibited by the retention policy"
        )
    active_validator.require_document("run.schema.json", result.value, path="run.json")
    contract = result.value.get("contract")
    if (
        not isinstance(contract, dict)
        or contract.get("schema_digest") != active_validator.schema_digest
    ):
        raise TraceValidationError(
            "Run index schema digest does not match the installed contract"
        )
    ensure_private_directory(run_root)
    path = run_root / "run.json"
    atomic_write(path, canonical_json_bytes(result.value))
    active_validator.validate_run(run_root).require_valid()
    return path


class TraceRecorder:
    """Durable recorder for one benchmark instance attempt.

    Construction is a preflight boundary and may raise. Once construction
    succeeds, runtime recording methods isolate failures and return ``None`` so
    tracing cannot interrupt or retry an agent attempt.
    """

    def __init__(
        self,
        config: TraceConfig,
        *,
        redactor: Redactor | None = None,
        validator: ContractValidator | None = None,
    ) -> None:
        self._set_initial_state(config, redactor, validator)
        try:
            self._prepare_configuration()
            self._validate_preflight()
            self._initialize_storage()
        except (
            OSError,
            TypeError,
            ValueError,
            TraceStorageError,
            TraceValidationError,
        ) as exc:
            self._close_preflight_journals()
            raise TraceInitializationError(
                "Benchmark tracing could not be initialized safely"
            ) from exc

    @classmethod
    def recover(
        cls,
        config: TraceConfig,
        *,
        redactor: Redactor | None = None,
        validator: ContractValidator | None = None,
    ) -> "TraceRecorder":
        """Rehydrate a stopped recorder from durable journals without appending."""

        recorder = cls.__new__(cls)
        recorder._set_initial_state(config, redactor, validator)
        try:
            recorder._prepare_configuration()
            recorder._validate_preflight()
            if (
                not recorder.attempt_dir.is_dir()
                or recorder.attempt_dir.is_symlink()
                or not (recorder.attempt_dir / "journal.jsonl").is_file()
                or (recorder.attempt_dir / "journal.jsonl").is_symlink()
                or not (recorder.attempt_dir / "native" / "index.jsonl").is_file()
                or (recorder.attempt_dir / "native" / "index.jsonl").is_symlink()
            ):
                raise TraceStorageError("Trace recovery journals are unavailable")
            journal = read_jsonl(
                recorder.attempt_dir / "journal.jsonl",
                allow_torn_final_line=True,
            )
            first_recorded_at = (
                journal.records[0].get("recorded_at") if journal.records else None
            )
            recorder._created_at = (
                first_recorded_at
                if isinstance(first_recorded_at, str)
                else format_timestamp(
                    datetime.fromtimestamp(
                        (recorder.attempt_dir / "journal.jsonl").stat().st_mtime,
                        UTC,
                    )
                )
            )
            recorder._closed = True
            recorder._recovery_mode = True
            recorder._add_issue(
                "warning",
                "trace.process_recovery",
                "Trace finalization resumed from durable journals",
            )
        except (
            OSError,
            TypeError,
            ValueError,
            TraceStorageError,
            TraceValidationError,
        ) as exc:
            raise TraceInitializationError(
                "Benchmark trace recovery could not be initialized safely"
            ) from exc
        return recorder

    def _set_initial_state(
        self,
        config: TraceConfig,
        redactor: Redactor | None,
        validator: ContractValidator | None,
    ) -> None:
        self._lock = threading.RLock()
        self._redactor = redactor or Redactor()
        self._validator = validator or ContractValidator(self._redactor)
        self._config = config
        self._created_at = utc_now()
        self._event_sequence = 0
        self._native_sequence = 0
        self._event_ids: set[str] = set()
        self._native_ids: set[str] = set()
        self._artifacts: dict[str, JsonObject] = {}
        self._redactions_applied = 0
        self._issues: dict[str, TraceIssue] = {}
        self._closed = False
        self._recovery_mode = False
        self._finalized: FinalizationResult | None = None
        self._event_journal: JsonlJournal | None = None
        self._native_journal: JsonlJournal | None = None
        self._artifact_store: ArtifactStore | None = None

    @property
    def attempt_dir(self) -> Path:
        return self._config.attempt_dir

    @property
    def identity(self) -> TraceIdentity:
        return self._config.identity

    def report_issue(
        self,
        code: str,
        message: str,
        *,
        severity: Literal["warning", "error"] = "error",
    ) -> None:
        """Record a sanitized adapter issue without raising into agent execution."""

        with self._lock:
            safe_code = (
                code
                if re.fullmatch(r"[a-z][a-z0-9._-]*", code)
                else "adapter.invalid_issue_code"
            )
            result = self._redactor.sanitize_text(message)
            self._redactions_applied += result.matches
            self._add_issue(severity, safe_code, result.value)

    def update_capabilities(
        self,
        capabilities: tuple[Capability, ...],
    ) -> bool:
        """Replace the preflight matrix with observed attempt capabilities."""

        with self._lock:
            if self._finalized is not None:
                self._add_issue(
                    "error",
                    "capabilities.update_after_finalize",
                    "Capabilities were submitted after trace finalization",
                )
                return False
            results = [
                self._redactor.sanitize_object(capability.as_json())
                for capability in capabilities
            ]
            previous = self._capabilities
            self._capabilities = tuple(result.value for result in results)
            try:
                document = self._capability_document(utc_now())
                self._validator.require_document(
                    "capabilities.schema.json",
                    document,
                    path="capabilities.json",
                )
                ValidationReport(
                    self._validator.validate_capability_semantics(document)
                ).require_valid()
            except TraceValidationError:
                self._capabilities = previous
                self._add_issue(
                    "error",
                    "capabilities.update_failed",
                    "The adapter submitted an invalid capability matrix",
                )
                return False
            self._redactions_applied += sum(result.matches for result in results)
            return True

    def store_text_artifact(
        self,
        value: str,
        *,
        media_type: str,
        role: str,
    ) -> JsonObject | None:
        with self._lock:
            if self._finalized is not None:
                self._add_issue(
                    "error",
                    "trace.record_after_finalize",
                    "An artifact was submitted after trace finalization",
                )
                return None
            try:
                store = self._require_artifact_store()
                result = store.put_text(value, media_type=media_type, role=role)
            except (OSError, TypeError, ValueError, TraceStorageError):
                self._add_issue(
                    "error",
                    "artifact.write_failed",
                    "A trace artifact could not be retained",
                )
                return None
            self._register_artifact(result)
            return result.reference

    def store_bytes_artifact(
        self,
        value: bytes,
        *,
        media_type: str,
        encoding: Literal["utf-8", "binary"],
        role: str,
    ) -> JsonObject | None:
        with self._lock:
            if self._finalized is not None:
                self._add_issue(
                    "error",
                    "trace.record_after_finalize",
                    "An artifact was submitted after trace finalization",
                )
                return None
            try:
                store = self._require_artifact_store()
                result = store.put_bytes(
                    value,
                    media_type=media_type,
                    encoding=encoding,
                    role=role,
                )
            except (OSError, TypeError, ValueError, TraceStorageError):
                self._add_issue(
                    "error",
                    "artifact.write_failed",
                    "A trace artifact could not be retained",
                )
                return None
            self._register_artifact(result)
            return result.reference

    def store_json_artifact(
        self,
        value: JsonValue,
        *,
        role: str,
        media_type: str = "application/json",
    ) -> JsonObject | None:
        with self._lock:
            if self._finalized is not None:
                self._add_issue(
                    "error",
                    "trace.record_after_finalize",
                    "An artifact was submitted after trace finalization",
                )
                return None
            try:
                store = self._require_artifact_store()
                result = store.put_json(value, role=role, media_type=media_type)
            except (OSError, TypeError, ValueError, TraceStorageError):
                self._add_issue(
                    "error",
                    "artifact.write_failed",
                    "A trace artifact could not be retained",
                )
                return None
            self._register_artifact(result)
            return result.reference

    def record_event(
        self,
        *,
        event_type: str,
        event_family: str,
        phase: Literal["start", "end", "instant"],
        status: str,
        span_id: str,
        origin: JsonObject,
        timing: JsonObject,
        payload: JsonObject | None = None,
        artifacts: tuple[JsonObject, ...] = (),
        occurred_at: datetime | str | None = None,
        event_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        parent_agent_id: str | None = None,
        turn_id: str | None = None,
        parent_span_id: str | None = None,
        error: JsonObject | None = None,
        relations: tuple[JsonObject, ...] = (),
    ) -> str | None:
        with self._lock:
            if self._finalized is not None:
                self._add_issue(
                    "error",
                    "trace.record_after_finalize",
                    "An event was submitted after trace finalization",
                )
                return None
            candidate_id = event_id or f"event-{uuid4().hex}"
            identifiers: JsonObject = {
                "event_id": candidate_id,
                "span_id": span_id,
            }
            for key, value in (
                ("session_id", session_id),
                ("agent_id", agent_id),
                ("parent_agent_id", parent_agent_id),
                ("turn_id", turn_id),
                ("parent_span_id", parent_span_id),
            ):
                if value is not None:
                    identifiers[key] = value
            if self._redactor.detect_json(identifiers):
                self._add_issue(
                    "error",
                    "event.sensitive_identity",
                    "An event identity contained credential-like material",
                )
                return None
            if candidate_id in self._event_ids:
                self._add_issue(
                    "error",
                    "event.duplicate_id",
                    "A duplicate event identifier was rejected",
                )
                return None
            for reference in artifacts:
                if not self._owns_artifact(reference):
                    self._add_issue(
                        "error",
                        "event.unknown_artifact",
                        "An event referenced an artifact not owned by this recorder",
                    )
                    return None

            sanitized_origin = self._redactor.sanitize_object(origin)
            sanitized_timing = self._redactor.sanitize_object(timing)
            sanitized_payload = self._redactor.sanitize_object(payload or {})
            sanitized_error = (
                self._redactor.sanitize_object(error) if error is not None else None
            )
            sanitized_relations = [
                self._redactor.sanitize_object(relation) for relation in relations
            ]
            redactions = (
                sanitized_origin.matches
                + sanitized_timing.matches
                + sanitized_payload.matches
                + (sanitized_error.matches if sanitized_error else 0)
                + sum(result.matches for result in sanitized_relations)
            )
            event: JsonObject = {
                "schema_version": "benchmark-trace/v1",
                "schema_digest": self._validator.schema_digest,
                "event_id": candidate_id,
                "sequence": self._event_sequence + 1,
                **self.identity.event_fields(),
                **identifiers,
                "occurred_at": format_timestamp(occurred_at),
                "recorded_at": utc_now(),
                "event_type": event_type,
                "event_family": event_family,
                "phase": phase,
                "status": status,
                "origin": sanitized_origin.value,
                "timing": sanitized_timing.value,
                "payload": sanitized_payload.value,
                "artifacts": [dict(reference) for reference in artifacts],
            }
            if sanitized_error is not None:
                event["error"] = sanitized_error.value
            if sanitized_relations:
                event["relations"] = [result.value for result in sanitized_relations]

            if self._validator.validate_document("event.schema.json", event):
                self._add_issue(
                    "error",
                    "event.invalid",
                    "An invalid normalized event was rejected",
                )
                return None
            journal = self._event_journal
            if journal is None:
                self._add_issue(
                    "error",
                    "journal.unavailable",
                    "The event journal is unavailable",
                )
                return None
            try:
                journal.append(event)
            except (OSError, TraceStorageError):
                self._add_issue(
                    "error",
                    "journal.append_failed",
                    "An event could not be durably appended",
                )
                return None

            self._event_sequence += 1
            self._event_ids.add(candidate_id)
            self._redactions_applied += redactions
            return candidate_id

    def record_native(
        self,
        *,
        source: str,
        content: JsonValue | bytes,
        media_type: str,
        role: str = "native.event",
        encoding: Literal["utf-8", "binary"] = "utf-8",
        event_ids: tuple[str, ...] = (),
        recorded_at: datetime | str | None = None,
        native_record_id: str | None = None,
    ) -> str | None:
        with self._lock:
            if self._finalized is not None:
                self._add_issue(
                    "error",
                    "trace.record_after_finalize",
                    "Native evidence was submitted after trace finalization",
                )
                return None
            if any(self._redactor.detect_json(value) for value in event_ids):
                self._add_issue(
                    "error",
                    "native.sensitive_identity",
                    "A native record identity contained credential-like material",
                )
                return None

            sanitized_source = self._redactor.sanitize_text(source)
            candidate_id = native_record_id or f"native-{uuid4().hex}"
            if (
                not sanitized_source.value
                or self._redactor.detect_json(candidate_id)
                or candidate_id in self._native_ids
            ):
                self._add_issue(
                    "error",
                    "native.invalid_identity",
                    "An invalid or duplicate native record identity was rejected",
                )
                return None
            artifact = self._store_native_content(
                content,
                media_type=media_type,
                role=role,
                encoding=encoding,
            )
            if artifact is None:
                return None
            record: JsonObject = {
                "schema_version": "benchmark-trace/v1",
                "schema_digest": self._validator.schema_digest,
                "native_record_id": candidate_id,
                "sequence": self._native_sequence + 1,
                "trace_id": self.identity.trace_id,
                "framework": self.identity.framework,
                "recorded_at": format_timestamp(recorded_at),
                "source": sanitized_source.value,
                "artifact": artifact,
                "event_ids": list(event_ids),
            }
            if self._validator.validate_document("native-index.schema.json", record):
                self._add_issue(
                    "error",
                    "native.invalid",
                    "An invalid native evidence record was rejected",
                )
                return None
            journal = self._native_journal
            if journal is None:
                self._add_issue(
                    "error",
                    "native.journal_unavailable",
                    "The native evidence journal is unavailable",
                )
                return None
            try:
                journal.append(record)
            except (OSError, TraceStorageError):
                self._add_issue(
                    "error",
                    "native.append_failed",
                    "Native evidence could not be durably indexed",
                )
                return None

            self._native_sequence += 1
            self._native_ids.add(candidate_id)
            self._redactions_applied += sanitized_source.matches
            return candidate_id

    def close(self) -> None:
        """Close live journals without finalizing, as a crash-recovery boundary."""

        with self._lock:
            if self._closed:
                return
            for journal, code, message in (
                (
                    self._event_journal,
                    "journal.close_failed",
                    "The event journal could not be closed cleanly",
                ),
                (
                    self._native_journal,
                    "native.close_failed",
                    "The native evidence journal could not be closed cleanly",
                ),
            ):
                if journal is None:
                    continue
                try:
                    journal.close()
                except (OSError, TraceStorageError):
                    self._add_issue("error", code, message)
            self._closed = True

    def finalize(self) -> FinalizationResult:
        """Finalize journals and return validation without affecting agent outcome."""

        with self._lock:
            if self._finalized is not None:
                return self._finalized
            self.close()
            try:
                events = read_jsonl(
                    self.attempt_dir / "journal.jsonl",
                    allow_torn_final_line=True,
                )
                native = read_jsonl(
                    self.attempt_dir / "native" / "index.jsonl",
                    allow_torn_final_line=True,
                )
            except TraceStorageError as exc:
                raise TraceFinalizationError(
                    "Trace journals could not be recovered"
                ) from exc

            if events.torn_final_line:
                self._add_issue(
                    "warning",
                    "journal.torn_final_line",
                    "A torn final event journal line was discarded",
                )
            if native.torn_final_line:
                self._add_issue(
                    "warning",
                    "native.torn_final_line",
                    "A torn final native journal line was discarded",
                )

            try:
                if self._recovery_mode:
                    self._redactions_applied += _retained_redaction_count(
                        events.records,
                        native.records,
                    )
                for index, event in enumerate(events.records, start=1):
                    self._validator.require_document(
                        "event.schema.json",
                        event,
                        path=f"journal.jsonl:{index}",
                    )
                for index, record in enumerate(native.records, start=1):
                    self._validator.require_document(
                        "native-index.schema.json",
                        record,
                        path=f"native/index.jsonl:{index}",
                    )
                atomic_write(
                    self.attempt_dir / "events.jsonl",
                    _jsonl_bytes(events.records),
                )
                atomic_write(
                    self.attempt_dir / "native" / "index.jsonl",
                    _jsonl_bytes(native.records),
                )
                result = self._write_final_documents(
                    events.records,
                    native.records,
                    dropped_events=1 if events.torn_final_line else 0,
                    recovered=(
                        self._recovery_mode
                        or events.torn_final_line
                        or native.torn_final_line
                    ),
                )
            except (OSError, TraceStorageError, TraceValidationError) as exc:
                raise TraceFinalizationError(
                    "Trace finalization could not be completed"
                ) from exc
            self._finalized = result
            return result

    def _prepare_configuration(self) -> None:
        if self._redactor.detect_json(self.identity.event_fields()):
            raise TraceValidationError("Trace identity contains sensitive material")

        producer = self._redactor.sanitize_object(self._config.producer.as_json())
        provenance = self._redactor.sanitize_object(self._config.provenance)
        execution = self._redactor.sanitize_object(self._config.execution)
        capabilities = [
            self._redactor.sanitize_object(capability.as_json())
            for capability in self._config.capabilities
        ]
        self._producer = producer.value
        self._provenance = provenance.value
        self._execution = execution.value
        self._capabilities = tuple(result.value for result in capabilities)
        self._redactions_applied += (
            producer.matches
            + provenance.matches
            + execution.matches
            + sum(result.matches for result in capabilities)
        )

    def _validate_preflight(self) -> None:
        generated_at = self._created_at
        capabilities = self._capability_document(generated_at)
        self._validator.require_document(
            "capabilities.schema.json",
            capabilities,
            path="capabilities.json",
        )
        report = ValidationReport(
            self._validator.validate_capability_semantics(capabilities)
        )
        report.require_valid()
        self._validator.require_document(
            "manifest.schema.json",
            self._manifest_document(
                finalized_at=generated_at,
                complete=False,
            ),
            path="manifest.json",
        )

    def _initialize_storage(self) -> None:
        create_private_directory(self.attempt_dir)
        ensure_private_directory(self.attempt_dir / "native")
        self._artifact_store = ArtifactStore(self.attempt_dir, self._redactor)
        self._event_journal = JsonlJournal.create(self.attempt_dir / "journal.jsonl")
        self._native_journal = JsonlJournal.create(
            self.attempt_dir / "native" / "index.jsonl"
        )

    def _close_preflight_journals(self) -> None:
        for journal in (self._event_journal, self._native_journal):
            if journal is None:
                continue
            try:
                journal.close()
            except TraceStorageError:
                pass

    def _require_artifact_store(self) -> ArtifactStore:
        if self._artifact_store is None:
            raise TraceStorageError("Trace artifact store is unavailable")
        return self._artifact_store

    def _register_artifact(self, result: ArtifactWrite) -> None:
        relative_path = result.reference["path"]
        assert isinstance(relative_path, str)
        if relative_path in self._artifacts:
            return
        self._artifacts[relative_path] = result.reference
        redaction = result.reference["redaction"]
        assert isinstance(redaction, dict)
        matches = redaction["matches"]
        assert isinstance(matches, int)
        self._redactions_applied += matches

    def _owns_artifact(self, reference: JsonObject) -> bool:
        relative_path = reference.get("path")
        if not isinstance(relative_path, str):
            return False
        owned = self._artifacts.get(relative_path)
        if owned is None:
            return False
        return all(
            owned.get(key) == reference.get(key)
            for key in (
                "sha256",
                "path",
                "size_bytes",
                "media_type",
                "encoding",
                "redaction",
            )
        )

    def _store_native_content(
        self,
        content: JsonValue | bytes,
        *,
        media_type: str,
        role: str,
        encoding: Literal["utf-8", "binary"],
    ) -> JsonObject | None:
        if isinstance(content, bytes):
            if media_type == "application/json" and encoding == "utf-8":
                try:
                    parsed = json.loads(content)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    parsed = None
                if parsed is not None:
                    return self.store_json_artifact(
                        parsed,
                        role=role,
                        media_type=media_type,
                    )
            return self.store_bytes_artifact(
                content,
                media_type=media_type,
                encoding=encoding,
                role=role,
            )
        if isinstance(content, str) and media_type != "application/json":
            return self.store_text_artifact(
                content,
                media_type=media_type,
                role=role,
            )
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                return self.store_text_artifact(
                    content,
                    media_type=media_type,
                    role=role,
                )
            return self.store_json_artifact(
                parsed,
                role=role,
                media_type=media_type,
            )
        return self.store_json_artifact(
            content,
            role=role,
            media_type=media_type,
        )

    def _write_final_documents(
        self,
        events: tuple[JsonObject, ...],
        native: tuple[JsonObject, ...],
        *,
        dropped_events: int,
        recovered: bool,
    ) -> FinalizationResult:
        finalized_at = utc_now()
        references = {
            str(reference["path"]): reference
            for document in (*events, *native)
            for reference in _artifact_references(document)
        }
        artifact_bytes = 0
        for reference in references.values():
            size = reference["size_bytes"]
            assert isinstance(size, int)
            artifact_bytes += size
        sequence_gaps = sum(
            event["sequence"] != expected
            for expected, event in enumerate(events, start=1)
        )
        recovery_issues = {
            "journal.torn_final_line",
            "native.torn_final_line",
            "trace.process_recovery",
        }
        has_non_recovery_issue = any(
            code not in recovery_issues for code in self._issues
        )
        finalization = (
            "partial"
            if has_non_recovery_issue
            else "recovered"
            if recovered
            else "clean"
        )
        status = "degraded" if self._issues else "healthy"
        health = self._health_document(
            generated_at=finalized_at,
            status=status,
            finalization=finalization,
            events_written=len(events),
            artifacts_written=len(references),
            artifact_bytes_written=artifact_bytes,
            dropped_events=dropped_events,
            sequence_gaps=sequence_gaps,
        )
        manifest = self._manifest_document(
            finalized_at=finalized_at,
            complete=status == "healthy" and finalization == "clean",
        )
        capabilities = self._capability_document(finalized_at)
        self._write_documents(capabilities, health, manifest)
        validation = self._validator.validate_attempt(self.attempt_dir)
        if validation.valid:
            return FinalizationResult(
                manifest=manifest,
                health=health,
                validation=validation,
            )

        self._add_issue(
            "error",
            "trace.validation_failed",
            "Finalized trace validation reported an observability defect",
        )
        health = self._health_document(
            generated_at=finalized_at,
            status="degraded",
            finalization="partial",
            events_written=len(events),
            artifacts_written=len(references),
            artifact_bytes_written=artifact_bytes,
            dropped_events=dropped_events,
            sequence_gaps=sequence_gaps,
        )
        manifest = self._manifest_document(
            finalized_at=finalized_at,
            complete=False,
        )
        self._write_documents(capabilities, health, manifest)
        return FinalizationResult(
            manifest=manifest,
            health=health,
            validation=self._validator.validate_attempt(self.attempt_dir),
        )

    def _write_documents(
        self,
        capabilities: JsonObject,
        health: JsonObject,
        manifest: JsonObject,
    ) -> None:
        self._validator.require_document(
            "capabilities.schema.json",
            capabilities,
            path="capabilities.json",
        )
        self._validator.require_document(
            "health.schema.json",
            health,
            path="health.json",
        )
        self._validator.require_document(
            "manifest.schema.json",
            manifest,
            path="manifest.json",
        )
        atomic_write(
            self.attempt_dir / "capabilities.json",
            canonical_json_bytes(capabilities),
        )
        atomic_write(
            self.attempt_dir / "health.json",
            canonical_json_bytes(health),
        )
        atomic_write(
            self.attempt_dir / "manifest.json",
            canonical_json_bytes(manifest),
        )

    def _capability_document(self, generated_at: str) -> JsonObject:
        return {
            "schema_version": "benchmark-trace/v1",
            "schema_digest": self._validator.schema_digest,
            "trace_id": self.identity.trace_id,
            "framework": self.identity.framework,
            "generated_at": generated_at,
            "capabilities": [dict(value) for value in self._capabilities],
        }

    def _manifest_document(
        self,
        *,
        finalized_at: str,
        complete: bool,
    ) -> JsonObject:
        return {
            "schema_version": "benchmark-trace/v1",
            "contract": {
                "name": "benchmark-trace",
                "version": "1.0.0",
                "schema_digest": self._validator.schema_digest,
            },
            **self.identity.event_fields(),
            "created_at": self._created_at,
            "finalized_at": finalized_at,
            "complete": complete,
            "producer": dict(self._producer),
            "provenance": dict(self._provenance),
            "execution": dict(self._execution),
            "files": {
                "journal": "journal.jsonl",
                "events": "events.jsonl",
                "capabilities": "capabilities.json",
                "health": "health.json",
                "native_index": "native/index.jsonl",
                "artifacts": "artifacts/sha256",
            },
        }

    def _health_document(
        self,
        *,
        generated_at: str,
        status: Literal["healthy", "degraded", "failed"],
        finalization: Literal["clean", "recovered", "partial", "unfinalized"],
        events_written: int,
        artifacts_written: int,
        artifact_bytes_written: int,
        dropped_events: int,
        sequence_gaps: int,
    ) -> JsonObject:
        return {
            "schema_version": "benchmark-trace/v1",
            "schema_digest": self._validator.schema_digest,
            "trace_id": self.identity.trace_id,
            "generated_at": generated_at,
            "status": status,
            "finalization": finalization,
            "failure_policy": "continue_agent_without_retry",
            "agent_outcome_affected": False,
            "benchmark_retry_triggered": False,
            "counters": {
                "events_written": events_written,
                "artifacts_written": artifacts_written,
                "artifact_bytes_written": artifact_bytes_written,
                "redactions_applied": self._redactions_applied,
                "dropped_events": dropped_events,
                "sequence_gaps": sequence_gaps,
            },
            "issues": [issue.as_json() for issue in self._issues.values()],
        }

    def _add_issue(
        self,
        severity: Literal["warning", "error"],
        code: str,
        message: str,
    ) -> None:
        with self._lock:
            timestamp = utc_now()
            existing = self._issues.get(code)
            if existing is None:
                self._issues[code] = TraceIssue(
                    severity=severity,
                    code=code,
                    message=message,
                    first_seen_at=timestamp,
                    last_seen_at=timestamp,
                )
                return
            self._issues[code] = TraceIssue(
                severity=(
                    "error" if "error" in {existing.severity, severity} else "warning"
                ),
                code=code,
                message=message,
                first_seen_at=existing.first_seen_at,
                last_seen_at=timestamp,
                count=existing.count + 1,
            )


def _jsonl_bytes(records: tuple[JsonObject, ...]) -> bytes:
    return b"".join(canonical_json_bytes(record) for record in records)


def _artifact_references(value: JsonValue) -> Iterator[JsonObject]:
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
            yield from _artifact_references(item)
    if isinstance(value, list):
        for item in value:
            yield from _artifact_references(item)


def _retained_redaction_count(
    events: tuple[JsonObject, ...],
    native: tuple[JsonObject, ...],
) -> int:
    references = {
        str(reference["path"]): reference
        for document in (*events, *native)
        for reference in _artifact_references(document)
    }
    artifact_matches = 0
    for reference in references.values():
        redaction = reference["redaction"]
        assert isinstance(redaction, dict)
        matches = redaction["matches"]
        assert isinstance(matches, int)
        artifact_matches += matches
    return artifact_matches + sum(
        _redaction_markers(document) for document in (*events, *native)
    )


def _redaction_markers(value: JsonValue) -> int:
    if isinstance(value, str):
        return value.count("<redacted:")
    if isinstance(value, dict):
        return sum(_redaction_markers(item) for item in value.values())
    if isinstance(value, list):
        return sum(_redaction_markers(item) for item in value)
    return 0
