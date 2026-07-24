"""Lossless, chunked storage for framework-native trace evidence."""

import base64
import binascii
import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from benchmarks.tracing.artifacts import ArtifactStore
from benchmarks.tracing.errors import TraceStorageError
from benchmarks.tracing.models import JsonObject, JsonValue
from benchmarks.tracing.redaction import Redactor
from benchmarks.tracing.storage import canonical_json_bytes


NATIVE_JOURNAL_FORMAT = "benchmark-trace/native-journal-v1"
NATIVE_CHUNK_MEDIA_TYPE = "application/vnd.benchmark-trace.native-records+jsonl+gzip"
NATIVE_CHUNK_ROLE = "native.chunk"
NATIVE_CHUNK_TARGET_BYTES = 1024 * 1024
_ROLE = re.compile(r"^[a-z][a-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class NativeContent:
    member: JsonObject
    matches: int


def retain_native_content(
    content: JsonValue | bytes,
    *,
    media_type: str,
    role: str,
    encoding: Literal["utf-8", "binary"],
    redactor: Redactor,
) -> NativeContent:
    """Sanitize native content and encode its retained bytes into one member."""

    if not media_type or len(media_type) > 128:
        raise ValueError("Native media type must contain 1-128 characters")
    if not _ROLE.fullmatch(role) or len(role) > 128:
        raise ValueError("Native artifact role is invalid")

    value, matches, rules = _retained_bytes(
        content,
        media_type=media_type,
        encoding=encoding,
        redactor=redactor,
    )
    return NativeContent(
        member={
            "content_sha256": hashlib.sha256(value).hexdigest(),
            "size_bytes": len(value),
            "media_type": media_type,
            "encoding": encoding,
            "role": role,
            "redaction": {
                "status": "applied" if matches else "not_required",
                "matches": matches,
                "rules": list(rules),
            },
            "content_base64": base64.b64encode(value).decode("ascii"),
        },
        matches=matches,
    )


def native_journal_record(
    *,
    native_record_id: str,
    sequence: int,
    trace_id: str,
    framework: str,
    recorded_at: str,
    source: str,
    event_ids: tuple[str, ...],
    content: NativeContent,
) -> JsonObject:
    return {
        "format": NATIVE_JOURNAL_FORMAT,
        "native_record_id": native_record_id,
        "sequence": sequence,
        "trace_id": trace_id,
        "framework": framework,
        "recorded_at": recorded_at,
        "source": source,
        "event_ids": list(dict.fromkeys(event_ids)),
        "member": {
            "native_record_id": native_record_id,
            **content.member,
        },
    }


def pack_native_journal(
    records: tuple[JsonObject, ...],
    *,
    artifact_store: ArtifactStore,
    schema_digest: str,
) -> tuple[JsonObject, ...]:
    """Convert durable internal records to v1 index entries and gzip chunks."""

    if not records:
        return ()
    if all(record.get("schema_version") == "benchmark-trace/v1" for record in records):
        return records
    if any(record.get("format") != NATIVE_JOURNAL_FORMAT for record in records):
        raise TraceStorageError("Native journal mixes incompatible record formats")

    chunks: list[list[JsonObject]] = []
    current: list[JsonObject] = []
    current_bytes = 0
    for record in records:
        member = _member(record)
        size = len(canonical_json_bytes(member))
        if current and current_bytes + size > NATIVE_CHUNK_TARGET_BYTES:
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(record)
        current_bytes += size
    if current:
        chunks.append(current)

    packed: list[JsonObject] = []
    for chunk in chunks:
        members = [_member(record) for record in chunk]
        content = gzip.compress(
            b"".join(canonical_json_bytes(member) for member in members),
            compresslevel=6,
            mtime=0,
        )
        redactions = [_redaction(member) for member in members]
        reference = artifact_store.put_retained_bytes(
            content,
            media_type=NATIVE_CHUNK_MEDIA_TYPE,
            encoding="binary",
            role=NATIVE_CHUNK_ROLE,
            redaction_matches=sum(_integer(value, "matches") for value in redactions),
            redaction_rules=tuple(
                dict.fromkeys(
                    str(rule)
                    for value in redactions
                    for rule in _string_list(value, "rules")
                )
            ),
        ).reference
        for record in chunk:
            entry: JsonObject = {
                "schema_version": "benchmark-trace/v1",
                "schema_digest": schema_digest,
                "native_record_id": _string(record, "native_record_id"),
                "sequence": _integer(record, "sequence"),
                "trace_id": _string(record, "trace_id"),
                "framework": _string(record, "framework"),
                "recorded_at": _string(record, "recorded_at"),
                "source": _string(record, "source"),
                "artifact": reference,
                "event_ids": [
                    cast(JsonValue, event_id)
                    for event_id in _string_list(record, "event_ids")
                ],
            }
            packed.append(entry)
    return tuple(packed)


def read_native_content(
    attempt_dir: Path,
    record: JsonObject,
    *,
    chunks: dict[str, dict[str, JsonObject]] | None = None,
) -> bytes:
    """Read one native record from either loose v1 or packed v1 storage."""

    reference = _object(record, "artifact")
    relative_path = _string(reference, "path")
    root = attempt_dir.resolve()
    candidate = root / relative_path
    path = candidate.resolve()
    if candidate.is_symlink() or not path.is_relative_to(root):
        raise TraceStorageError(
            f"Native artifact path is outside the attempt: {relative_path}"
        )
    if reference.get("media_type") != NATIVE_CHUNK_MEDIA_TYPE:
        try:
            return path.read_bytes()
        except OSError as exc:
            raise TraceStorageError(f"Native artifact is unreadable: {path}") from exc

    cache = chunks if chunks is not None else {}
    members = cache.get(relative_path)
    if members is None:
        members = read_native_chunk(path)
        cache[relative_path] = members
    native_record_id = _string(record, "native_record_id")
    member = members.get(native_record_id)
    if member is None:
        raise TraceStorageError(
            f"Native chunk omits indexed record: {native_record_id}"
        )
    return decode_native_member(member)


def read_native_chunk(path: Path) -> dict[str, JsonObject]:
    if path.is_symlink() or not path.is_file():
        raise TraceStorageError(f"Native chunk is missing or unsafe: {path}")
    try:
        content = gzip.decompress(path.read_bytes())
    except (OSError, gzip.BadGzipFile, EOFError) as exc:
        raise TraceStorageError(f"Native chunk is not valid gzip: {path}") from exc

    members: dict[str, JsonObject] = {}
    for index, line in enumerate(content.splitlines(), start=1):
        if not line:
            raise TraceStorageError(f"Native chunk has an empty line: {path}:{index}")
        try:
            member = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TraceStorageError(
                f"Native chunk has invalid JSON: {path}:{index}"
            ) from exc
        if not isinstance(member, dict):
            raise TraceStorageError(
                f"Native chunk member is not an object: {path}:{index}"
            )
        native_record_id = _string(member, "native_record_id")
        if native_record_id in members:
            raise TraceStorageError(f"Native chunk repeats record: {native_record_id}")
        decode_native_member(member)
        members[native_record_id] = member
    return members


def decode_native_member(member: JsonObject) -> bytes:
    try:
        content = base64.b64decode(
            _string(member, "content_base64"),
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise TraceStorageError("Native chunk member has invalid base64") from exc
    if len(content) != _integer(member, "size_bytes"):
        raise TraceStorageError("Native chunk member size does not match")
    if hashlib.sha256(content).hexdigest() != _string(member, "content_sha256"):
        raise TraceStorageError("Native chunk member digest does not match")
    if _string(member, "encoding") not in {"utf-8", "binary"}:
        raise TraceStorageError("Native chunk member encoding is invalid")
    _string(member, "media_type")
    role = _string(member, "role")
    if not _ROLE.fullmatch(role):
        raise TraceStorageError("Native chunk member role is invalid")
    _redaction(member)
    return content


def _retained_bytes(
    content: JsonValue | bytes,
    *,
    media_type: str,
    encoding: Literal["utf-8", "binary"],
    redactor: Redactor,
) -> tuple[bytes, int, tuple[str, ...]]:
    if isinstance(content, bytes):
        if media_type == "application/json" and encoding == "utf-8":
            try:
                parsed = json.loads(content)
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = None
            if parsed is not None:
                result = redactor.sanitize_json(parsed)
                return (
                    canonical_json_bytes(result.value),
                    result.matches,
                    result.rules,
                )
        if encoding == "utf-8":
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("Native UTF-8 content contains invalid bytes") from exc
            result = redactor.sanitize_text(text)
            return result.value.encode("utf-8"), result.matches, result.rules
        result = redactor.sanitize_bytes(content)
        return result.value, result.matches, result.rules

    if isinstance(content, str) and media_type != "application/json":
        result = redactor.sanitize_text(content)
        return result.value.encode("utf-8"), result.matches, result.rules
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            result = redactor.sanitize_text(content)
            return result.value.encode("utf-8"), result.matches, result.rules
        result = redactor.sanitize_json(parsed)
        return canonical_json_bytes(result.value), result.matches, result.rules
    result = redactor.sanitize_json(content)
    return canonical_json_bytes(result.value), result.matches, result.rules


def _member(record: JsonObject) -> JsonObject:
    member = _object(record, "member")
    if _string(member, "native_record_id") != _string(record, "native_record_id"):
        raise TraceStorageError("Native journal member identity does not match")
    decode_native_member(member)
    return member


def _redaction(member: JsonObject) -> JsonObject:
    value = _object(member, "redaction")
    status = _string(value, "status")
    matches = _integer(value, "matches")
    rules = _string_list(value, "rules")
    if status not in {"applied", "not_required"}:
        raise TraceStorageError("Native member redaction status is invalid")
    if (status == "applied") != (matches > 0):
        raise TraceStorageError("Native member redaction count is inconsistent")
    if (matches > 0) != bool(rules):
        raise TraceStorageError("Native member redaction rules are inconsistent")
    return value


def _object(value: JsonObject, key: str) -> JsonObject:
    item = value.get(key)
    if not isinstance(item, dict):
        raise TraceStorageError(f"Native record field is not an object: {key}")
    return item


def _string(value: JsonObject, key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise TraceStorageError(f"Native record field is not a string: {key}")
    return item


def _integer(value: JsonObject, key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool) or item < 0:
        raise TraceStorageError(f"Native record field is not an integer: {key}")
    return item


def _string_list(value: JsonObject, key: str) -> list[str]:
    item = value.get(key)
    if not isinstance(item, list) or any(not isinstance(entry, str) for entry in item):
        raise TraceStorageError(f"Native record field is not a string list: {key}")
    return [str(entry) for entry in item]
