"""Private, durable storage primitives for benchmark traces."""

import json
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from benchmarks.tracing.errors import TraceStorageError
from benchmarks.tracing.models import JsonObject, JsonValue


DIRECTORY_MODE = 0o700
FILE_MODE = 0o600


@dataclass(frozen=True, slots=True)
class JournalRead:
    records: tuple[JsonObject, ...]
    torn_final_line: bool


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def format_timestamp(value: datetime | str | None) -> str:
    if value is None:
        return utc_now()
    if isinstance(value, str):
        return value
    timestamp = value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    return timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical_json_bytes(value: JsonValue) -> bytes:
    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise TraceStorageError("Trace value is not canonical JSON") from exc
    return serialized.encode("utf-8") + b"\n"


def create_private_directory(path: Path) -> None:
    """Create a new directory and all missing parents with private permissions."""

    if path.exists() or path.is_symlink():
        raise TraceStorageError(f"Trace directory already exists: {path}")

    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        if cursor.is_symlink():
            raise TraceStorageError(f"Trace path contains a symbolic link: {cursor}")
        missing.append(cursor)
        cursor = cursor.parent

    if not cursor.is_dir() or cursor.is_symlink():
        raise TraceStorageError(f"Trace parent is not a safe directory: {cursor}")

    for directory in reversed(missing):
        try:
            directory.mkdir(mode=DIRECTORY_MODE)
        except FileExistsError as exc:
            raise TraceStorageError(
                f"Trace directory was created concurrently: {directory}"
            ) from exc
        directory.chmod(DIRECTORY_MODE)


def ensure_private_directory(path: Path) -> None:
    if path.is_symlink():
        raise TraceStorageError(f"Trace directory must not be a symbolic link: {path}")
    path.mkdir(mode=DIRECTORY_MODE, parents=True, exist_ok=True)
    if not path.is_dir():
        raise TraceStorageError(f"Trace path is not a directory: {path}")
    path.chmod(DIRECTORY_MODE)


def atomic_write(path: Path, content: bytes) -> None:
    """Write one private file atomically and fsync its containing directory."""

    ensure_private_directory(path.parent)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, FILE_MODE)
        _write_all(descriptor, content)
        os.fsync(descriptor)
        os.fchmod(descriptor, FILE_MODE)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        path.chmod(FILE_MODE)
        _fsync_directory(path.parent)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise TraceStorageError(f"Failed to write trace file: {path}") from exc


def replace_with_hard_link(source: Path, target: Path) -> None:
    """Atomically make ``target`` share ``source`` storage when supported.

    Finalized traces retain both contract paths for v1 compatibility, but do not
    need two physical copies of the validated event stream. Filesystems without
    hard-link support fall back to an ordinary atomic copy.
    """

    if source.is_symlink() or not source.is_file():
        raise TraceStorageError(f"Trace link source is not a safe file: {source}")
    ensure_private_directory(target.parent)
    temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
    try:
        os.link(source, temporary, follow_symlinks=False)
        temporary.chmod(FILE_MODE)
        os.replace(temporary, target)
        target.chmod(FILE_MODE)
        _fsync_directory(target.parent)
    except OSError:
        temporary.unlink(missing_ok=True)
        atomic_write(target, source.read_bytes())


def read_jsonl(path: Path, *, allow_torn_final_line: bool) -> JournalRead:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise TraceStorageError(f"Failed to read trace journal: {path}") from exc

    torn_final_line = bool(content) and not content.endswith(b"\n")
    if torn_final_line and not allow_torn_final_line:
        raise TraceStorageError(f"Trace JSONL has a torn final line: {path}")

    lines = content.splitlines()
    if torn_final_line:
        lines = lines[:-1]

    records: list[JsonObject] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise TraceStorageError(
                f"Trace JSONL contains an empty record at line {line_number}: {path}"
            )
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TraceStorageError(
                f"Trace JSONL contains invalid JSON at line {line_number}: {path}"
            ) from exc
        if not isinstance(record, dict):
            raise TraceStorageError(
                f"Trace JSONL record is not an object at line {line_number}: {path}"
            )
        records.append(record)
    return JournalRead(records=tuple(records), torn_final_line=torn_final_line)


class JsonlJournal:
    """Synchronous JSONL append log with an fsync durability boundary."""

    def __init__(self, path: Path, descriptor: int) -> None:
        self.path = path
        self._descriptor: int | None = descriptor
        self._lock = threading.Lock()

    @classmethod
    def create(cls, path: Path) -> "JsonlJournal":
        ensure_private_directory(path.parent)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, FILE_MODE)
            os.fchmod(descriptor, FILE_MODE)
            _fsync_directory(path.parent)
        except OSError as exc:
            raise TraceStorageError(f"Failed to create trace journal: {path}") from exc
        return cls(path, descriptor)

    @property
    def closed(self) -> bool:
        return self._descriptor is None

    def append(self, record: JsonObject) -> None:
        content = canonical_json_bytes(record)
        with self._lock:
            if self._descriptor is None:
                raise TraceStorageError(f"Trace journal is closed: {self.path}")
            try:
                _write_all(self._descriptor, content)
                os.fsync(self._descriptor)
            except OSError as exc:
                raise TraceStorageError(
                    f"Failed to append trace journal: {self.path}"
                ) from exc

    def close(self) -> None:
        with self._lock:
            if self._descriptor is None:
                return
            try:
                os.fsync(self._descriptor)
                os.close(self._descriptor)
            except OSError as exc:
                raise TraceStorageError(
                    f"Failed to close trace journal: {self.path}"
                ) from exc
            finally:
                self._descriptor = None


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written == 0:
            raise OSError("write returned zero bytes")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
