"""Content-addressed, policy-filtered trace artifact storage."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from benchmarks.tracing.errors import TraceStorageError
from benchmarks.tracing.models import JsonObject, JsonValue
from benchmarks.tracing.redaction import RedactionResult, Redactor
from benchmarks.tracing.storage import (
    FILE_MODE,
    atomic_write,
    canonical_json_bytes,
    ensure_private_directory,
)


@dataclass(frozen=True, slots=True)
class ArtifactWrite:
    reference: JsonObject
    created: bool


class ArtifactStore:
    """Store immutable artifacts under their sanitized SHA-256 digest."""

    def __init__(self, attempt_dir: Path, redactor: Redactor | None = None) -> None:
        self._attempt_dir = attempt_dir
        self._root = attempt_dir / "artifacts" / "sha256"
        self._redactor = redactor or Redactor()
        ensure_private_directory(attempt_dir / "artifacts")
        ensure_private_directory(self._root)

    def put_text(
        self,
        value: str,
        *,
        media_type: str,
        role: str,
    ) -> ArtifactWrite:
        result = self._redactor.sanitize_text(value)
        return self._persist(
            RedactionResult(
                value=result.value.encode("utf-8"),
                matches=result.matches,
                rules=result.rules,
            ),
            media_type=media_type,
            encoding="utf-8",
            role=role,
        )

    def put_bytes(
        self,
        value: bytes,
        *,
        media_type: str,
        encoding: str,
        role: str,
    ) -> ArtifactWrite:
        if encoding not in {"utf-8", "binary"}:
            raise ValueError(f"Unsupported artifact encoding: {encoding}")
        if encoding == "utf-8":
            try:
                value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("UTF-8 artifact contains invalid bytes") from exc
        return self._persist(
            self._redactor.sanitize_bytes(value),
            media_type=media_type,
            encoding=encoding,
            role=role,
        )

    def put_json(
        self,
        value: JsonValue,
        *,
        role: str,
        media_type: str = "application/json",
    ) -> ArtifactWrite:
        result = self._redactor.sanitize_json(value)
        return self._persist(
            RedactionResult(
                value=canonical_json_bytes(result.value),
                matches=result.matches,
                rules=result.rules,
            ),
            media_type=media_type,
            encoding="utf-8",
            role=role,
        )

    def _persist(
        self,
        result: RedactionResult[bytes],
        *,
        media_type: str,
        encoding: str,
        role: str,
    ) -> ArtifactWrite:
        if not media_type or not role:
            raise ValueError("Artifact media type and role must not be empty")

        digest = hashlib.sha256(result.value).hexdigest()
        relative_path = f"artifacts/sha256/{digest[:2]}/{digest}"
        path = self._attempt_dir / relative_path
        ensure_private_directory(path.parent)
        created = self._write_if_missing(path, result.value)
        reference: JsonObject = {
            "sha256": digest,
            "path": relative_path,
            "size_bytes": len(result.value),
            "media_type": media_type,
            "encoding": encoding,
            "role": role,
            "redaction": {
                "status": "applied" if result.matches else "not_required",
                "matches": result.matches,
                "rules": list(result.rules),
            },
        }
        return ArtifactWrite(reference=reference, created=created)

    def _write_if_missing(self, path: Path, content: bytes) -> bool:
        if path.is_symlink():
            raise TraceStorageError(
                f"Artifact path must not be a symbolic link: {path}"
            )
        if path.exists():
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise TraceStorageError(
                    f"Failed to read existing artifact: {path}"
                ) from exc
            if existing != content:
                raise TraceStorageError(
                    f"Artifact digest collision or corruption detected: {path}"
                )
            path.chmod(FILE_MODE)
            return False
        atomic_write(path, content)
        return True
