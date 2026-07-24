"""Typed values shared by the benchmark trace recorder."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4


type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type TraceSeverity = Literal["warning", "error"]


@dataclass(frozen=True, slots=True)
class TraceIdentity:
    """Stable identity shared by every document in one attempt trace."""

    trace_id: str
    run_id: str
    benchmark: str
    framework: str
    instance_id: str
    attempt: int

    def __post_init__(self) -> None:
        values = (
            self.trace_id,
            self.run_id,
            self.benchmark,
            self.framework,
            self.instance_id,
        )
        if any(not value for value in values):
            raise ValueError("Trace identity fields must not be empty")
        if self.attempt < 1:
            raise ValueError("Trace attempt must be positive")

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        benchmark: str,
        framework: str,
        instance_id: str,
        attempt: int = 1,
    ) -> "TraceIdentity":
        return cls(
            trace_id=f"trace-{uuid4().hex}",
            run_id=run_id,
            benchmark=benchmark,
            framework=framework,
            instance_id=instance_id,
            attempt=attempt,
        )

    def event_fields(self) -> JsonObject:
        return {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "benchmark": self.benchmark,
            "framework": self.framework,
            "instance_id": self.instance_id,
            "attempt": self.attempt,
        }


@dataclass(frozen=True, slots=True)
class TraceProducer:
    name: str
    version: str

    def as_json(self) -> JsonObject:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True, slots=True)
class Capability:
    category: str
    state: str
    coverage: str
    timing: str
    evidence: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def as_json(self) -> JsonObject:
        return {
            "category": self.category,
            "state": self.state,
            "coverage": self.coverage,
            "timing": self.timing,
            "evidence": list(self.evidence),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class TraceConfig:
    """Preflight configuration for one attempt recorder."""

    attempt_dir: Path
    identity: TraceIdentity
    producer: TraceProducer
    provenance: JsonObject
    execution: JsonObject
    capabilities: tuple[Capability, ...]


@dataclass(frozen=True, slots=True)
class TraceIssue:
    severity: TraceSeverity
    code: str
    message: str
    first_seen_at: str
    last_seen_at: str
    count: int = 1

    def as_json(self) -> JsonObject:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "count": self.count,
        }
