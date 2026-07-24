"""Framework-neutral benchmark tracing contract and Python reference recorder."""

from benchmarks.tracing.errors import (
    TraceError,
    TraceFinalizationError,
    TraceInitializationError,
    TraceStorageError,
    TraceValidationError,
)
from benchmarks.tracing.models import (
    Capability,
    TraceConfig,
    TraceIdentity,
    TraceProducer,
)
from benchmarks.tracing.recorder import (
    FinalizationResult,
    TraceRecorder,
    attempt_directory,
    encode_instance_id,
    write_run_index,
)
from benchmarks.tracing.redaction import RedactionResult, Redactor
from benchmarks.tracing.timeline import (
    TimelineEntry,
    build_timeline,
    render_timeline,
)
from benchmarks.tracing.validation import (
    ContractValidator,
    ValidationIssue,
    ValidationReport,
)


CONTRACT_NAME = "benchmark-trace"
CONTRACT_VERSION = "1.0.0"
SCHEMA_VERSION = "benchmark-trace/v1"


__all__ = [
    "CONTRACT_NAME",
    "CONTRACT_VERSION",
    "SCHEMA_VERSION",
    "Capability",
    "ContractValidator",
    "FinalizationResult",
    "TimelineEntry",
    "TraceConfig",
    "TraceError",
    "TraceFinalizationError",
    "TraceIdentity",
    "TraceInitializationError",
    "TraceProducer",
    "TraceRecorder",
    "TraceStorageError",
    "TraceValidationError",
    "ValidationIssue",
    "ValidationReport",
    "attempt_directory",
    "build_timeline",
    "encode_instance_id",
    "RedactionResult",
    "Redactor",
    "render_timeline",
    "write_run_index",
]
