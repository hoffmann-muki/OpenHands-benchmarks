"""Framework-neutral benchmark tracing contract and Python reference recorder."""

from benchmarks.tracing.constants import (
    CONTRACT_NAME,
    CONTRACT_VERSION,
    SCHEMA_VERSION,
)
from benchmarks.tracing.errors import (
    TraceError,
    TraceFinalizationError,
    TraceInitializationError,
    TraceStorageError,
    TraceValidationError,
)
from benchmarks.tracing.integration import (
    DirectTraceHarness,
    TraceHarnessAdapter,
    TraceRun,
    TraceSelection,
    TraceSelectionStrategy,
    attach_trace_run,
    create_trace_run,
    finalize_trace_run,
)
from benchmarks.tracing.models import (
    Capability,
    TraceConfig,
    TraceIdentity,
    TraceProducer,
)
from benchmarks.tracing.native import (
    NATIVE_CHUNK_MEDIA_TYPE,
    read_native_content,
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


__all__ = [
    "CONTRACT_NAME",
    "CONTRACT_VERSION",
    "SCHEMA_VERSION",
    "Capability",
    "ContractValidator",
    "DirectTraceHarness",
    "FinalizationResult",
    "NATIVE_CHUNK_MEDIA_TYPE",
    "TimelineEntry",
    "TraceConfig",
    "TraceError",
    "TraceFinalizationError",
    "TraceHarnessAdapter",
    "TraceIdentity",
    "TraceInitializationError",
    "TraceProducer",
    "TraceRecorder",
    "TraceRun",
    "TraceSelection",
    "TraceSelectionStrategy",
    "TraceStorageError",
    "TraceValidationError",
    "ValidationIssue",
    "ValidationReport",
    "attempt_directory",
    "attach_trace_run",
    "build_timeline",
    "create_trace_run",
    "encode_instance_id",
    "finalize_trace_run",
    "RedactionResult",
    "Redactor",
    "render_timeline",
    "read_native_content",
    "write_run_index",
]
