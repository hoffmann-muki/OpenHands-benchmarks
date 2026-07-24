"""Framework-native instrumentation adapters for benchmark tracing."""

from benchmarks.tracing.adapters.openhands import (
    OpenHandsTraceAdapter,
    openhands_capabilities,
)


__all__ = ["OpenHandsTraceAdapter", "openhands_capabilities"]
