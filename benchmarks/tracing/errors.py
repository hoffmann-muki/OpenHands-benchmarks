"""Errors raised by the benchmark trace recorder."""


class TraceError(RuntimeError):
    """Base class for trace subsystem failures."""


class TraceInitializationError(TraceError):
    """Raised when tracing cannot be initialized safely before agent execution."""


class TraceFinalizationError(TraceError):
    """Raised when an attempt trace cannot be finalized."""


class TraceStorageError(TraceError):
    """Raised when durable trace storage cannot be read or written."""


class TraceValidationError(TraceError):
    """Raised when a trace document or finalized trace is invalid."""
