# Python reference recorder

The Python recorder is the normative implementation of the Phase 1 contract. It
contains no framework-specific behavior and makes no provider, benchmark, Docker,
or network call.

## Lifecycle

1. Construct a `TraceConfig` with a complete capability matrix, immutable
   identity, revision provenance, and effective execution settings.
2. Construct `TraceRecorder(config)` before starting an agent or issuing a
   provider request.
3. Submit normalized events, native evidence, and artifacts while the attempt
   runs.
4. Call `finalize()` after the agent outcome is already known.
5. Inspect `FinalizationResult.health` and `FinalizationResult.validation`
   separately from the benchmark result.
6. Write `run.json` through `write_run_index` after all selected attempts are
   finalized.

Construction is the preflight boundary. Invalid configuration, incomplete
capabilities, unsafe paths, or unavailable storage raise
`TraceInitializationError` before inference can begin.

After successful construction, expected trace input and storage failures do not
raise into the agent loop. `record_event`, `record_native`, and artifact methods
return `None`, aggregate a non-sensitive health issue, and allow the agent to
continue without a benchmark or provider retry. `finalize()` can raise
`TraceFinalizationError` after agent execution if durable trace files cannot be
recovered or written.

## Configuration sketch

```python
from benchmarks.tracing import (
    TraceConfig,
    TraceIdentity,
    TraceProducer,
    TraceRecorder,
    attempt_directory,
)

identity = TraceIdentity.create(
    run_id="run-...",
    benchmark="swe-bench-verified",
    framework="openhands",
    instance_id="owner/project__issue-1",
)
config = TraceConfig(
    attempt_dir=attempt_directory(trace_root, identity.instance_id, 1),
    identity=identity,
    producer=TraceProducer("openhands-adapter", "1.0.0"),
    provenance=provenance,
    execution=effective_execution_config,
    capabilities=complete_capability_matrix,
)
trace = TraceRecorder(config)
```

Adapters store lossless output before referencing it from an event:

```python
stdout = trace.store_text_artifact(
    command_stdout,
    media_type="text/plain",
    role="tool.stdout",
)
event_id = trace.record_event(
    event_type="shell.end",
    event_family="shell",
    phase="end",
    status="completed",
    span_id=span_id,
    origin=native_origin,
    timing=native_timing,
    payload={"exit_code": exit_code},
    artifacts=(stdout,) if stdout is not None else (),
)
```

An artifact reference is accepted only if the same recorder created it. Native
JSON should be passed as a mapping, list, or JSON byte stream so structured
authentication and usage fields can be removed before it is retained. Arbitrary
shell output should be stored as text or bytes; source text mentioning
`cost.py`, tokens, or similarly named research content is preserved.

## Durability and recovery

Each accepted event is serialized as canonical compact JSON and appended under a
thread lock. The recorder calls `fsync` before returning its event identifier.
Artifacts and finalized documents use private temporary files, file `fsync`,
atomic replacement, and directory `fsync`. Directories use mode `0700`; files
use mode `0600`.

The finalizer rejects malformed complete journal records. It may discard only an
unterminated final line, records recovery in `health.json`, increments
`dropped_events`, and marks the manifest incomplete.

If the process that owned the live recorder stops, the coordinator can rebuild a
finalizer from the same non-secret configuration:

```python
recovered = TraceRecorder.recover(config)
result = recovered.finalize()
```

Recovery never resumes event append or agent execution. It reads the durable
journals, reconstructs retained redaction counts where observable, finalizes the
attempt as degraded/recovered, and validates the result. Structured fields that
were removed entirely cannot be counted exactly after a process loss; this is
why recovered traces cannot be marked complete.

## Validation and timelines

`ContractValidator.validate_attempt(path)` checks schemas, exact schema digest,
identity, sequences, native links, event relations, capability evidence, span
boundaries and parent cycles, artifact metadata and bytes, redaction policy,
health counters, permissions, and journal/finalized-event agreement.

`ContractValidator.validate_run(path)` additionally checks selection/attempt
parity, canonical instance paths, unique attempt identity, terminal status, and
every referenced attempt trace.

`build_timeline(path)` reconstructs sequence, wall-clock offset, nesting, actor,
duration, artifact roles, and command detail. `render_timeline(entries)` produces
a deterministic human-readable view. Neither function reads artifact contents or
adds model calls.

## Adapter responsibilities

- Initialize tracing before any action that can spend API credits.
- Never pass a complete process environment; construct explicit metadata.
- Preserve framework-native IDs and timing fidelity honestly.
- Use one recorder as the single writer for an attempt.
- Treat `None` from a runtime recording method as trace degradation, not as an
  agent failure or retry signal.
- Finalize after preserving the benchmark result.
- Keep framework-native files outside the trace only when they cannot be
  intercepted safely; the normalized trace must never copy them unsanitized.
