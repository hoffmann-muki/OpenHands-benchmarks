# Benchmark Trace v1 vocabulary

This document defines the stable vocabulary used by `benchmark-trace/v1`. JSON
Schema validates individual records; the requirements below define ordering,
correlation, completeness, and cross-file behavior.

## Identity and correlation

- A run is one framework executing one benchmark against one selected instance
  set. `run_id` is unique across invocations.
- A trace is one benchmark instance attempt. `trace_id` is unique across runs,
  instances, and attempts.
- `event_id` is unique within a trace. `sequence` starts at 1 and increases by
  exactly one in journal append order.
- `span_id` identifies an activity. Start and end events for the same activity
  share it. `parent_span_id` expresses nesting without requiring timestamps from
  different clock domains to be directly comparable.
- `session_id`, `agent_id`, `parent_agent_id`, and `turn_id` preserve native
  framework identity when it exists. Adapters must not invent native identity;
  generated canonical identity must be documented in the capability report.
- Native records use a separate monotonically increasing sequence. A native
  record can support zero or more normalized events through `event_ids`.

## Event types

Event types are dot-delimited, lowercase names. Adapters should use the most
specific applicable type from this baseline and may add types without changing
the schema:

- `instance.start`, `instance.end`
- `attempt.start`, `attempt.end`
- `harness.start`, `harness.end`
- `container.start`, `container.end`
- `evaluator.start`, `evaluator.end`
- `agent.session_start`, `agent.session_end`
- `model.request`, `model.response`, `provider.request`, `provider.response`
- `tool.start`, `tool.end`
- `shell.start`, `shell.end`
- `file.read`, `file.write`, `file.patch`
- `search.start`, `search.end`
- `browser.start`, `browser.end`
- `delegation.start`, `delegation.end`
- `context.compaction_start`, `context.compaction_end`
- `memory.read`, `memory.write`, `memory.update`
- `patch.extract_start`, `patch.extract_end`
- `trace.issue`

`phase=start` requires `status=started`. `phase=end` carries the terminal status.
Instant events describe an already-observed fact and do not imply a matching
event.

## Time

`occurred_at` is the best available UTC wall-clock time for the native boundary.
`recorded_at` is when the adapter accepted the event. Both use RFC 3339.

Timing fidelity is explicit:

- `native_monotonic`: the framework exposed a monotonic measurement. The clock
  domain is named by `clock_id`.
- `native_wall`: duration comes directly from native wall-clock timestamps.
- `derived`: the adapter paired observable boundaries and calculated duration.
- `not_available`: the source exposed no defensible duration.

Monotonic values are comparable only inside the same `clock_id`. A completed
monotonic span includes start, end, and duration. Start events may contain only
the start value. Cross-process ordering relies on causal identifiers and wall
timestamps, not on comparing unrelated monotonic clocks.

## Artifacts

Large or lossless content is stored as an immutable artifact. References include
the SHA-256 digest of the bytes actually written, their byte length, media type,
text/binary encoding, semantic role, and redaction result. The path is always:

```text
artifacts/sha256/<first two digest characters>/<full digest>
```

Artifacts are written atomically and deduplicated by digest. An adapter must not
truncate non-secret content to fit an event. Small structured metadata may stay
inline in `payload`; full command output, provider bodies when exposed, native
exports, and large tool results belong in artifacts.

High-frequency native records may share a lossless chunk artifact with media
type
`application/vnd.benchmark-trace.native-records+jsonl+gzip`. The decompressed
JSONL contains one member per indexed `native_record_id`. Each member preserves
the retained content bytes as base64 together with their original media type,
encoding, role, SHA-256 digest, byte length, and redaction result. Every native
index row retains its own sequence, source, timestamp, canonical-event links,
and record identity even when many rows reference the same chunk. Chunking is a
physical representation only; it must not coalesce, sample, or discard native
records.

All content is sanitized before hashing or persistence. Therefore a digest
identifies the retained, policy-compliant bytes rather than sensitive source
bytes.

## Capability states

Every adapter reports every category listed by the capability schema exactly
once:

- `captured`: observed directly through a native hook, stream, or export.
- `derived`: reconstructed from observable native boundaries.
- `not_exposed`: the framework does not expose the information in this mode.
- `disabled`: the benchmark configuration deliberately disables the feature.
- `not_observed`: supported by the adapter but absent in this attempt.

`coverage` describes content completeness; `timing` independently describes time
fidelity. Missing activity must never be interpreted as unavailable activity
without consulting `capabilities.json`.

## Journal and finalization

Each accepted event is synchronously appended to `journal.jsonl` before it is
considered durable. A finalizer validates and deterministically rewrites complete
records to `events.jsonl`. It may discard only a torn final journal line.
Finalized implementations may hard-link `journal.jsonl` to `events.jsonl` so
the two required v1 paths share one immutable byte stream. Before finalization,
native evidence is likewise appended to a single private journal and converted
to lossless content-addressed chunks only after the durable stream is closed.
Warnings produce `degraded` health; any error-level observability defect
produces `failed` health. Rejected normalized events and discarded torn event
records increment `dropped_events`. Only a healthy trace may use `clean`
finalization and set its manifest `complete` flag.

Tracing is observational. Initialization failure aborts before agent/provider
work begins. A failure after agent work starts must not interrupt, retry, or
otherwise change the agent attempt. It is recorded in `health.json`, and trace
validation fails separately from benchmark evaluation.

## Schema bundle digest

The schema digest fingerprints all `*.schema.json` files in this directory:

1. Sort files by filename.
2. Parse each file as JSON and serialize it as UTF-8 with keys sorted, no
   insignificant whitespace, and non-ASCII characters preserved.
3. For each file, hash `filename`, a newline, the canonical JSON, and a newline.
4. Store the lowercase SHA-256 hex digest in every contract header or
   `schema_digest` field.

Changing any schema changes the digest. A vocabulary clarification that does not
change a schema does not change it.
