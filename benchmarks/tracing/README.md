# Benchmark tracing

`benchmarks.tracing` defines a framework-neutral, research-grade trace contract
for agent benchmark execution. OpenCode, OpenHands, and Hermes will each use
framework-native instrumentation adapters while emitting the same normalized
format.

Phase 4 adds the second framework-native adapter. The OpenHands and OpenCode
adapters are wired opt-in to their respective SWE-bench Verified and SWE-bench
Pro inference runners. Neither calls a model during setup or validation, alters
agent prompts or delegation, or patches Harbor. Hermes and Terminal-Bench
wiring remain later phases.

## Design boundary

The normalized trace records observable activity: lifecycle boundaries, agent
sessions, model/provider exchanges where exposed, tool calls and results,
commands, file/search/browser activity, delegation, context compaction, memory
activity, patch extraction, and evaluation lifecycle. Framework-native evidence
is retained alongside normalized events after applying the same safety policy.

Harbor is an outer execution harness, not an agent framework. Terminal-Bench
integrations will emit only generic harness/container/evaluator lifecycle events
around the framework-native trace. Existing Harbor logs, results, and ATIF data
remain auxiliary artifacts rather than a fourth tracing adapter.

The contract intentionally excludes:

- hidden provider or framework internals that are not exposed;
- operating-system syscalls beneath a framework tool invocation;
- token usage and cost accounting;
- automatic snapshots of the entire workspace;
- credentials, authentication material, or unrestricted environment dumps.

## Versioned assets

- [`spec/v1/vocabulary.md`](spec/v1/vocabulary.md) defines identity, event,
  timing, capability, ordering, and recovery semantics.
- [`spec/v1/redaction.md`](spec/v1/redaction.md) defines mandatory pre-persistence
  sanitization and retention behavior.
- `spec/v1/*.schema.json` are JSON Schema Draft 2020-12 contracts.
- [`conformance/v1/README.md`](conformance/v1/README.md) describes valid and
  invalid fixtures shared by Python and TypeScript implementations.
- [`runtime.md`](runtime.md) documents the Python lifecycle, failure boundary,
  recovery behavior, and adapter-facing API.

`benchmark-trace/v1` is the compatibility label. `1.0.0` is the contract release.
Every trace also records the deterministic SHA-256 digest of the complete schema
bundle so an implementation can detect schema drift exactly.

## Trace layout

```text
<trace-root>/
├── run.json
└── instances/
    └── <percent-encoded-instance-id>/
        └── attempt-1/
            ├── manifest.json
            ├── journal.jsonl
            ├── events.jsonl
            ├── capabilities.json
            ├── health.json
            ├── native/
            │   └── index.jsonl
            └── artifacts/
                └── sha256/
                    └── <digest-prefix>/
                        └── <digest>
```

Instance path components use uppercase percent-encoding of UTF-8 bytes for
characters outside `[A-Za-z0-9._~-]`. Identity always comes from the unmodified
`instance_id` inside the documents, not by decoding the directory name.

`journal.jsonl` is the crash-tolerant append log. `events.jsonl` is the validated,
deterministically finalized event stream. JSONL files contain one compact JSON
object per line and end with a newline.

## Failure semantics

Tracing must not change benchmark behavior. Trace initialization failure is a
preflight error and aborts before any provider request. Once an agent attempt
starts, tracing failure must not interrupt or retry it. The benchmark result is
preserved, trace health becomes degraded or failed, and trace validation reports
the observability failure separately.

## Python reference implementation

The reference implementation provides:

- synchronous, thread-safe journal appends with an `fsync` durability boundary;
- content-addressed artifacts sanitized before hashing or persistence;
- deterministic journal finalization and torn-final-line recovery;
- fresh-process recovery for stopped recorders without resuming agent execution;
- schema and semantic validation across attempts and run indexes;
- nested, timestamped timeline reconstruction;
- fixed failure behavior: runtime recording methods return `None` on trace
  failure, while finalization reports observability validity independently.

The implementation is intentionally single-writer per attempt. Framework
adapters must funnel concurrent events through one recorder rather than opening
the same journal from multiple processes.

## OpenHands adapter

Pass `--trace-dir <base-directory>` to the SWE-bench Verified or SWE-bench Pro
inference CLI. Tracing is disabled when the flag is absent. Each invocation
creates a private `trace-run-<uuid>` directory beneath the supplied base so a
later invocation cannot overwrite an earlier run.

The adapter consumes the native synchronous `Conversation` callback stream. It
retains sanitized native evidence and normalizes observable model responses,
tool inputs and complete outputs, shell/file/search/browser activity,
delegation boundaries, ACP tool activity, messages, hooks, state updates,
errors, pauses, and condenser boundaries. Native action/observation timestamps
are paired to derive tool durations. Trace initialization occurs before
workspace or provider work; trace finalization occurs only after the benchmark
outcome is fixed.

Provider completion logs are not enabled by the benchmark integration because
the SDK's separate built-in completion logger writes outside this recorder's
pre-persistence safety boundary. If an already-enabled native completion-log
event reaches the adapter, its structured credentials and token/cost accounting
are removed before retention.

The parent remote conversation exposes a delegated task's boundary and result,
but does not forward the internal subagent conversation event stream. The
capability matrix reports that limitation rather than claiming complete
delegation coverage. Token events and token/cost accounting are intentionally
excluded by contract. Memory and harness/container/evaluator lifecycle are also
reported as not exposed by this adapter; those require their own observable
integration boundaries.

The evaluator's outer timeout cancels an asyncio task but cannot terminate its
already-running worker thread. The default native inference deadline is ten
minutes shorter, so it normally finalizes first with `timeout` status. If a
worker ignores both interruption and the native deadline, `run.json` is omitted
rather than claiming a complete run; the durable attempt journal remains
available for explicit recorder recovery after the process stops.

## OpenCode adapter

The OpenCode implementation is a TypeScript recorder and adapter in the
OpenCode repository. Passing `--trace-dir <base-directory>` to its SWE-bench
Verified or SWE-bench Pro inference CLI creates a private
`trace-run-<uuid>` directory and emits the same `benchmark-trace/v1` contract.
Tracing is disabled when the flag is absent, requires a fresh benchmark run,
and is initialized before container setup or any provider request.

An internal opt-in CLI stream publishes OpenCode's native event bus with a
strict per-process sequence. The adapter retains sanitized native evidence and
normalizes root and child sessions, assistant-message model boundaries,
pending/running/final tool state, complete inputs and outputs, shell/file/search
and browser activity, native task delegation, compaction boundaries, errors,
and native wall-clock durations. It observes all subagent session events on the
same native event stream without replacing or constraining OpenCode's task
delegation behavior.

Exact provider request and response bodies are not exposed by this mode, and
operating-system activity below an OpenCode tool invocation is outside the
observable boundary. The capability report states those limitations. Credential
fields and recognizable credential text are removed or replaced before
persistence; token usage and cost accounting are excluded. A post-start tracing
failure is isolated from benchmark results and cannot trigger a benchmark or
provider retry.
