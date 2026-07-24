# Benchmark tracing

`benchmarks.tracing` defines a framework-neutral, research-grade trace contract
for agent benchmark execution. OpenCode, OpenHands, and Hermes will each use
framework-native instrumentation adapters while emitting the same normalized
format.

Phase 1 contains the contract only. It does not enable tracing in a benchmark,
call a model, alter agent prompts or delegation, or patch Harbor. Runtime
recorders and framework adapters are later phases.

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
