# Benchmark tracing

`benchmarks.tracing` defines a framework-neutral, research-grade trace contract
for agent benchmark execution. OpenCode, OpenHands, and Hermes use
framework-native instrumentation adapters while emitting the same normalized
format.

The integration wires the three framework-native adapters into SWE-bench
Verified, SWE-bench Pro, and Terminal-Bench 2.1. Tracing remains opt-in, does
not alter agent prompts or native delegation, and does not patch Harbor.
Development and contract validation use synthetic native events and do not
call a model.

## Portable integration model

Tracing is composed from independent layers rather than implemented by each
benchmark:

1. The generic run coordinator owns private run creation, canonical instance
   selection, attempt discovery, coverage checks, and `run.json`.
2. A framework adapter translates the agent's native event stream. The
   OpenCode, OpenHands, and Hermes adapters do not know which benchmark invoked
   them.
3. A harness adapter supplies only harness-owned topology and lifecycle facts.
   `DirectTraceHarness` accepts the instance order from an ordinary benchmark
   runner. `HarborTraceHarness` resolves task order from Harbor's lock file and
   accounts for concurrent infrastructure attempts.
4. The recorder, safety policy, contract validator, artifact store, and timeline
   tooling are shared unchanged.

A new benchmark using an existing framework and direct runner creates a
`TraceRun`, constructs the existing framework adapter for each attempt, and
finalizes with `DirectTraceHarness`. A new Harbor dataset uses the existing
Harbor adapter instead. No new event extraction or trace schema is required.
Only a genuinely new execution harness needs a `TraceHarnessAdapter`
implementation. Each framework runtime implements that harness boundary once
and reuses it across benchmarks, instead of creating a
benchmark/framework-specific integration.

Benchmark-specific values are limited to identity, provenance, effective
execution settings, and the selected instance order. Optional benchmark facts
belong in normalized metadata or artifacts rather than new recorder logic.
Unobservable harness internals remain explicit capability limitations.

## Design boundary

The normalized trace records observable activity: lifecycle boundaries, agent
sessions, model/provider exchanges where exposed, tool calls and results,
commands, file/search/browser activity, delegation, context compaction, memory
activity, patch extraction, and evaluation lifecycle. Framework-native evidence
is retained alongside normalized events after applying the same safety policy.

Harbor is an outer execution harness, not an agent framework. Terminal-Bench
integrations add generic Harbor agent-phase and task-container observations
around each framework-native trace. Harbor does not expose verifier lifecycle
through the installed-agent boundary, so evaluator lifecycle is reported as
`not_exposed`; its native logs, verifier results, and ATIF data remain
authoritative auxiliary artifacts rather than a fourth tracing adapter.

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
        └── attempt-<n>/
            ├── preflight.json
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

## Collection during benchmark execution

`--trace-dir` configures in-process capture; it does not invoke a separate log
collector. When the flag is present, the benchmark runner creates the trace run
before provider work, connects the framework-native adapter to the live agent
event stream, and finalizes the trace after the benchmark outcome is fixed.
Commands, arguments, outputs, native evidence, timestamps, and durations are
therefore captured as the corresponding agent actions occur. Nothing needs to
be run alongside the benchmark to collect them.

The supplied directory is a stable trace base. Every invocation creates a
private `trace-run-<uuid>` child. SWE runners print the exact child path in
their invocation output; Terminal-Bench runners also persist it in their
benchmark manifest. The researcher CLI can discover finalized child runs when
given the stable base, so automation may retain either path.

Tracing is opt-in because it retains detailed agent activity and can use
meaningful storage. Once enabled, a post-start recorder failure is isolated
from the agent and benchmark result and is reported through trace health.

## Researcher and recovery CLI

The `benchmark-trace` command is a version-aware interface over traces from
OpenCode, OpenHands, or Hermes and from any benchmark using the contract. It has
no collection or agent-launch command. Analysis commands are read-only;
`recover` is the sole mutation and only finalizes a stopped durable journal:

```bash
# Run from the OpenHands-benchmarks environment.
uv run benchmark-trace validate /path/to/traces
uv run benchmark-trace inspect /path/to/traces/trace-run-<uuid>
uv run benchmark-trace summarize /path/to/traces/trace-run-<uuid>
uv run benchmark-trace render /path/to/traces/trace-run-<uuid>
uv run benchmark-trace render --include-artifacts /path/to/traces/trace-run-<uuid>
uv run benchmark-trace compare \
  /path/to/opencode/trace-run-<uuid> \
  /path/to/openhands/trace-run-<uuid> \
  /path/to/hermes/trace-run-<uuid>
uv run benchmark-trace gate \
  /path/to/opencode/trace-run-<uuid> \
  /path/to/openhands/trace-run-<uuid> \
  /path/to/hermes/trace-run-<uuid>
uv run benchmark-trace recover /path/to/interrupted/attempt-1
```

Every command supports `--format json` for scripts. `validate` accepts run
directories, attempt directories, their index documents, or one or more stable
trace bases. The other commands require one resolved run or attempt;
`compare` and `gate` require at least two. Validation exits `0` for valid traces
and `1` for contract failures. `gate` exits `0` only when every producer output
is valid, healthy, complete, and configuration-comparable. Unsupported
versions, ambiguous paths, and unreadable input exit `2`.

`inspect` reports identity, revision provenance, effective execution settings,
health, and capabilities. `summarize` aggregates normalized activity, durations,
coverage, redaction, and loss counters. `compare` checks contract, benchmark,
instance-selection, attempt topology, agent configuration, and execution parity
before showing cross-trace metrics and capability differences. `render`
reconstructs a deterministic, timestamped, nested timeline and always shows
artifact references. It reads retained contents only with
`--include-artifacts`. No command calculates token usage or cost.

The CLI currently dispatches only `benchmark-trace/v1`. A future contract
version must receive an explicit reader rather than being silently interpreted
with v1 semantics.

## Terminal-Bench 2.1 bridge

Pass `--trace-dir <base-directory>` to any framework's Terminal-Bench 2.1
inference CLI. The host wrapper creates a private trace run only after local
preflight succeeds. Each Harbor installed-agent adapter reads the resolved task
identity, effective agent timeout, and Docker image from Harbor's pinned trial
configuration. A locked allocator assigns per-instance attempt ordinals safely
when multiple Harbor trials run concurrently.

OpenHands and Hermes record sanitized normalized traces inside the task's
`/logs/agent/benchmark-trace` mount and promote a completed attempt into the
private host trace root after agent execution. OpenCode emits its opt-in native
frames through the installed agent's JSONL output after applying the contract's
credential and accounting sanitizer at the source. The host adapter records
non-secret timing and provenance before the provider can run, normalizes those
frames into a private per-attempt staging area, atomically promotes the finalized
attempt into the same contract, and removes only the internal trace frames from
the ordinary OpenCode log. Harbor's original artifacts are otherwise unchanged.
The OpenCode run manifest checkpoints enough non-secret trace identity to make
this promotion idempotently recoverable after a host crash with
`bench:terminal --recover-traces-from <manifest>`, without relaunching an agent.

The run index is written only when every expected instance has the requested
number of finalized attempts. Infrastructure retries may therefore create
additional numbered trace attempts, but cannot be mistaken for a requested
semantic attempt. For non-explicit selections, task order comes from Harbor's
resolved `lock.json` rather than trial completion order. Abrupt task-container
termination can leave only a partial journal; in that case `run.json` is omitted
instead of claiming complete coverage, while the benchmark result and Harbor
retry policy remain unchanged.

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
excluded by contract. Memory remains unobserved. Terminal-Bench adds the
generic Harbor and container observations described above; evaluator lifecycle
remains outside the installed-agent boundary.

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
provider retry. Terminal-Bench preserves the timestamps carried by OpenCode's
native frames and adds the resolved Harbor task image and agent deadline.

## Hermes adapter

The Hermes implementation is a Python recorder and native callback adapter in
the Hermes Agent repository. Passing `--trace-dir <base-directory>` to its
SWE-bench Verified or SWE-bench Pro inference CLI creates a private
`trace-run-<uuid>` directory and emits `benchmark-trace/v1`. Tracing is disabled
when absent, requires a fresh benchmark run and clean exact source revision, and
initializes before coordinator construction.

The adapter uses Hermes' public `AIAgent` callbacks without replacing its native
`delegate_task` orchestration. It retains sanitized native evidence and
normalizes worker/session lifecycle, derived root model-turn boundaries,
complete root tool inputs and outputs, native tool duration measurements,
shell/file/search activity, native child lifecycle/text/tool-start events, and
completed context-compaction facts.

Hermes' parent callback exposes only a bounded child output-tail summary rather
than complete child tool results or model exchanges, and its compaction callback
reports completion without a start boundary or duration. Provider
request/response bodies are not enabled. Memory and browser activity are
disabled in these benchmark workers, while controller-owned final Git capture
and container teardown remain outside the worker adapter. The capability matrix
reports each boundary explicitly. Credentials are sanitized before persistence,
and token usage and cost accounting are removed by policy. Post-start trace
failures cannot affect the agent result or initiate a provider or benchmark
retry. Terminal-Bench uses the same native callbacks and records the resolved
Harbor task image and agent deadline without replacing Hermes delegation.
