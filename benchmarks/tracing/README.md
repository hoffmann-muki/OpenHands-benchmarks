# Benchmark tracing

`benchmarks.tracing` defines a framework-neutral, research-grade trace contract
for agent benchmark execution. OpenCode, OpenHands, and Hermes use
framework-native instrumentation adapters while emitting the same normalized
format.

The integration wires the three framework-native adapters into SWE-bench
Verified, SWE-bench Pro, and Terminal-Bench 2.1. Inference tracing defaults to
each harness repository's `.benchmark-traces/` directory, does not alter agent
prompts or native delegation, and does not patch Harbor.
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
3. A harness adapter supplies only harness-owned topology and externally
   observable attempt metadata.
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

The normalized trace records observable activity: three generic lifecycle
envelopes, agent sessions, atomic model/provider exchanges where exposed, tool
calls and results, commands, file/search/browser activity, delegation, context
compaction, memory activity, patch extraction, and evaluation lifecycle.
Framework-native evidence is retained alongside normalized events after
applying the same safety policy.

Startup and shutdown are intentionally coarse. Detailed capture begins when the
coding agent is entered and ends when its conversation returns; the subsystem
does not encode every benchmark-specific setup, evaluator, or teardown
operation. The three envelopes still account for the whole attempt, while
research views identify any intervals inside agent execution that lack a more
specific observable activity.

Harbor is an outer execution harness, not an agent framework. Terminal-Bench
integrations retain Harbor provenance and task-container identity within the
same generic startup/execution/shutdown envelopes used by direct runners; they
do not add a second Harbor-specific lifecycle. Harbor does not expose verifier
lifecycle through the installed-agent boundary, so evaluator lifecycle is
reported as `not_exposed`; its native logs, verifier results, and ATIF data
remain authoritative auxiliary artifacts rather than a fourth tracing adapter.

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

`benchmark-trace/v1` is the active schema label. `1.2.0` is the contract release.
Every trace also records the deterministic SHA-256 digest of the complete schema
bundle. Only the installed digest is accepted; the label alone does not grant
compatibility with an earlier physical representation.

Release 1.2 adds the mandatory deterministic `execution-tree.json` projection.
It pairs start/end boundaries into activities, nests them by span ownership,
marks overlapping sibling activities, and accounts for every canonical event.
Release 1.1 made the generic startup/execution/shutdown envelopes mandatory,
kept source and capture clocks distinct, and added concurrency-preserving
ordering and execution-attribution analysis. Earlier physical trace layouts
are not accepted by the installed reader.

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
            ├── execution-tree.json
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
On filesystems with hard-link support, a finalized trace keeps both required v1
paths as directory entries backed by the same inode; unsupported filesystems
fall back to an atomic copy.

`execution-tree.json` is generated automatically from the finalized
`events.jsonl`; it is never assembled independently during agent execution.
The projection pairs span boundaries, keeps sequential siblings in source-time
order, marks overlapping siblings with concurrency groups and cross-links, and
retains input/output payloads plus artifact references. Its source digest and
event-accounting totals make stale or incomplete projections detectable.
`events.jsonl` remains the canonical evidence and the tree can always be
regenerated from it.

Native evidence is appended during execution to one durable internal journal.
Finalization writes its sanitized payloads into size-bounded deterministic gzip
chunks and rewrites `native/index.jsonl` as the ordinary per-record v1 index.
Each index row keeps its sequence, timestamps, source, identity, and normalized
event links, while many rows may reference the same immutable chunk artifact.
This preserves every native delta and state update without creating one
filesystem object per streaming event. This chunk representation is mandatory;
one-artifact-per-record native indexes are rejected. Python consumers use
`read_native_content(attempt_dir, index_record)` to resolve a record from its
required chunk.

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

Benchmark inference configures in-process capture; it does not invoke a separate
log collector. The runner creates the trace run beneath the repository-local
`.benchmark-traces/` base before provider work, connects the framework-native
adapter to the live agent event stream, and finalizes the trace after the
benchmark outcome is fixed.
Commands, arguments, outputs, native evidence, timestamps, and durations are
therefore captured as the corresponding agent actions occur. Nothing needs to
be run alongside the benchmark to collect them.

Use `--trace-dir <base-directory>` to override the harness-local base or
`--no-trace` for an intentional untraced run. Every traced invocation creates a
private `trace-run-<uuid>` child. SWE runners print the exact child path in their
invocation output; Terminal-Bench runners also persist it in their benchmark
manifest. The researcher CLI can discover finalized child runs when given the
stable base, so automation may retain either path.

Tracing retains detailed agent activity and can use meaningful storage. A
post-start recorder failure is isolated from the agent and benchmark result and
is reported through trace health.

The researcher summary reports logical native-record count separately from
physical native-artifact and chunk counts. Repeated native state snapshots and
transport heartbeats remain in the lossless chunks for exact chronology; views
may aggregate them without modifying retained evidence. Canonical event
artifacts remain individually content-addressed because they are already
deduplicated and benefit from direct random access.

## AgentSight companion profiles

Every traced SWE-bench Verified, SWE-bench Pro, and Terminal-Bench 2.1 attempt
starts AgentSight automatically. The generic profiling adapter knows only the
runtime topology:

- a host scope follows the OpenHands evaluation worker and its descendants,
  including host-side orchestration activity; and
- a task-container scope uses a privileged, network-isolated sidecar filtered
  by the container's PID namespace, including later `docker exec` processes.

SWE-bench uses both scopes because OpenHands orchestration is host-side while
the agent-server, provider client, and repository tools run in the task
container. Harbor installs the complete Terminal-Bench agent inside its `main`
task container, so
Terminal-Bench uses only one task-container sidecar. It does not start a
redundant host collector or require sudo.

This is independent systems evidence, not a replacement for the OpenHands
semantic adapter. It does not change prompts, delegation, provider attempts,
timeouts, or benchmark retries. For both SWE-bench and Terminal-Bench, the
adapter asks the exact OpenHands Python runtime in the task container which
TLS-bearing binary it uses—its loaded `libssl`, or the Python executable when
OpenSSL is statically embedded—and exposes that binary through
`/proc/<container-init>/root`, and filters TLS events to the task PID namespace.
This captures plaintext TLS/HTTP evidence without a proxy, a global TLS probe,
or another collector. If discovery fails, best-effort mode retains the
process/filesystem/network profile and reports `containerTls.active=false`;
strict mode aborts before inference. The sidecar disables stdio because
namespace-wide stdio capture is not available.

Each attempt stores its aggregate profile under `profiles/agentsight/`.
`profile.json`, `health.json`, and `summary.json` describe cross-scope status.
Each `sources/<scope>/` directory contains AgentSight provenance and health,
`capture.db`, and the compressed `system-events.jsonl.zst` evidence journal.
The supervisor accepts readiness only when its schema, profile ID, and scope ID
match the current attempt, and stops all applicable collectors before workspace
teardown.
For Terminal-Bench it stops the single sidecar before Harbor teardown, stages
the completed profile in the trial log, and co-locates it with the semantic
attempt during promotion. The aggregate correlation record names the run,
benchmark, framework, instance, and attempt.

AgentSight can visualize the host and task-container databases as one
scope-aware timeline without copying or rewriting them:

```bash
agentsight report --profile-dir <attempt>/profiles/agentsight serve
```

The loader validates source profile identities, merges by normalized wall-clock
time, and preserves `scope_id` on every row. Equal numeric PIDs and row IDs in
different scopes remain distinct, concurrent activity remains concurrent, and
the UI provides scope lanes and filtering. The same command works for
single-scope Terminal-Bench profiles.

Profiling is best-effort by default and never changes the benchmark result or
retry policy. Set `BENCHMARK_AGENTSIGHT_STRICT=1` to require every expected
scope before provider work, or `BENCHMARK_AGENTSIGHT=off` to disable it.
`AGENTSIGHT_BIN` and `AGENTSIGHT_IMAGE` select the host executable and sidecar
image; `AGENTSIGHT_READY_TIMEOUT_SECONDS` and
`AGENTSIGHT_STOP_TIMEOUT_SECONDS` configure supervision. Before an unprivileged
host collector starts, the generic tracing privilege helper now performs the
equivalent of `sudo -v` automatically. It reads the password from
`~/.config/benchmark-tools/sudo-password` by default; use
`BENCHMARK_SUDO_PASSWORD_FILE` to select another private regular file and
`BENCHMARK_SUDO_TIMEOUT_SECONDS` to configure validation timeout. The file must
belong to the benchmark user and grant no group or other permissions. Its
contents are supplied only on sudo's standard input and never enter command
arguments, environment variables, logs, profiles, or Git. Password-file
authorization occurs before evidence capture becomes ready. A small privileged
supervisor then owns the collector and observes a private stop marker, so
shutdown needs neither another sudo call nor the password and remains reliable
even after the host's sudo ticket expires. If the file is absent, an
already-valid sudo ticket is accepted non-interactively.

Parsed authentication headers are removed before AgentSight persistence, but
profiles can still contain prompts, responses, commands, output, paths, and
network targets. Treat the trace attempt as sensitive. `events.jsonl` remains
the canonical framework trace; AgentSight provides correlated OS-level
evidence without forcing concurrent activity into a serial order.

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
uv run benchmark-trace render --order capture /path/to/traces/trace-run-<uuid>
uv run benchmark-trace render --order sequence /path/to/traces/trace-run-<uuid>
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
native-source and physical-chunk counts, coverage, redaction, and loss counters.
It also reports detailed execution coverage, explicit unattributed gaps,
concurrency, source/capture inversions, capture delay, and observed lanes.
`compare` checks contract, benchmark,
instance-selection, attempt topology, agent configuration, and execution parity
before showing cross-trace metrics and capability differences. `render`
reconstructs a deterministic, timestamped, nested timeline and always shows
artifact references. Source-time order is the default; `--order capture`
reveals adapter/transport arrival order and `--order sequence` preserves durable
journal order. All views retain concurrent spans and agent/session lanes rather
than claiming a globally serial execution. Artifact contents are read only with
`--include-artifacts`. No command calculates token usage or cost.

The CLI currently dispatches only `benchmark-trace/v1`. A future contract
version must receive an explicit reader rather than being silently interpreted
with v1 semantics.

## Terminal-Bench 2.1 bridge

Each framework's Terminal-Bench 2.1 inference CLI defaults to its harness
repository's `.benchmark-traces/` base. The host wrapper creates a private trace
run only after local preflight succeeds. Each Harbor installed-agent adapter
reads the resolved task identity, effective agent timeout, and Docker image from
Harbor's pinned trial configuration. A locked allocator assigns per-instance
attempt ordinals safely when multiple Harbor trials run concurrently.

OpenHands and Hermes record sanitized normalized traces inside the task's
`/logs/agent/benchmark-trace` mount and promote a completed attempt into the
private host trace root after agent execution. OpenCode emits its native
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

The SWE-bench Verified and SWE-bench Pro inference CLIs default to the
OpenHands-benchmarks repository's `.benchmark-traces/` base. Each invocation
creates a private `trace-run-<uuid>` directory beneath the selected base so a
later invocation cannot overwrite an earlier run.

The adapter consumes the native synchronous `Conversation` callback stream. It
retains sanitized native evidence and normalizes observable model responses,
tool inputs and complete outputs, shell/file/search/browser activity,
delegation boundaries, ACP tool activity, messages, hooks, state updates,
errors, pauses, and condenser boundaries. Native action/observation timestamps
are paired to derive tool durations, while observable agent-loop boundaries
form atomic model-turn spans. Trace initialization occurs before workspace or
provider work; detailed execution begins immediately before the conversation
and ends when it returns; trace finalization occurs only after the benchmark
outcome is fixed.

Provider completion logs are not enabled by the benchmark integration because
the SDK's separate built-in completion logger writes outside this recorder's
pre-persistence safety boundary. If an already-enabled native completion-log
event reaches the adapter, its structured credentials and token/cost accounting
are removed before retention.

The parent remote conversation exposes a delegated task's boundary and result
but does not forward the internal subagent stream live. Before trace
finalization, SWE-bench automatically imports the agent-server's durable child
conversation archive and Terminal-Bench imports the equivalent local
persistence directory. Child sessions, model turns, tool inputs and outputs,
and their source timestamps are normalized beneath the matching logical
delegation. Because this evidence arrives after the child finishes,
`occurred_at` preserves native source time while `recorded_at` records archival
ingestion time. Missing, incomplete, or unlinked child evidence degrades the
trace and keeps delegation coverage `partial`; a complete import reports
`full`. Token events and token/cost accounting are intentionally excluded by
contract. Memory remains unobserved. Terminal-Bench retains Harbor provenance
and task-container identity inside the generic phase envelopes; evaluator
lifecycle remains outside the installed-agent boundary.

The evaluator's outer timeout cancels an asyncio task but cannot terminate its
already-running worker thread. The default native inference deadline is ten
minutes shorter, so it normally finalizes first with `timeout` status. If a
worker ignores both interruption and the native deadline, `run.json` is omitted
rather than claiming a complete run; the durable attempt journal remains
available for explicit recorder recovery after the process stops.

## OpenCode adapter

The OpenCode implementation is a TypeScript recorder and adapter in the
OpenCode repository. Its SWE-bench Verified and SWE-bench Pro inference CLIs
create private `trace-run-<uuid>` directories beneath the repository-local
`.benchmark-traces/` base and emit the same `benchmark-trace/v1` contract.
Tracing requires a fresh benchmark run and is initialized before container
setup or any provider request.

An internal trace-enabled CLI stream publishes OpenCode's native event bus with a
strict per-process sequence. The adapter retains sanitized native evidence and
normalizes root and child sessions, assistant-message model boundaries,
pending/running/final tool state, complete inputs and outputs, shell/file/search
and browser activity, native task delegation, compaction boundaries, errors,
and native wall-clock durations. It observes all subagent session events on the
same native event stream without replacing or constraining OpenCode's task
delegation behavior.

Native frame timestamps remain `occurred_at`; recorder arrival remains
`recorded_at`. This distinction is especially important for Terminal-Bench,
where frames cross the installed-agent/host bridge before finalization.

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
the Hermes Agent repository. Its SWE-bench Verified and SWE-bench Pro inference
CLIs create private `trace-run-<uuid>` directories beneath the repository-local
`.benchmark-traces/` base and emit `benchmark-trace/v1`. Tracing requires a
fresh benchmark run and clean exact source revision, and initializes before
coordinator construction.

The adapter uses Hermes' public `AIAgent` callbacks without replacing its native
`delegate_task` orchestration. It retains sanitized native evidence and
normalizes worker/session lifecycle, derived root model-turn boundaries,
complete root tool inputs and outputs, native tool duration measurements,
shell/file/search activity, and completed context-compaction facts. Hermes'
native child step and progress hooks additionally relay child model turns as
atomic spans and correlate child tool inputs with complete sanitized results and
native durations.

Child tool arguments are the display-safe values supplied by Hermes, and exact
child/provider request and response bodies are not exposed. The compaction
callback reports completion without a start boundary or duration. Memory and
browser activity are disabled in these benchmark workers, while
controller-owned final Git capture and container teardown remain outside the
worker adapter. The capability matrix reports each boundary explicitly.
Credentials are sanitized before persistence, and token usage and cost
accounting are removed by policy. Post-start trace failures cannot affect the
agent result or initiate a provider or benchmark retry. Terminal-Bench uses the
same native callbacks and records the resolved Harbor task image and agent
deadline without replacing Hermes delegation.
