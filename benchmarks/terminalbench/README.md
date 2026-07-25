# Terminal-Bench 2.1 Evaluation

This module provides integration with [Terminal-Bench](https://tbench.ai), a benchmark for evaluating AI agents on terminal-based tasks. The integration uses [Harbor](https://harborframework.com) as the evaluation harness with the `openhands-sdk` agent.

## Overview

Terminal-Bench evaluates how well AI agents can handle real-world, end-to-end tasks in command-line environments, including:
- Compiling code
- Training models
- Setting up servers
- System administration tasks

## Prerequisites

1. **Install Harbor**: Harbor is the official harness for running Terminal-Bench 2.1.

```bash
pip install harbor
# or
uv pip install harbor
```

2. **Docker**: Harbor requires Docker to be installed and running.

3. **LLM API Key**: Configure your LLM provider credentials.

```bash
export OPENROUTER_API_KEY=...
```

## Usage

### Running Inference

Run the Terminal-Bench evaluation using the OpenHands SDK agent:

```bash
# Run one task as a safe smoke evaluation
uv run terminalbench-infer

# Run specific tasks
uv run terminalbench-infer --task-id hello-world

# Run tasks from a file
uv run terminalbench-infer --select tasks.txt

# Run the official Terminal-Bench 2.1 dataset explicitly
uv run terminalbench-infer \
  --dataset terminal-bench/terminal-bench-2-1

# Limit the run to 5 tasks (useful for CI smoke tests)
uv run terminalbench-infer --n-limit 5

# Run the complete dataset locally without uploading it
uv run terminalbench-infer --all-tasks

# Run with multiple workers
uv run terminalbench-infer --num-workers 4

# Run the enforced official leaderboard protocol
uv run terminalbench-infer \
  --leaderboard \
  --num-workers 4

# Preview the credential-free Harbor command
uv run terminalbench-infer --dry-run

# Tracing is automatic; override its repository-local base if needed
uv run terminalbench-infer \
  --run-id traced-terminal-smoke \
  --trace-dir /path/to/traces
```

The default model is `openrouter/qwen/qwen3-coder-next`, authenticated from
`OPENROUTER_API_KEY`, with temperature `0.1` and a 24-iteration supervisor
budget passed explicitly to Harbor. The run is deliberately limited to one
task, one outer worker, and one coordinator-led attempt. Harbor first installs
the declared SDK dependencies, then overlays the exact clean vendored SDK
revision; the full commit is recorded in the manifest. Each SDK LLM call permits
one provider request attempt, including in the explicit single-agent mode.
Inside the default attempt, OpenHands uses its native task tool to delegate
investigation, execution, and independent
verification to fresh `benchmark-navigator`, `benchmark-patcher`, and
`benchmark-reviewer` subagents sequentially, with iteration caps of 10, 18,
and 12 respectively. Delegation adds model calls but
does not create additional Harbor attempts. Pass `--disable-delegation` only
for an intentional single-agent comparison. `--leaderboard`
removes all task filters, requires the official 89-task dataset, raises the run
to at least five attempts per task, and enables a public Harbor upload. Harbor's
`--max-retries` behavior is available for infrastructure failures without adding
semantic retries to agent work.

Tracing defaults to the repository-local `.benchmark-traces/` base. Use
`--trace-dir <base-directory>` to override it or `--no-trace` for an intentional
untraced run. A traced invocation creates a private `trace-run-<uuid>` only
after preflight succeeds and requires the exact clean OpenHands-benchmarks and
vendored SDK revisions. The native OpenHands conversation callback records
timestamped model, tool, shell, file, search, delegation, and session activity
inside each task container. It also imports durable child conversations before
finalization, preserving source timestamps and nesting each child beneath its
logical delegation. The Harbor adapter then promotes the sanitized attempt into
the host trace root.

The trace records Harbor's resolved task identity, effective agent timeout, and
container image. Concurrent trials receive locked per-instance attempt
ordinals, and `run.json` is written only when all requested instances and
attempts finalized. Harbor's verifier lifecycle is outside the installed-agent
boundary and is reported as `not_exposed`; native Harbor results, logs, and ATIF
trajectories remain authoritative. Trace failures after agent execution starts
do not change the benchmark result or cause a retry. Token usage and cost are
intentionally excluded from the normalized research trace.

### LLM Configuration

To override the default model, pass an LLM configuration file explicitly:

```json
{
  "model": "anthropic/claude-sonnet-4-20250514",
  "api_key": "YOUR_API_KEY"
}
```

```bash
uv run terminalbench-infer .llm_config/claude.json
```

Or use a LiteLLM proxy:

```json
{
  "model": "litellm_proxy/anthropic/claude-sonnet-4-20250514",
  "base_url": "https://your-proxy.example.com",
  "api_key": "YOUR_API_KEY"
}
```

### Evaluating Results

After running inference, evaluate the results:

```bash
uv run terminalbench-eval ./evaluation_outputs/.../output.jsonl
```

This generates a report file (`output.report.json`) with:
- Total/completed/resolved instance counts
- Success rate
- Aggregate metrics (cost, tokens)

The inference command also creates this deterministic report automatically after
Harbor completes. `terminalbench-eval` remains available to regenerate it later.

## Output Format

### Inference Output (`output.jsonl`)

Each line contains:

```json
{
  "instance_id": "task-name",
  "test_result": {
    "trajectory_path": "path/to/trajectory.json",
    "total_steps": 15,
    "final_metrics": {
      "total_prompt_tokens": 5000,
      "total_completion_tokens": 1000,
      "total_cost_usd": 0.05
    }
  },
  "instruction": "Task description...",
  "history": [...],
  "metrics": {...}
}
```

### Evaluation Report (`output.report.json`)

```json
{
  "total_instances": 100,
  "completed_instances": 95,
  "resolved_instances": 80,
  "unresolved_instances": 15,
  "error_instances": 5,
  "aggregate_metrics": {
    "total_cost_usd": 5.25,
    "total_prompt_tokens": 500000,
    "total_completion_tokens": 100000
  }
}
```

## Architecture

The integration follows the Harbor agent adapter pattern:

1. **Harbor Harness**: Manages task containers and lifecycle
2. **OpenHands SDK Supervisor**: Owns the final outcome inside each container
3. **Native OpenHands Subagents**: Investigate, execute, and verify sequentially
4. **ATIF Trajectories**: Results stored in Agent Trajectory Interchange Format

```
┌──────────────────────────────────────────────────┐
│                 Harbor Harness                   │
│  ┌────────────────────────────────────────────┐  │
│  │           Task Container                   │  │
│  │  ┌──────────────────────────────────────┐  │  │
│  │  │    OpenHands SDK Supervisor          │  │  │
│  │  │  - Terminal tool                     │  │  │
│  │  │  - File editor tool                  │  │  │
│  │  │  - Task tracker tool                 │  │  │
│  │  │  - Blocking task/subagent tool       │  │  │
│  │  └──────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

## Official Submissions

Use `--leaderboard` for official Terminal-Bench 2.1 submissions. The preset
enforces the complete dataset, at least five attempts per task, and a public
Harbor upload. It rejects task filters, task limits, nonofficial datasets, and an
explicit attempt count below five.

## Reproducibility And Artifacts

Each run pins the installed `openhands-sdk` version and records a credential-free
Harbor command in `manifest.json`. The manifest also captures the dataset, model,
Harbor version, environment, attempts, concurrency, infrastructure retries,
agent topology, timestamps, artifact paths, and final status. Parent and
subagent token usage and cost are aggregated, while durable subagent
conversations are retained beneath the agent conversation logs. Harbor stdout and stderr are
streamed live and retained as separate log files.

The native Harbor job directory remains authoritative and includes verifier
results, agent logs, and ATIF trajectories. `output.jsonl` and
`output.report.json` are auxiliary OpenHands-format views of those results.
Credentials are inherited through the Harbor process environment and are not
written to command arguments, manifests, or metadata.

By default, each run is stored under
`evaluation_outputs/terminal-bench-2.1/runs/<run-id>/`. A dry run only prints
the resolved command and does not create this directory.

## References

- [Terminal-Bench](https://tbench.ai) - The benchmark
- [Terminal-Bench 2.1 dataset](https://github.com/harbor-framework/terminal-bench-2-1) - Official dataset and submission protocol
- [Harbor](https://harborframework.com) - The evaluation harness
- [OpenHands SDK](https://github.com/OpenHands/software-agent-sdk) - The agent SDK
- [ATIF Specification](https://github.com/laude-institute/harbor/blob/main/docs/rfcs/0001-trajectory-format.md) - Trajectory format
