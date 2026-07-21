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

## Usage

### Running Inference

Run the Terminal-Bench evaluation using the OpenHands SDK agent:

```bash
# Run one task as a safe smoke evaluation
uv run terminalbench-infer .llm_config/claude.json

# Run specific tasks
uv run terminalbench-infer .llm_config/claude.json --task-id hello-world

# Run tasks from a file
uv run terminalbench-infer .llm_config/claude.json --select tasks.txt

# Run the official Terminal-Bench 2.1 dataset explicitly
uv run terminalbench-infer .llm_config/claude.json \
  --dataset terminal-bench/terminal-bench-2-1

# Limit the run to 5 tasks (useful for CI smoke tests)
uv run terminalbench-infer .llm_config/claude.json --n-limit 5

# Run the complete dataset locally without uploading it
uv run terminalbench-infer .llm_config/claude.json --all-tasks

# Run with multiple workers
uv run terminalbench-infer .llm_config/claude.json --num-workers 4

# Run the enforced official leaderboard protocol
uv run terminalbench-infer .llm_config/claude.json \
  --leaderboard \
  --num-workers 4

# Preview the credential-free Harbor command
uv run terminalbench-infer .llm_config/claude.json --dry-run
```

The default is deliberately limited to one task, one outer worker, and one
coordinator-led attempt. Inside that attempt, OpenHands uses its native task
tool to delegate investigation, execution, and independent verification to
fresh subagents sequentially. Delegation adds model calls but does not create
additional Harbor attempts. Pass `--disable-delegation` only for an intentional
single-agent comparison. `--leaderboard`
removes all task filters, requires the official 89-task dataset, raises the run
to at least five attempts per task, and enables a public Harbor upload. Harbor's
`--max-retries` behavior is available for infrastructure failures without adding
semantic retries to agent work.

### LLM Configuration

Create an LLM configuration file (e.g., `.llm_config/claude.json`):

```json
{
  "model": "anthropic/claude-sonnet-4-20250514",
  "api_key": "YOUR_API_KEY"
}
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
