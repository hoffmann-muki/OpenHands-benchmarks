# SWE-Bench Pro Benchmark Evaluation

This directory contains the OpenHands benchmark integration for [SWE-Bench Pro](https://scale.com/leaderboard/swe_bench_pro_public), a public long-horizon software engineering benchmark released by Scale AI.

## Dataset

- **Dataset**: `ScaleAI/SWE-bench_Pro`, pinned to revision
  `7ab5114912baf22bb098818e604c02fe7ad2c11f`
- **Split**: `test`
- **Official harness**: [`scaleapi/SWE-bench_Pro-os`](https://github.com/scaleapi/SWE-bench_Pro-os)
- **Official images**: `jefzda/sweap-images:<dockerhub_tag>`

## Running Inference

SWE-Bench Pro reuses the phased image-build pipeline from `benchmarks/swebench/`, but resolves official base images from each dataset row's `dockerhub_tag` field.

### Build agent-server images

```bash
uv run python -m benchmarks.swebenchpro.build_images \
  --dataset ScaleAI/SWE-bench_Pro \
  --split test \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal \
  --max-workers 1
```

The builder defaults to the same tracked smoke instance used by inference:
`instance_qutebrowser__qutebrowser-5fdc83e5da6222fe61163395baaad7ae57fa2cb4-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d`.

### Run inference

```bash
export OPENROUTER_API_KEY=...
uv run swebenchpro-infer
```

Local Docker is the inference default. The runner selects the same
`ghcr.io/openhands/eval-agent-server:{SDK_SHA7}-{DOCKERFILE_HASH7}-{TASK_TAG}-source-minimal`
tag produced by the phased builder, so no image or version override is needed.
Pass `--workspace remote` or `--workspace apptainer` only when intentionally
using those backends.

The tag is derived from the full clean vendored SDK revision, which is recorded
in evaluation metadata. Each SDK LLM call permits one provider request attempt;
a provider failure is not retried inside the turn.
For cross-framework subset parity, an implicit `--n-limit N` takes the first
`N` dataset rows. Explicit `--select` files preserve their declared ID order,
reject duplicates, and take precedence over `--n-limit`.

The safe defaults select the tracked qutebrowser instance above and use
`openrouter/qwen/qwen3-coder-next`, one 24-iteration coordinator run, a shared
30-minute inference deadline, one inference worker, and one coordinator-led
attempt. `n_critic_runs` is one and exception retries are disabled. Within that
attempt, the supervisor delegates investigation, implementation, and independent
review sequentially to fresh native OpenHands subagents configured for 10, 18,
and 12 iterations respectively. Delegation uses the SDK's unmodified native task
tool. Freshness, ordering, and avoiding task resumption are coordinator instructions
rather than benchmark-side interception. These delegations add model calls but
are not benchmark retries. Use an explicit `--n-limit 0` only for a deliberate
full-dataset run; any explicitly supplied limit clears the implicit smoke
selection. Raise `--num-workers` explicitly when concurrent inference is
intended. Additional attempts require explicit `--n-critic-runs` or
`--max-retries` overrides. Pass `--disable-delegation` only for an intentional
single-agent comparison.

Use the dedicated command for reproducible single-agent inference:

```bash
uv run swebenchpro-single-infer
```

It constructs the native OpenHands default agent without the task/delegation
tool and fixes the official Pro dataset and test split. The defaults otherwise
match the multi-agent runner except for selecting
`openrouter/poolside/laguna-s-2.1:free`: temperature `0.1`, one worker, one
benchmark attempt, zero exception retries, one provider attempt, a 24-iteration
agent budget, and a 30-minute deadline. The lone agent is
explicitly responsible for investigation, implementation, focused verification,
and final-diff review. Semantic tracing and AgentSight profiling remain enabled
and record the topology as `single-agent`.

For cross-framework parity, the concise default prompt renders the public Pro
`problem_statement`, `requirements`, `interface`, and `repo_language` fields.
Evaluator-only gold and test-patch fields are not rendered into the agent
prompt.

An explicit LLM configuration path remains supported and overrides the default
OpenRouter model. An explicit `--n-limit` clears the tracked smoke selection
unless `--select` is also supplied.

Remote and apptainer workspaces use the same image tags produced by the phased build pipeline.

## Running Evaluation

The evaluation wrapper converts OpenHands `output.jsonl` files into the official SWE-Bench Pro patch format, downloads a pinned checkout of the official harness on first use, materializes the required dataset rows as JSONL, and then invokes the upstream evaluation script.

```bash
uv run swebenchpro-eval path/to/output.jsonl \
  --dataset ScaleAI/SWE-bench_Pro \
  --split test
```

Evaluation uses local Docker with one worker by default, so no backend flag is
required.
Use `--no-use-local-docker` only when intentionally opting into Modal.

Helpful options:

- `--skip-evaluation`: only write the converted patch file.
- `--official-harness-dir <path>`: use an existing local checkout of `scaleapi/SWE-bench_Pro-os` instead of downloading the pinned archive.
- `--use-local-docker`: explicitly select the default local Docker backend.
- `--no-use-local-docker`: opt into Modal instead of local Docker.
- `--block-network`: disable network access inside evaluation containers.

The script writes:

- `output.swebenchpro.json`: converted patches in the upstream JSON format.
- `output.report.json`: OpenHands-style evaluation summary with `resolved_ids` for downstream tooling.
- `cost_report.jsonl`: aggregated usage and proxy-cost summary.
