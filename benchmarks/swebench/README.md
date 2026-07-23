# SWE-Bench Benchmark Evaluation

This directory contains the implementation for running SWE-Bench evaluation using OpenHands agents.

## Overview

SWE-Bench is a benchmark for evaluating AI agents on real-world software engineering tasks derived from GitHub issues. The benchmark tests an agent's ability to understand problem statements, navigate codebases, and generate patches that resolve issues.

## Dataset

- **Source**: Princeton NLP
- **Datasets**: 
  - `princeton-nlp/SWE-bench` - Full dataset
  - `princeton-nlp/SWE-bench_Lite` - Smaller curated subset
  - `princeton-nlp/SWE-bench_Verified` - Verified instances
- **Splits**: `test`, `dev`

## Usage

### Docker Workspace (Local Evaluation)

#### Step 1: Build Docker Images

Before running inference, you need to build Docker images for the SWE-Bench instances. Each instance requires a specific environment setup based on the repository and issue.

```bash
uv run python -m benchmarks.swebench.build_images \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal \
  --max-workers 1
```

The builder defaults to the same tracked smoke instance used by inference:
`scikit-learn__scikit-learn-13439`.

#### Step 2: Run Inference

Run evaluation using the built Docker images:

```bash
export OPENROUTER_API_KEY=...
uv run swebench-infer
```

Local Docker is the inference default. The runner and phased builder both use
`ghcr.io/openhands/eval-agent-server:{SDK_SHA7}-{DOCKERFILE_HASH7}-{INSTANCE_TAG}-source-minimal`,
so a locally prebuilt image is selected without an image-tag override. Pass
`--workspace remote` or `--workspace apptainer` only when intentionally using
those backends.

The image tag is derived from the full clean vendored SDK revision, which is
also recorded in evaluation metadata. The default prompt is the same concise
public issue contract used by the peer runners. Each SDK LLM call permits one
provider request attempt; a provider failure is not retried inside the turn.

The safe defaults select `scikit-learn__scikit-learn-13439` and use
`openrouter/qwen/qwen3-coder-next`, one 24-iteration coordinator run, a shared
30-minute inference deadline, one inference worker, and one coordinator-led
attempt. `n_critic_runs` is one and exception retries are disabled. Within that
attempt, the supervisor delegates investigation, implementation, and independent
review sequentially to fresh native OpenHands subagents configured for 10, 18,
and 12 iterations respectively. Delegation uses the SDK's unmodified native task
tool. Freshness, ordering, and avoiding task resumption are coordinator instructions
rather than benchmark-side interception. These delegations add model calls but
are not benchmark retries. Use `--select '' --n-limit 0` only for a deliberate
full-dataset run, and raise `--num-workers` explicitly when concurrent inference
is intended. Additional attempts require explicit `--n-critic-runs` or
`--max-retries` overrides. Pass `--disable-delegation` only for an intentional
single-agent comparison.

You can resume a previous run by re-running the same command with the same `--output-dir`. Previously completed instances are automatically skipped.

**Selecting specific instances:**

You can run evaluation on a specific subset by creating a text file with instance IDs:

```bash
# Create instances.txt with one instance ID per line
echo "django__django-11333" > instances.txt
echo "astropy__astropy-12345" >> instances.txt

# Run with selection
uv run swebench-infer path/to/llm_config.json \
    --select instances.txt \
    --n-limit 0 \
    --workspace docker
```

An explicit LLM config remains supported and overrides the default OpenRouter
model. Pass `--select ''` to clear the tracked smoke selection.

### Remote Workspace (Scalable Cloud Evaluation)

Remote workspace enables running evaluations at scale by using a cloud-based runtime API to provision containers. This is ideal for large-scale benchmark runs with high parallelization.

#### Step 1: Pre-build and Push Images

Images must be pre-built and pushed to a **public** container registry before running remote evaluations.

**Option A: Automated Build via PR Label (Recommended)**

1. Create or update a PR in this repository
2. Add one of the following labels to the PR to trigger image builds:
   - `build-swebench-50`: Build 50 images (quick testing, ~5-10 minutes)
   - `build-swebench-200`: Build 200 images (medium testing, ~20-40 minutes)
   - `build-swebench`: Build all images (full evaluation, ~1-2 hours)
3. The GitHub Action will automatically:
   - Build agent-server images for instances in `princeton-nlp/SWE-bench_Verified` (test split)
   - Push images to `ghcr.io/openhands/eval-agent-server` with tags like:
     ```
     ghcr.io/openhands/eval-agent-server:{SDK_SHA7}-{DOCKERFILE_HASH7}-{INSTANCE_TAG}-source-minimal
     ```
   - The docutils/roman layer is applied in-place (no suffix) for allowlisted repos that need it (currently `sphinx-doc`)
   - Post a comment on [issue #81](https://github.com/OpenHands/benchmarks/issues/81) with the build results

**Option B: Manual Build**

```bash
uv run python -m benchmarks.swebench.build_images \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal \
  --select '' \
  --push \
  --max-workers 32
```

**Important Notes:**
- Images must be **publicly accessible** for the remote runtime to pull them
- The SDK SHA is automatically detected from the `vendor/software-agent-sdk` submodule
- Each SWE-Bench instance gets its own unique image tag based on the repository and issue

#### Step 2: Set Up Environment Variables

```bash
# Required: Your runtime API key
export RUNTIME_API_KEY="your-runtime-api-key-here"

# Optional: Override default runtime API URL
export RUNTIME_API_URL="https://runtime.eval.all-hands.dev"

# Optional: Override the complete image-tag prefix for a prebuilt image set
# (defaults to SDK SHA7 + SDK Dockerfile hash7 from the vendored submodule)
export IMAGE_TAG_PREFIX="abc1234-def5678"
```

#### Step 3: Run Inference with Remote Workspace

Run evaluation using the remote workspace with high parallelization:

```bash
uv run swebench-infer .llm_config/sonnet-4-5.json \
    --dataset princeton-nlp/SWE-bench_Verified \
    --split test \
    --workspace remote \
    --num-workers 32 \
    --max-iterations 24 \
    --inference-timeout 1800 \
    --select '' \
    --n-limit 200
```

**Command Options Explained:**
- `--workspace remote`: Use remote runtime instead of local Docker
- `--num-workers 32`: Run 32 instances in parallel (adjust based on your quota)
- `--max-iterations 24`: Maximum coordinator iterations per run
- `--inference-timeout 1800`: Shared agent deadline in seconds
- `--n-limit 200`: Limit to first 200 instances (optional, for testing)

**Example: Full-scale Evaluation**

```bash
# Run on all instances with maximum parallelization
uv run swebench-infer .llm_config/sonnet-4-5.json \
    --workspace remote \
    --num-workers 64 \
    --max-iterations 24 \
    --inference-timeout 1800 \
    --select '' \
    --n-limit 0
```

**Example: Subset Evaluation**

```bash
# Test on a small subset first
echo "django__django-11333" > test_instances.txt
echo "django__django-12155" >> test_instances.txt

uv run swebench-infer .llm_config/sonnet-4-5.json \
    --select test_instances.txt \
    --n-limit 0 \
    --workspace remote \
    --num-workers 2 \
    --max-iterations 24 \
    --inference-timeout 1800
```

#### Troubleshooting Remote Workspace

**Error: "RUNTIME_API_KEY environment variable is not set"**
- Solution: Export the `RUNTIME_API_KEY` environment variable before running

**Error: "Agent server image ... does not exist in container registry"**
- Solution: Ensure images are pre-built and pushed using Step 1
- Verify the SDK SHA matches between your local submodule and the built images
- Check that images are publicly accessible in the registry

**Error: "Connection timeout" or API errors**
- Solution: Check your network connectivity
- Verify the `RUNTIME_API_URL` is correct
- Ensure your API key has sufficient quota for the number of workers

### Comparing Docker vs Remote Workspace

| Aspect | Docker Workspace | Remote Workspace |
|--------|-----------------|------------------|
| **Setup** | Simple, no prerequisites | Requires pre-built images + API key |
| **Scale** | Limited by local resources | Hundreds of parallel workers |
| **Speed** | Slower for large evaluations | Much faster with parallelization |
| **Cost** | Local compute only | API usage costs |
| **Use Case** | Development, testing | Production benchmarks, research |

### Apptainer Workspace for HPC Clusters

#### Option 1: Pre-build and push images using a separate machine with Docker support

```bash
uv run python -m benchmarks.swebench.build_images \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal \
  --select '' \
  --push
```

The wrapper layer (`docutils<0.21`, `roman`) is applied in-place for allowlisted repos during this build pipeline (currently `sphinx-doc`).

#### Option 2: Build local Apptainer SIFs on the HPC machine

If a pre-built agent-server image is missing from the registry, Apptainer mode
falls back to building a local SIF from the official SWE-Bench image and the
checked-out OpenHands SDK submodule. This does not require a Docker daemon.

```bash
export OPENHANDS_APPTAINER_BUILD_ROOT=/scratch/$USER/swebench-apptainer-agent-images
```

Set `OPENHANDS_APPTAINER_FORCE_BUILD=1` to rebuild a local SIF even when a
matching registry image exists.

#### Run on HPC with Apptainer

**Optionally**, you can override the default location where Apptainer cache is saved using the below environment variables:

```bash
export APPTAINER_CACHEDIR=<desired path to directory> # ensure that this directory exists
export APPTAINER_TMPDIR=<desired path to directory> # ensure that this directory exists
```

```bash
uv run swebench-infer path/to/llm_config.json \
    --dataset princeton-nlp/SWE-bench_Verified \
    --split test \
    --workspace apptainer
```

In `apptainer` mode, SWE-Bench first tries to use pre-built registry images. If
the expected registry tag is unavailable, it builds a local Apptainer SIF
instead.

## Evaluation

After running inference (with either workspace type), evaluate the generated patches using the official SWE-Bench evaluation:

Evaluation uses local Docker with one worker by default. Pass `--modal` only
when intentionally opting into Modal.

**Basic evaluation:**

```bash
uv run swebench-eval output.jsonl --run-id my_eval
```

**Advanced options:**

```bash
# Specify custom dataset and output file
uv run swebench-eval output.jsonl \
  --dataset princeton-nlp/SWE-bench_Lite \
  --output-file results.swebench.jsonl

# Only convert format without running evaluation
uv run swebench-eval output.jsonl --skip-evaluation
```

**Local Apptainer evaluation:**

```bash
uv run swebench-eval output.jsonl \
  --run-id my_eval \
  --apptainer \
  --apptainer-sandbox-root ~/.cache/openhands/swebench-apptainer
```

The Apptainer evaluator pulls the official SWE-bench instance images, converts
them to reusable writable sandboxes, applies each model patch, runs the
SWE-bench eval script, and grades the resulting test log locally. This is useful
on hosts where Docker is unavailable and Modal is not configured. Apptainer
evaluation currently runs sequentially; `--workers` is accepted for CLI
compatibility but ignored.

The evaluation script will:
1. Convert OpenHands output format to SWE-Bench prediction format
2. Run the official SWE-Bench evaluation harness, or local Apptainer evaluation
   when `--apptainer` is used, unless `--skip-evaluation` is set
3. Report pass/fail results for each instance

## References

- [SWE-Bench Paper](https://arxiv.org/abs/2310.06770)
- [SWE-Bench GitHub](https://github.com/princeton-nlp/SWE-bench)
- [SWE-Bench Leaderboard](https://www.swebench.com/)
