#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR
readonly OPENHANDS_REPO=${OPENHANDS_BENCHMARKS_REPO:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}
readonly DEVELOPMENT_ROOT=${BENCHMARK_WORKSPACE_ROOT:-$(dirname "$OPENHANDS_REPO")}
readonly MANIFEST="$SCRIPT_DIR/manifest.json"
EXPERIMENT_ID=$(jq -r '.experiment_id' "$MANIFEST")
readonly EXPERIMENT_ID
readonly STATE_ROOT="$SCRIPT_DIR"
readonly KEY_FILE=${OPENROUTER_KEY_FILE:-$DEVELOPMENT_ROOT/openrouter-key}
readonly EVALUATOR_PYTHON=${SWEBENCH_PYTHON:-$OPENHANDS_REPO/.venv/bin/python}
readonly HARBOR_BIN=${HARBOR_BIN:-harbor}
MODEL=$(jq -r '.configuration.model' "$MANIFEST")
readonly MODEL
EXPECTED_HARBOR_VERSION=$(jq -r '.configuration.harbor_version' "$MANIFEST")
readonly EXPECTED_HARBOR_VERSION
readonly LEDGER="$STATE_ROOT/ledger.jsonl"
readonly SOURCE_LOCK="$STATE_ROOT/source-lock.json"
readonly MINIMUM_FREE_BYTES=$((30 * 1024 * 1024 * 1024))
REPETITIONS=$(jq -r '.design.repetitions_per_instance' "$MANIFEST")
readonly REPETITIONS
INSTANCES_PER_BENCHMARK=$(jq -r '.design.instances_per_benchmark' "$MANIFEST")
readonly INSTANCES_PER_BENCHMARK

readonly OPENCODE_REPO=${OPENCODE_REPO:-$DEVELOPMENT_ROOT/opencode}
readonly HERMES_REPO=${HERMES_REPO:-$DEVELOPMENT_ROOT/hermes-agent}

mkdir -p "$STATE_ROOT/logs" "$STATE_ROOT/status"

record() {
  python3 - "$LEDGER" "$1" "$2" "$3" "$4" "$5" "$6" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ledger, framework, benchmark, repetition, phase, status, detail = sys.argv[1:]
record = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "framework": framework,
    "benchmark": benchmark,
    "repetition": int(repetition),
    "phase": phase,
    "status": status,
    "detail": detail,
}
with Path(ledger).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record, sort_keys=True) + "\n")
PY
}

phase_completed() {
  local marker="$STATE_ROOT/status/$1-$2-r$(printf '%02d' "$3")-$4.complete"
  [[ -f "$marker" ]]
}

mark_completed() {
  local marker="$STATE_ROOT/status/$1-$2-r$(printf '%02d' "$3")-$4.complete"
  python3 - "$marker" <<'PY'
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    stream.write(datetime.now(timezone.utc).isoformat() + "\n")
os.replace(temporary, path)
PY
}

run_logged() {
  local framework=$1 benchmark=$2 repetition=$3 phase=$4
  shift 4
  local log="$STATE_ROOT/logs/$framework-$benchmark-r$(printf '%02d' "$repetition")-$phase.log"
  record "$framework" "$benchmark" "$repetition" "$phase" running "$log"
  set +e
  "$@" 2>&1 | tee -a "$log"
  local code=${PIPESTATUS[0]}
  set -e
  if (( code != 0 )); then
    record "$framework" "$benchmark" "$repetition" "$phase" failed "exit=$code log=$log"
    return "$code"
  fi
  mark_completed "$framework" "$benchmark" "$repetition" "$phase"
  record "$framework" "$benchmark" "$repetition" "$phase" completed "$log"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Required command is unavailable: %s\n' "$1" >&2
    exit 2
  fi
}

assert_capacity() {
  local free
  free=$(df --output=avail -B1 "$DEVELOPMENT_ROOT" | tail -1 | tr -d ' ')
  if (( free < MINIMUM_FREE_BYTES )); then
    printf 'Stopping before disk exhaustion: %s bytes free, %s required.\n' "$free" "$MINIMUM_FREE_BYTES" >&2
    exit 3
  fi
}

assert_idle_docker() {
  local running
  running=$(docker ps -q | wc -l)
  if (( running != 0 )); then
    printf 'Refusing to overlap a batch with %s existing Docker container(s).\n' "$running" >&2
    exit 3
  fi
}

assert_source_clean() {
  local repo=$1
  if ! git -C "$repo" diff --quiet HEAD -- . ':(exclude).benchmark-traces/**'; then
    printf 'Tracked source changes would invalidate the experiment: %s\n' "$repo" >&2
    exit 3
  fi
  local unexpected
  unexpected=$(git -C "$repo" ls-files --others --exclude-standard | grep -v '^\.benchmark-traces/' || true)
  if [[ -n "$unexpected" ]]; then
    printf 'Unexpected source-visible untracked files in %s:\n%s\n' "$repo" "$unexpected" >&2
    exit 3
  fi
}

assert_repository_ready() {
  local repo=$1
  if [[ $(git -C "$repo" branch --show-current) != play ]]; then
    printf 'Experiment repository is not on play: %s\n' "$repo" >&2
    exit 3
  fi
  if [[ $(git -C "$repo" rev-parse HEAD) != $(git -C "$repo" rev-parse origin/play) ]]; then
    printf 'Experiment repository is not synchronized with origin/play: %s\n' "$repo" >&2
    exit 3
  fi
  assert_source_clean "$repo"
}

initialize_source_lock() {
  if [[ -f "$SOURCE_LOCK" ]]; then
    jq -e '
      .schema_version == 1 and
      (.revisions.opencode | type == "string" and length == 40) and
      (.revisions.openhands | type == "string" and length == 40) and
      (.revisions.hermes | type == "string" and length == 40)
    ' "$SOURCE_LOCK" >/dev/null
    return
  fi

  assert_repository_ready "$OPENCODE_REPO"
  assert_repository_ready "$OPENHANDS_REPO"
  assert_repository_ready "$HERMES_REPO"
  python3 - "$SOURCE_LOCK" \
    "$(git -C "$OPENCODE_REPO" rev-parse HEAD)" \
    "$(git -C "$OPENHANDS_REPO" rev-parse HEAD)" \
    "$(git -C "$HERMES_REPO" rev-parse HEAD)" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "revisions": {
        "opencode": sys.argv[2],
        "openhands": sys.argv[3],
        "hermes": sys.argv[4],
    },
}
fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
os.replace(temporary, path)
PY
}

assert_source_revision() {
  local repo=$1 framework=$2
  if [[ $(git -C "$repo" rev-parse HEAD) != $(jq -r ".revisions.$framework" "$SOURCE_LOCK") ]]; then
    printf 'Source revision changed after experiment launch: %s\n' "$repo" >&2
    exit 3
  fi
  assert_source_clean "$repo"
}

validate_design() {
  python3 "$STATE_ROOT/validate.py"
  bash -n "$STATE_ROOT/run-matrix.sh"
}

preflight() {
  validate_design
  local command
  for command in bun df docker flock git jq python3 stat uv "$HARBOR_BIN"; do
    require_command "$command"
  done
  if [[ ! -s "$KEY_FILE" ]]; then
    printf 'OpenRouter key file is missing or empty: %s\n' "$KEY_FILE" >&2
    exit 2
  fi
  if [[ $(stat -c '%a' "$KEY_FILE") != 600 ]]; then
    printf 'OpenRouter key file must have mode 600: %s\n' "$KEY_FILE" >&2
    exit 2
  fi
  if [[ ! -x "$EVALUATOR_PYTHON" ]]; then
    printf 'SWE-bench evaluator Python is unavailable: %s\n' "$EVALUATOR_PYTHON" >&2
    exit 2
  fi
  if [[ $("$EVALUATOR_PYTHON" -c "import importlib.metadata; print(importlib.metadata.version('swebench'))") != 4.1.0 ]]; then
    printf 'SWE-bench evaluator must be version 4.1.0: %s\n' "$EVALUATOR_PYTHON" >&2
    exit 2
  fi
  if [[ $("$HARBOR_BIN" --version) != "$EXPECTED_HARBOR_VERSION" ]]; then
    printf 'Harbor must be version %s: %s\n' "$EXPECTED_HARBOR_VERSION" "$HARBOR_BIN" >&2
    exit 2
  fi
  docker info --format '{{.ServerVersion}}' >/dev/null
  assert_capacity
  assert_idle_docker
  initialize_source_lock
  assert_source_revision "$OPENCODE_REPO" opencode
  assert_source_revision "$OPENHANDS_REPO" openhands
  assert_source_revision "$HERMES_REPO" hermes
}

load_credentials() {
  OPENROUTER_API_KEY=$(<"$KEY_FILE")
  export OPENROUTER_API_KEY
  export HF_HUB_DISABLE_XET=1
  unset HERMES_BENCH_MODEL OPENCODE_BENCH_MODEL OPENCODE_MODEL OPENROUTER_MODEL
  trap 'unset OPENROUTER_API_KEY' EXIT
}

load_ids() {
  local benchmark=$1
  mapfile -t IDS < <(awk 'NF { print }' "$STATE_ROOT/$benchmark.txt")
  if [[ ${#IDS[@]} -ne "$INSTANCES_PER_BENCHMARK" ]]; then
    printf 'Expected %s explicit IDs for %s, found %s.\n' \
      "$INSTANCES_PER_BENCHMARK" "$benchmark" "${#IDS[@]}" >&2
    exit 3
  fi
  IDS_JSON=$(jq -R -s 'split("\n") | map(select(length > 0))' "$STATE_ROOT/$benchmark.txt")
}

instance_args() {
  INSTANCE_ARGS=()
  local id
  for id in "${IDS[@]}"; do
    INSTANCE_ARGS+=(--instance-id "$id")
  done
}

task_args() {
  TASK_ARGS=()
  local id
  for id in "${IDS[@]}"; do
    TASK_ARGS+=(--task-name "$id")
  done
}

find_openhands_output() {
  [[ -d "$1" ]] || return 0
  find "$1" -type f -name output.jsonl -print 2>/dev/null | sort | tail -1
}

openhands_output_matches_ids() {
  python3 - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected = {
    line.strip()
    for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
    if line.strip()
}
actual = []
try:
    for line in output.read_text(encoding="utf-8").splitlines():
        if line.strip():
            actual.append(json.loads(line)["instance_id"])
except (KeyError, OSError, TypeError, ValueError):
    raise SystemExit(1) from None
raise SystemExit(0 if len(actual) == len(expected) and set(actual) == expected else 1)
PY
}

run_opencode_verified() {
  local repetition=$1 run_id summary evaluation
  load_ids swe-bench-verified
  instance_args
  run_id="$EXPERIMENT_ID-opencode-verified-r$(printf '%02d' "$repetition")"
  summary="$OPENCODE_REPO/.benchmark-runs/swe-bench-verified/runs/$run_id/summary.json"
  evaluation="$OPENCODE_REPO/.benchmark-runs/swe-bench-verified/runs/$run_id/evaluation-manifest.json"
  if ! jq -e --argjson count "$INSTANCES_PER_BENCHMARK" \
    '.complete == true and .selectedCount == $count' "$summary" >/dev/null 2>&1; then
    run_logged opencode swe-bench-verified "$repetition" inference \
      bash -lc 'cd "$1/packages/opencode" && bun run bench:swe-verified:single -- --run-id "$2" --model "$3" --timeout-ms 900000 --inference-workers 1 --max-infrastructure-retries 0 "${@:4}"' \
      benchmark-command "$OPENCODE_REPO" "$run_id" "$MODEL" "${INSTANCE_ARGS[@]}"
  fi
  if ! jq -e '.status == "completed"' "$evaluation" >/dev/null 2>&1; then
    run_logged opencode swe-bench-verified "$repetition" evaluation \
      bash -lc 'cd "$1/packages/opencode" && bun run bench:swe-verified:single -- --evaluate-only --run-id "$2" --max-workers 1 --evaluation-timeout-seconds 3600 --python "$3"' \
      benchmark-command "$OPENCODE_REPO" "$run_id" "$EVALUATOR_PYTHON"
  fi
}

run_opencode_terminal() {
  local repetition=$1 run_id output_root manifest
  load_ids terminal-bench-2.1
  task_args
  run_id="$EXPERIMENT_ID-opencode-terminal-r$(printf '%02d' "$repetition")"
  output_root="$OPENCODE_REPO/.benchmark-runs/terminal-bench-2.1"
  manifest="$output_root/runs/$run_id/manifest.json"
  if ! jq -e \
    --arg model "$MODEL" \
    --argjson ids "$IDS_JSON" \
    '.status == "completed" and .model == $model and .agentTopology == "single-agent" and .attempts == 1 and .concurrency == 1 and .maxRetries == 0 and .taskNames == $ids' \
    "$manifest" >/dev/null 2>&1; then
    run_logged opencode terminal-bench-2.1 "$repetition" benchmark \
      bash -lc 'cd "$1/packages/opencode" && bun run bench:terminal:single -- --output-dir "$2" --run-id "$3" --model "$4" --harbor-bin "$5" --max-tasks "$6" --attempts 1 --concurrency 1 --max-retries 0 "${@:7}"' \
      benchmark-command "$OPENCODE_REPO" "$output_root" "$run_id" "$MODEL" "$HARBOR_BIN" "$INSTANCES_PER_BENCHMARK" "${TASK_ARGS[@]}"
  fi
}

run_openhands_verified() {
  local repetition=$1 run_id output_base output
  load_ids swe-bench-verified
  run_id="$EXPERIMENT_ID-openhands-verified-r$(printf '%02d' "$repetition")"
  output_base="$OPENHANDS_REPO/.benchmark-runs/$EXPERIMENT_ID/swe-bench-verified/r$(printf '%02d' "$repetition")"
  output=$(find_openhands_output "$output_base")
  if [[ -z "$output" ]] || ! openhands_output_matches_ids "$output" "$STATE_ROOT/swe-bench-verified.txt"; then
    run_logged openhands swe-bench-verified "$repetition" inference \
      bash -lc 'cd "$1" && OPENHANDS_SUPPRESS_BANNER=1 uv run swebench-verified-single-infer --output-dir "$2" --note "$3" --select "$4" --n-limit "$5" --num-workers 1 --n-critic-runs 1 --max-retries 0 --max-iterations 24 --inference-timeout 900' \
      benchmark-command "$OPENHANDS_REPO" "$output_base" "$run_id" "$STATE_ROOT/swe-bench-verified.txt" "$INSTANCES_PER_BENCHMARK"
    output=$(find_openhands_output "$output_base")
  fi
  if [[ -z "$output" ]]; then
    printf 'OpenHands did not produce output.jsonl for %s.\n' "$run_id" >&2
    exit 3
  fi
  if ! phase_completed openhands swe-bench-verified "$repetition" evaluation; then
    run_logged openhands swe-bench-verified "$repetition" evaluation \
      bash -lc 'cd "$1" && OPENHANDS_SUPPRESS_BANNER=1 uv run swebench-eval "$2" --dataset princeton-nlp/SWE-bench_Verified --split test --run-id "$3" --no-modal --workers 1 --timeout 3600' \
      benchmark-command "$OPENHANDS_REPO" "$output" "$run_id"
  fi
}

run_openhands_terminal() {
  local repetition=$1 run_id output_root manifest
  load_ids terminal-bench-2.1
  run_id="$EXPERIMENT_ID-openhands-terminal-r$(printf '%02d' "$repetition")"
  output_root="$OPENHANDS_REPO/.benchmark-runs"
  manifest="$output_root/terminal-bench-2.1/runs/$run_id/manifest.json"
  if ! jq -e \
    --arg model "$MODEL" \
    --argjson ids "$IDS_JSON" \
    '.status == "completed" and .model == $model and .agent_topology == "single-agent" and .attempts == 1 and .concurrency == 1 and .max_retries == 0 and .task_ids == $ids' \
    "$manifest" >/dev/null 2>&1; then
    run_logged openhands terminal-bench-2.1 "$repetition" benchmark \
      bash -lc 'cd "$1" && OPENHANDS_SUPPRESS_BANNER=1 uv run terminalbench-single-infer --output-dir "$2" --run-id "$3" --select "$4" --n-limit "$5" --num-workers 1 --n-attempts 1 --max-retries 0 --harbor-bin "$6"' \
      benchmark-command "$OPENHANDS_REPO" "$output_root" "$run_id" "$STATE_ROOT/terminal-bench-2.1.txt" "$INSTANCES_PER_BENCHMARK" "$HARBOR_BIN"
  fi
}

run_hermes_verified() {
  local repetition=$1 run_id summary evaluation
  load_ids swe-bench-verified
  instance_args
  run_id="$EXPERIMENT_ID-hermes-verified-r$(printf '%02d' "$repetition")"
  summary="$HERMES_REPO/.benchmark-runs/swe-bench-verified/runs/$run_id/summary.json"
  evaluation="$HERMES_REPO/.benchmark-runs/swe-bench-verified/runs/$run_id/evaluation-manifest.json"
  if ! jq -e --argjson count "$INSTANCES_PER_BENCHMARK" \
    '.complete == true and .selectedCount == $count' "$summary" >/dev/null 2>&1; then
    run_logged hermes swe-bench-verified "$repetition" inference \
      bash -lc 'cd "$1" && uv run hermes-swebench-verified-single infer --run-id "$2" --model "$3" --agent-timeout-seconds 900 --max-instances "$4" "${@:5}"' \
      benchmark-command "$HERMES_REPO" "$run_id" "$MODEL" "$INSTANCES_PER_BENCHMARK" "${INSTANCE_ARGS[@]}"
  fi
  if ! jq -e '.status == "completed"' "$evaluation" >/dev/null 2>&1; then
    run_logged hermes swe-bench-verified "$repetition" evaluation \
      bash -lc 'cd "$1" && uv run hermes-swebench-verified-single evaluate --run-id "$2" --python "$3" "${@:4}"' \
      benchmark-command "$HERMES_REPO" "$run_id" "$EVALUATOR_PYTHON" "${INSTANCE_ARGS[@]}"
  fi
}

run_hermes_terminal() {
  local repetition=$1 run_id output_root manifest
  load_ids terminal-bench-2.1
  task_args
  run_id="$EXPERIMENT_ID-hermes-terminal-r$(printf '%02d' "$repetition")"
  output_root="$HERMES_REPO/.benchmark-runs/terminal-bench-2.1"
  manifest="$output_root/runs/$run_id/manifest.json"
  if ! jq -e \
    --arg model "$MODEL" \
    --argjson ids "$IDS_JSON" \
    '.status == "completed" and .model == $model and .agentTopology == "single-agent" and .attempts == 1 and .concurrency == 1 and .maxRetries == 0 and .taskNames == $ids' \
    "$manifest" >/dev/null 2>&1; then
    run_logged hermes terminal-bench-2.1 "$repetition" benchmark \
      bash -lc 'cd "$1" && uv run hermes-terminalbench-single --output-dir "$2" --run-id "$3" --model "$4" --harbor-bin "$5" --max-tasks "$6" --attempts 1 --concurrency 1 --max-retries 0 "${@:7}"' \
      benchmark-command "$HERMES_REPO" "$output_root" "$run_id" "$MODEL" "$HARBOR_BIN" "$INSTANCES_PER_BENCHMARK" "${TASK_ARGS[@]}"
  fi
}

run_framework() {
  local framework=$1 repetition
  assert_source_revision "$(framework_repo "$framework")" "$framework"
  for repetition in $(seq 1 "$REPETITIONS"); do
    assert_capacity
    assert_idle_docker
    "run_${framework}_verified" "$repetition"
    assert_capacity
    assert_idle_docker
    "run_${framework}_terminal" "$repetition"
  done
}

framework_repo() {
  case $1 in
    opencode) printf '%s\n' "$OPENCODE_REPO" ;;
    openhands) printf '%s\n' "$OPENHANDS_REPO" ;;
    hermes) printf '%s\n' "$HERMES_REPO" ;;
    *) return 2 ;;
  esac
}

case ${1:-all} in
  validate)
    validate_design
    exit 0
    ;;
  check)
    preflight
    printf 'Experiment preflight passed: %s (%s agent runs).\n' \
      "$EXPERIMENT_ID" "$(jq -r '.design.expected_agent_runs' "$MANIFEST")"
    exit 0
    ;;
  opencode | openhands | hermes | all) ;;
  *)
    printf 'Usage: %s [validate|check|opencode|openhands|hermes|all]\n' "$0" >&2
    exit 2
    ;;
esac

preflight
load_credentials

case ${1:-all} in
  opencode | openhands | hermes) run_framework "$1" ;;
  all)
    run_framework opencode
    run_framework openhands
    run_framework hermes
    ;;
esac
