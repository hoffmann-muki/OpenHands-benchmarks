#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR
readonly OPENHANDS_REPO=${OPENHANDS_BENCHMARKS_REPO:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}
readonly DEVELOPMENT_ROOT=${BENCHMARK_WORKSPACE_ROOT:-$(dirname "$OPENHANDS_REPO")}
readonly EXPERIMENT_ID=single-agent-10x3-20260810
readonly STATE_ROOT="$SCRIPT_DIR"
readonly KEY_FILE=${OPENROUTER_KEY_FILE:-$DEVELOPMENT_ROOT/openrouter-key}
readonly EVALUATOR_PYTHON=${SWEBENCH_PYTHON:-$OPENHANDS_REPO/.venv/bin/python}
readonly LEDGER="$STATE_ROOT/ledger.jsonl"
readonly SOURCE_LOCK="$STATE_ROOT/source-lock.json"
readonly MINIMUM_FREE_BYTES=$((30 * 1024 * 1024 * 1024))
REPETITIONS=$(jq -r '.design.repetitions_per_instance' "$STATE_ROOT/manifest.json")
readonly REPETITIONS
INSTANCES_PER_BENCHMARK=$(jq -r '.design.instances_per_benchmark' "$STATE_ROOT/manifest.json")
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

preflight() {
  python3 "$STATE_ROOT/validate.py"
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
  "$EVALUATOR_PYTHON" -c "import importlib.metadata; importlib.metadata.version('swebench')"
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
  trap 'unset OPENROUTER_API_KEY' EXIT
}

select_benchmark() {
  local benchmark=$1
  if [[ "$benchmark" == swe-bench-lite ]]; then
    DATASET=princeton-nlp/SWE-bench_Lite
    COMMAND_SUFFIX=lite
  elif [[ "$benchmark" == swe-bench-verified ]]; then
    DATASET=princeton-nlp/SWE-bench_Verified
    COMMAND_SUFFIX=verified
  else
    printf 'Unsupported benchmark: %s\n' "$benchmark" >&2
    exit 2
  fi
  mapfile -t IDS < <(awk 'NF { print }' "$STATE_ROOT/$benchmark.txt")
  [[ ${#IDS[@]} -eq "$INSTANCES_PER_BENCHMARK" ]]
}

instance_args() {
  INSTANCE_ARGS=()
  local id
  for id in "${IDS[@]}"; do
    INSTANCE_ARGS+=(--instance-id "$id")
  done
}

run_opencode() {
  assert_source_revision "$OPENCODE_REPO" opencode
  local benchmark repetition run_id summary evaluation
  for benchmark in swe-bench-lite swe-bench-verified; do
    select_benchmark "$benchmark"
    instance_args
    for repetition in $(seq 1 "$REPETITIONS"); do
      assert_capacity
      assert_idle_docker
      run_id="$EXPERIMENT_ID-opencode-$COMMAND_SUFFIX-r$(printf '%02d' "$repetition")"
      summary="$OPENCODE_REPO/.benchmark-runs/$benchmark/runs/$run_id/summary.json"
      evaluation="$OPENCODE_REPO/.benchmark-runs/$benchmark/runs/$run_id/evaluation-manifest.json"
      if ! jq -e --argjson count "$INSTANCES_PER_BENCHMARK" \
        '.complete == true and .selectedCount == $count' "$summary" >/dev/null 2>&1; then
        run_logged opencode "$benchmark" "$repetition" inference \
          bash -lc 'cd "$1/packages/opencode" && bun run "bench:swe-$2:single" -- --run-id "$3" "${@:4}"' \
          benchmark-command "$OPENCODE_REPO" "$COMMAND_SUFFIX" "$run_id" "${INSTANCE_ARGS[@]}"
      fi
      if ! jq -e '.status == "completed"' "$evaluation" >/dev/null 2>&1; then
        run_logged opencode "$benchmark" "$repetition" evaluation \
          bash -lc 'cd "$1/packages/opencode" && bun run "bench:swe-$2:single" -- --evaluate-only --run-id "$3" --max-workers 1 --evaluation-timeout-seconds 3600 --python "$4"' \
          benchmark-command "$OPENCODE_REPO" "$COMMAND_SUFFIX" "$run_id" "$EVALUATOR_PYTHON"
      fi
    done
  done
}

find_openhands_output() {
  find "$1" -type f -name output.jsonl -print 2>/dev/null | sort | tail -1
}

run_openhands() {
  assert_source_revision "$OPENHANDS_REPO" openhands
  local benchmark repetition run_id output_base output
  for benchmark in swe-bench-lite swe-bench-verified; do
    select_benchmark "$benchmark"
    for repetition in $(seq 1 "$REPETITIONS"); do
      assert_capacity
      assert_idle_docker
      run_id="$EXPERIMENT_ID-openhands-$COMMAND_SUFFIX-r$(printf '%02d' "$repetition")"
      output_base="$OPENHANDS_REPO/.benchmark-runs/$EXPERIMENT_ID/$benchmark/r$(printf '%02d' "$repetition")"
      output=$(find_openhands_output "$output_base")
      if [[ -z "$output" ]] || (( $(awk 'NF { count++ } END { print count+0 }' "$output") != INSTANCES_PER_BENCHMARK )); then
        run_logged openhands "$benchmark" "$repetition" inference \
          bash -lc 'cd "$1" && OPENHANDS_SUPPRESS_BANNER=1 uv run "swebench-$2-single-infer" --output-dir "$3" --note "$4" --select "$5" --n-limit "$6" --num-workers 1 --n-critic-runs 1 --max-retries 0 --disable-delegation --inference-timeout 900' \
          benchmark-command "$OPENHANDS_REPO" "$COMMAND_SUFFIX" "$output_base" "$run_id" "$STATE_ROOT/$benchmark.txt" "$INSTANCES_PER_BENCHMARK"
        output=$(find_openhands_output "$output_base")
      fi
      [[ -n "$output" ]]
      if ! phase_completed openhands "$benchmark" "$repetition" evaluation; then
        run_logged openhands "$benchmark" "$repetition" evaluation \
          bash -lc 'cd "$1" && OPENHANDS_SUPPRESS_BANNER=1 uv run swebench-eval "$2" --dataset "$3" --split test --run-id "$4" --no-modal --workers 1 --timeout 3600' \
          benchmark-command "$OPENHANDS_REPO" "$output" "$DATASET" "$run_id"
      fi
    done
  done
}

run_hermes() {
  assert_source_revision "$HERMES_REPO" hermes
  local benchmark repetition run_id summary evaluation
  for benchmark in swe-bench-lite swe-bench-verified; do
    select_benchmark "$benchmark"
    instance_args
    for repetition in $(seq 1 "$REPETITIONS"); do
      assert_capacity
      assert_idle_docker
      run_id="$EXPERIMENT_ID-hermes-$COMMAND_SUFFIX-r$(printf '%02d' "$repetition")"
      summary="$HERMES_REPO/.benchmark-runs/$benchmark/runs/$run_id/summary.json"
      evaluation="$HERMES_REPO/.benchmark-runs/$benchmark/runs/$run_id/evaluation-manifest.json"
      if ! jq -e --argjson count "$INSTANCES_PER_BENCHMARK" \
        '.complete == true and .selectedCount == $count' "$summary" >/dev/null 2>&1; then
        run_logged hermes "$benchmark" "$repetition" inference \
          bash -lc 'cd "$1" && uv run "hermes-swebench-$2-single" infer --run-id "$3" "${@:4}"' \
          benchmark-command "$HERMES_REPO" "$COMMAND_SUFFIX" "$run_id" "${INSTANCE_ARGS[@]}"
      fi
      if ! jq -e '.status == "completed"' "$evaluation" >/dev/null 2>&1; then
        run_logged hermes "$benchmark" "$repetition" evaluation \
          bash -lc 'cd "$1" && uv run "hermes-swebench-$2-single" evaluate --run-id "$3" --python "$4" "${@:5}"' \
          benchmark-command "$HERMES_REPO" "$COMMAND_SUFFIX" "$run_id" "$EVALUATOR_PYTHON" "${INSTANCE_ARGS[@]}"
      fi
    done
  done
}

case ${1:-all} in
  check)
    preflight
    printf 'Experiment preflight passed: %s (%s agent runs).\n' \
      "$EXPERIMENT_ID" "$(jq -r '.design.expected_agent_runs' "$STATE_ROOT/manifest.json")"
    exit 0
    ;;
  opencode | openhands | hermes | all) ;;
  *)
    printf 'Usage: %s [check|opencode|openhands|hermes|all]\n' "$0" >&2
    exit 2
    ;;
esac

preflight
load_credentials

case ${1:-all} in
  opencode) run_opencode ;;
  openhands) run_openhands ;;
  hermes) run_hermes ;;
  all)
    run_opencode
    run_openhands
    run_hermes
    ;;
esac
