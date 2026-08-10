# Single-agent 10×3 experiment

This experiment runs ten repository-stratified SWE-bench Lite instances and ten
repository-stratified SWE-bench Verified instances three times on OpenCode,
OpenHands, and Hermes: 180 independent agent attempts in total.

The runner preserves the single-agent parity configuration: one inference and
evaluation worker, 24 agent turns, a 900-second agent timeout, one provider
attempt per turn, and zero benchmark retries. Semantic tracing and AgentSight
profiling remain enabled by each harness.

Run the complete matrix from this directory:

```bash
mkdir -p logs
nohup setsid ./supervise-matrix.sh > logs/supervisor.log 2>&1 < /dev/null &
```

Use `./run-matrix.sh opencode`, `openhands`, or `hermes` to run or resume one
framework. The ignored ledger, completion markers, and source lock make resumes
idempotent while keeping generated state out of Git. Harness outputs remain in
each sibling repository's `.benchmark-runs` and `.benchmark-traces` directories.

Run `./run-matrix.sh check` after all three `play` branches are committed and
pushed to validate the design, dependencies, source lock, and local capacity
without starting inference.

The default key file is `<workspace>/openrouter-key` and must have mode `0600`.
Set `OPENROUTER_KEY_FILE` or `BENCHMARK_WORKSPACE_ROOT` to override local paths.
