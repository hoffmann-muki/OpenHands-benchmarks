# Single-agent 5×1 paired experiment

This experiment runs five SWE-bench Verified instances and five Terminal-Bench
2.1 tasks once on OpenCode, OpenHands, and Hermes: 30 independent agent
attempts in total. The explicit cohorts are identical to the earlier 5×1
multi-agent study, enabling paired single-agent versus multi-agent comparisons.

All three frameworks use their dedicated native single-agent entrypoints with
the same model and execution policy: temperature 0.1, a 24-turn agent budget, a
900-second agent timeout, one provider attempt per turn, one benchmark attempt,
one worker, and zero infrastructure retries. Semantic tracing and AgentSight
profiling remain enabled by default. SWE-bench evaluation uses one local Docker
worker and a 3,600-second test timeout; Terminal-Bench evaluation remains part
of Harbor 0.20.0's single end-to-end trial.

Run a credential-free design validation first:

```bash
./run-matrix.sh validate
```

After all three `play` branches are committed and pushed, run the complete
preflight without starting inference:

```bash
./run-matrix.sh check
```

Run or resume the complete matrix from an attached terminal in this directory:

```bash
./run-matrix.sh all
```

Use `./run-matrix.sh opencode`, `openhands`, or `hermes` to run or resume one
framework. The ignored ledger, completion markers, and source lock make resumes
idempotent while keeping orchestration state out of Git. Benchmark outputs and
automatically collected traces remain under each harness repository's
`.benchmark-runs` and `.benchmark-traces` directories. Keep the invoking
terminal or execution session attached until the command exits.

The default key file is `<workspace>/openrouter-key` and must have mode `0600`.
Set `OPENROUTER_KEY_FILE` or `BENCHMARK_WORKSPACE_ROOT` to override local paths.
