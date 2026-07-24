"""Dependency-light bootstrap for the benchmark trace researcher CLI."""

import os
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["BENCHMARK_TRACE_CLI"] = "1"
    from benchmarks.tracing.cli import main as tracing_main

    return tracing_main(argv)
