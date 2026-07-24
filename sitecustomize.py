"""
Top-level sitecustomize to ensure our Modal logging patch is always applied.

Python will auto-import ``sitecustomize`` if it is importable on ``sys.path``.
During evaluation ``/workspace/benchmarks`` is on ``PYTHONPATH``, so placing
this file at the repo root guarantees the patch runs before swebench is used.
"""

import os
import sys
from pathlib import Path


_TRACE_CLI = os.environ.get("BENCHMARK_TRACE_CLI") == "1" or Path(sys.argv[0]).name in {
    "benchmark-trace",
    "benchmark-trace.exe",
}


if not _TRACE_CLI:
    print("benchmarks sitecustomize imported", file=sys.stderr, flush=True)
    try:
        # Reuse the actual patch logic that lives alongside the benchmarks package.
        from benchmarks.utils.sitecustomize import _apply_modal_logging_patch

        _apply_modal_logging_patch()
    except Exception:
        # Avoid breaking startup for non-swebench runs; logging is best-effort.
        pass
