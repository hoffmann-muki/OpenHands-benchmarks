"""OpenHands Benchmarking Suite"""

import os


if os.environ.get("BENCHMARK_TRACE_CLI") != "1":
    # Pre-import these tools to register pydantic models for benchmark runtime
    # serialization. Trace research tooling has no dependency on that registry.
    from openhands.tools.file_editor import FileEditorTool  # noqa: F401
    from openhands.tools.task_tracker import TaskTrackerTool  # noqa: F401
    from openhands.tools.terminal import TerminalTool  # noqa: F401
