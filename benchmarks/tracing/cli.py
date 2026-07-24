"""Researcher CLI for benchmark-trace artifacts.

This module is deliberately read-only. Benchmark runners and framework-native
adapters collect traces while agents execute; these commands analyze the
resulting artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Sequence

from benchmarks.tracing.errors import TraceError
from benchmarks.tracing.models import JsonObject, JsonValue
from benchmarks.tracing.research import (
    compare_traces,
    discover_trace_targets,
    inspect_trace,
    render_trace,
    resolve_trace_target,
    summarize_trace,
    validation_as_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark-trace",
        description=(
            "Validate and analyze traces already captured by benchmark runners. "
            "This command never launches an agent or collects a trace."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="Validate one or more runs, attempts, or trace-base directories.",
    )
    validate.add_argument("paths", nargs="+", type=Path)
    _format_argument(validate)

    inspect = subparsers.add_parser(
        "inspect",
        help="Inspect trace identity, provenance, execution, health, and capabilities.",
    )
    inspect.add_argument("path", type=Path)
    _format_argument(inspect)

    summarize = subparsers.add_parser(
        "summarize",
        help="Summarize normalized activity, timing, coverage, and storage.",
    )
    summarize.add_argument("path", type=Path)
    _format_argument(summarize)

    compare = subparsers.add_parser(
        "compare",
        help="Compare two or more normalized traces and their comparability.",
    )
    compare.add_argument("paths", nargs="+", type=Path)
    _format_argument(compare)

    render = subparsers.add_parser(
        "render",
        help="Render deterministic nested timelines for a run or attempt.",
    )
    render.add_argument("path", type=Path)
    _format_argument(render)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _validate(args.paths, args.format)
        if args.command == "inspect":
            document = inspect_trace(resolve_trace_target(args.path))
            _write(document, args.format, _inspect_text)
            return 0
        if args.command == "summarize":
            document = summarize_trace(resolve_trace_target(args.path))
            _write(document, args.format, _summary_text)
            return 0
        if args.command == "compare":
            if len(args.paths) < 2:
                parser.error("compare requires at least two trace paths")
            document = compare_traces(
                tuple(resolve_trace_target(path) for path in args.paths)
            )
            _write(document, args.format, _comparison_text)
            return 0
        if args.command == "render":
            document = render_trace(resolve_trace_target(args.path))
            _write(document, args.format, _timeline_text)
            return 0
    except (TraceError, OSError, ValueError) as exc:
        print(f"benchmark-trace: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unknown command: {args.command}")


def _validate(paths: list[Path], output_format: str) -> int:
    results: list[JsonValue] = []
    valid = True
    for path in paths:
        for target in discover_trace_targets(path):
            result = validation_as_json(target)
            results.append(result)
            if result["valid"] is not True:
                valid = False
    document: JsonObject = {
        "valid": valid,
        "traces": results,
    }
    _write(document, output_format, _validation_text)
    return 0 if valid else 1


def _format_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. Default: text.",
    )


def _write(
    document: JsonObject,
    output_format: str,
    renderer: Callable[[JsonObject], str],
) -> None:
    if output_format == "json":
        print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(renderer(document), end="")


def _validation_text(document: JsonObject) -> str:
    lines = [f"valid: {'yes' if document['valid'] is True else 'no'}"]
    traces = document["traces"]
    assert isinstance(traces, list)
    for trace in traces:
        assert isinstance(trace, dict)
        lines.append(
            f"{trace['path']} [{trace['kind']}]: "
            f"{'valid' if trace['valid'] is True else 'invalid'}"
        )
        issues = trace["issues"]
        assert isinstance(issues, list)
        for issue in issues:
            assert isinstance(issue, dict)
            lines.append(
                f"  {issue['severity']} {issue['code']} "
                f"{issue['path']}: {issue['message']}"
            )
    return "\n".join(lines) + "\n"


def _inspect_text(document: JsonObject) -> str:
    identity = document["identity"]
    assert isinstance(identity, dict)
    lines = [
        f"kind: {document['kind']}",
        f"path: {document['path']}",
        f"benchmark: {identity['benchmark']}",
        f"framework: {identity['framework']}",
        f"run_id: {identity['run_id']}",
    ]
    if document["kind"] == "attempt":
        lines.extend(
            [
                f"instance_id: {identity['instance_id']}",
                f"attempt: {identity['attempt']}",
            ]
        )
        health = document["health"]
        records = document["records"]
        assert isinstance(health, dict)
        assert isinstance(records, dict)
        lines.extend(
            [
                f"health: {health['status']}",
                f"finalization: {health['finalization']}",
                f"events: {records['events']}",
                f"native_records: {records['native']}",
            ]
        )
    else:
        attempts = document["attempts"]
        selection = document["selection"]
        assert isinstance(attempts, list)
        assert isinstance(selection, dict)
        instance_ids = selection["instance_ids"]
        assert isinstance(instance_ids, list)
        lines.extend(
            [
                f"selection: {selection['strategy']}",
                f"instances: {len(instance_ids)}",
                f"attempts: {len(attempts)}",
            ]
        )
        for attempt in attempts:
            assert isinstance(attempt, dict)
            lines.append(
                f"  {attempt['instance_id']} attempt={attempt['attempt']} "
                f"health={attempt['health']} complete={attempt['complete']}"
            )
    return "\n".join(lines) + "\n"


def _summary_text(document: JsonObject) -> str:
    identity = document["identity"]
    coverage = document["coverage"]
    events = document["events"]
    activity = document["activity"]
    storage = document["storage"]
    assert isinstance(identity, dict)
    assert isinstance(coverage, dict)
    assert isinstance(events, dict)
    assert isinstance(activity, dict)
    assert isinstance(storage, dict)
    lines = [
        f"path: {document['path']}",
        f"benchmark: {identity['benchmark']}",
        f"framework: {identity['framework']}",
        f"instances: {coverage['instances']}",
        f"attempts: {coverage['attempts']}",
        f"events: {events['total']}",
        f"native_records: {events['native_records']}",
        f"model_turns: {activity['model_turns']}",
        f"tool_calls: {activity['tool_calls']}",
        f"delegations: {activity['delegations']}",
        f"compactions: {activity['compactions']}",
        f"trace_issues: {activity['trace_issues']}",
        f"dropped_events: {storage['dropped_events']}",
        f"redactions: {storage['redactions']}",
    ]
    return "\n".join(lines) + "\n"


def _comparison_text(document: JsonObject) -> str:
    checks = document["checks"]
    traces = document["traces"]
    assert isinstance(checks, dict)
    assert isinstance(traces, list)
    lines = [
        f"comparable: {'yes' if document['comparable'] is True else 'no'}",
        "checks: "
        + ", ".join(
            f"{key}={'same' if value is True else 'different'}"
            for key, value in checks.items()
        ),
        (
            "framework | benchmark | instances | attempts | events | tools | "
            "delegations | compactions | issues"
        ),
    ]
    for trace in traces:
        assert isinstance(trace, dict)
        lines.append(
            f"{trace['framework']} | {trace['benchmark']} | "
            f"{trace['instances']} | {trace['attempts']} | {trace['events']} | "
            f"{trace['tool_calls']} | {trace['delegations']} | "
            f"{trace['compactions']} | {trace['trace_issues']}"
        )
    differences = document["capability_differences"]
    assert isinstance(differences, dict)
    lines.append(
        "capability_differences: " + (", ".join(differences) if differences else "none")
    )
    return "\n".join(lines) + "\n"


def _timeline_text(document: JsonObject) -> str:
    timelines = document["timelines"]
    assert isinstance(timelines, list)
    lines: list[str] = []
    for timeline in timelines:
        assert isinstance(timeline, dict)
        lines.append(
            f"== {timeline['instance_id']} attempt={timeline['attempt']} "
            f"trace={timeline['trace_id']} =="
        )
        entries = timeline["entries"]
        assert isinstance(entries, list)
        for entry in entries:
            assert isinstance(entry, dict)
            duration = (
                f" duration={entry['duration_ms']:.3f}ms"
                if isinstance(entry.get("duration_ms"), int | float)
                else ""
            )
            detail = f" {entry['detail']}" if "detail" in entry else ""
            depth = entry["depth"]
            assert isinstance(depth, int)
            lines.append(
                f"+{entry['relative_ms']:012.3f}ms "
                f"{'  ' * depth}{entry['event_type']} "
                f"{entry['phase']}/{entry['status']} actor={entry['actor']}"
                f"{duration}{detail}"
            )
    return "\n".join(lines) + ("\n" if lines else "")


if __name__ == "__main__":
    raise SystemExit(main())
