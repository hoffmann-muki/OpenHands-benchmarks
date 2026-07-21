#!/usr/bin/env python3
"""Bootstrap Harbor's OpenHands SDK runner with native task delegation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from openhands.tools.preset.default import register_builtins_agents
from openhands.tools.task import TaskToolSet


BASE_RUNNER_PATH = Path("/installed-agent/run_agent_base.py")
CONVERSATION_LOG_DIR = Path("/logs/agent/conversation")


def load_base_runner(path: Path = BASE_RUNNER_PATH) -> ModuleType:
    spec = importlib.util.spec_from_file_location("harbor_openhands_sdk_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Harbor OpenHands runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_delegation(module: Any) -> None:
    """Add task delegation, durable traces, and aggregate usage accounting."""
    register_builtins_agents(enable_browser=False)

    original_agent = module.Agent
    original_conversation = module.Conversation
    original_build_trajectory = module.build_trajectory
    captured: dict[str, Any] = {}

    def delegating_agent(*args: Any, **kwargs: Any) -> Any:
        tools = list(kwargs.get("tools", []))
        if not any(getattr(tool, "name", None) == TaskToolSet.name for tool in tools):
            tools.append(module.Tool(name=TaskToolSet.name))
        kwargs["tools"] = tools
        return original_agent(*args, **kwargs)

    def persistent_conversation(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("persistence_dir", CONVERSATION_LOG_DIR)
        conversation = original_conversation(*args, **kwargs)
        captured["conversation"] = conversation
        return conversation

    def build_trajectory(
        events: list[dict[str, Any]],
        llm_metrics: dict[str, Any],
        model_name: str,
        system_prompt: str | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        conversation = captured.get("conversation")
        if conversation is not None:
            combined = conversation.conversation_stats.get_combined_metrics()
            usage = combined.accumulated_token_usage
            llm_metrics = {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "cached_tokens": usage.cache_read_tokens if usage else 0,
                "cost_usd": combined.accumulated_cost,
            }
        return original_build_trajectory(
            events,
            llm_metrics,
            model_name,
            system_prompt=system_prompt,
            tool_definitions=tool_definitions,
        )

    setattr(module, "Agent", delegating_agent)
    setattr(module, "Conversation", persistent_conversation)
    setattr(module, "build_trajectory", build_trajectory)


def main() -> None:
    module = load_base_runner()
    configure_delegation(module)
    module.main()


if __name__ == "__main__":
    main()
