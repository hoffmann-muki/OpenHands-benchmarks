"""Fresh-only task delegation for fixed-budget SWE benchmark runs."""

from typing import TYPE_CHECKING

from openhands.sdk.tool import ToolDefinition, register_tool
from openhands.tools.task import TaskToolSet
from openhands.tools.task.impl import TaskExecutor
from openhands.tools.task.manager import Task, TaskManager


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState
    from openhands.tools.task.manager import ConfirmationHandler


class FreshOnlyTaskManager(TaskManager):
    """Task manager that never restores a completed subagent conversation."""

    def _resume_task(self, resume: str, subagent_type: str) -> Task:
        raise ValueError(
            "SWE benchmark task resumption is disabled; start the next fixed phase "
            "as a fresh task"
        )


class FreshOnlyTaskToolSet(TaskToolSet):
    """Task tool set whose executor rejects every subagent resume request."""

    name = "swe_benchmark_task_tool_set"

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        confirmation_handler: "ConfirmationHandler | None" = None,
    ) -> list[ToolDefinition]:
        tools = super().create(
            conv_state=conv_state,
            confirmation_handler=confirmation_handler,
        )
        return [
            tool.set_executor(
                TaskExecutor(
                    manager=FreshOnlyTaskManager(
                        confirmation_handler=confirmation_handler
                    )
                )
            )
            for tool in tools
        ]


register_tool(FreshOnlyTaskToolSet.name, FreshOnlyTaskToolSet)
