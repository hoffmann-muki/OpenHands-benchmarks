from types import SimpleNamespace
from typing import Callable
from unittest.mock import Mock, call, patch
from uuid import UUID

import pytest

from benchmarks.utils.fake_user_response import (
    run_conversation_with_fake_user_response,
)
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.conversation.state import ConversationExecutionStatus


def test_timeout_interrupts_remote_conversation() -> None:
    conversation = Mock()
    error = ConversationRunError(UUID(int=1), TimeoutError("timed out"))
    conversation.run.side_effect = error

    with pytest.raises(ConversationRunError) as exc_info:
        run_conversation_with_fake_user_response(
            conversation,
            timeout_seconds=30,
        )

    assert exc_info.value is error
    conversation.interrupt.assert_called_once_with()


def test_non_timeout_failure_does_not_interrupt_conversation() -> None:
    conversation = Mock()
    error = ConversationRunError(UUID(int=1), RuntimeError("failed"))
    conversation.run.side_effect = error

    with pytest.raises(ConversationRunError) as exc_info:
        run_conversation_with_fake_user_response(
            conversation,
            timeout_seconds=30,
        )

    assert exc_info.value is error
    conversation.interrupt.assert_not_called()


def test_deadline_watchdog_interrupts_a_blocked_remote_run() -> None:
    conversation = Mock()
    conversation.interrupt.side_effect = [RuntimeError("run not started"), None]
    timer = Mock()
    callbacks: list[Callable[[], None]] = []

    def make_timer(interval: float, callback: Callable[[], None]) -> Mock:
        assert interval == 30
        callbacks.append(callback)
        return timer

    def expire_during_run(*, timeout: float) -> None:
        assert timeout > 0
        callbacks[0]()

    conversation.run.side_effect = expire_during_run

    with (
        patch(
            "benchmarks.utils.fake_user_response.threading.Timer",
            side_effect=make_timer,
        ),
        pytest.raises(TimeoutError, match="shared 30s timeout"),
    ):
        run_conversation_with_fake_user_response(
            conversation,
            timeout_seconds=30,
        )

    timer.start.assert_called_once_with()
    timer.cancel.assert_called_once_with()
    assert conversation.interrupt.call_count == 2


def test_deadline_is_authoritative_for_interrupt_induced_failure() -> None:
    conversation = Mock()
    timer = Mock()
    callbacks: list[Callable[[], None]] = []
    error = ConversationRunError(UUID(int=1), RuntimeError("interrupted"))

    def make_timer(interval: float, callback: Callable[[], None]) -> Mock:
        assert interval == 30
        callbacks.append(callback)
        return timer

    def expire_during_run(*, timeout: float) -> None:
        assert timeout > 0
        callbacks[0]()
        raise error

    conversation.run.side_effect = expire_during_run

    with (
        patch(
            "benchmarks.utils.fake_user_response.threading.Timer",
            side_effect=make_timer,
        ),
        pytest.raises(TimeoutError, match="shared 30s timeout") as exc_info,
    ):
        run_conversation_with_fake_user_response(
            conversation,
            timeout_seconds=30,
        )

    assert exc_info.value.__cause__ is error
    assert conversation.interrupt.call_count == 2


def test_continuations_share_one_deadline() -> None:
    conversation = Mock()
    conversation.state = SimpleNamespace(
        execution_status=ConversationExecutionStatus.FINISHED,
        events=[],
    )

    with (
        patch(
            "benchmarks.utils.fake_user_response.time.monotonic",
            side_effect=[100.0, 101.0, 110.0],
        ),
        patch(
            "benchmarks.utils.fake_user_response._agent_finished_with_finish_action",
            side_effect=[False, True],
        ),
        patch(
            "benchmarks.utils.fake_user_response._agent_sent_message",
            return_value=True,
        ),
    ):
        run_conversation_with_fake_user_response(
            conversation,
            fake_user_response_fn=lambda _: "continue",
            timeout_seconds=20,
        )

    assert conversation.run.call_args_list == [call(timeout=19.0), call(timeout=10.0)]
    conversation.send_message.assert_called_once_with("continue")
