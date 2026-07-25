"""OpenHands SDK callback adapter for the benchmark trace contract."""

import json
import tarfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from benchmarks.tracing.models import (
    Capability,
    JsonObject,
    JsonValue,
    TraceIdentity,
)
from benchmarks.tracing.recorder import FinalizationResult, TraceRecorder
from openhands.sdk.event import (
    ACPToolCallEvent,
    ActionEvent,
    AgentErrorEvent,
    Condensation,
    CondensationRequest,
    CondensationSummaryEvent,
    ConversationStateUpdateEvent,
    Event,
    HookExecutionEvent,
    InterruptEvent,
    LLMCompletionLogEvent,
    MessageEvent,
    ObservationEvent,
    PauseEvent,
    StreamingDeltaEvent,
    SystemPromptEvent,
    TokenEvent,
    UserRejectObservation,
)
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.llm import TextContent


AdapterStatus = Literal["completed", "failed", "cancelled", "timeout", "degraded"]

_CAPABILITY_CATEGORIES = (
    "agent.session",
    "model.turn",
    "provider.exchange",
    "tool.invocation",
    "tool.result",
    "tool.timing",
    "shell",
    "file",
    "search",
    "browser",
    "delegation",
    "context.compaction",
    "memory",
    "harness.lifecycle",
    "container.lifecycle",
    "evaluator.lifecycle",
    "patch",
    "native.evidence",
)


@dataclass(frozen=True, slots=True)
class _ObservedEvent:
    event_id: str
    event_type: str


@dataclass(frozen=True, slots=True)
class _PendingSpan:
    span_id: str
    start_event_id: str
    start_type: str
    end_type: str
    event_family: str
    started_at: datetime
    tool_name: str


@dataclass(frozen=True, slots=True)
class _DelegationLink:
    prompt: str
    agent_id: str
    span_id: str


@dataclass(frozen=True, slots=True)
class _ArchivedConversation:
    session_id: str
    status: AdapterStatus
    events: tuple[Event, ...]
    complete: bool


def openhands_capabilities(
    *,
    observed: Mapping[str, set[str]] | None = None,
    delegation_enabled: bool,
    condenser_enabled: bool,
    browser_enabled: bool,
    completion_logs_enabled: bool = False,
    container_enabled: bool = False,
    evaluator_enabled: bool = False,
    child_archive_ingested: bool = False,
    child_archive_complete: bool = False,
) -> tuple[Capability, ...]:
    """Build an exhaustive, attempt-specific OpenHands capability matrix."""

    evidence = observed or {}
    disabled = {
        *(() if delegation_enabled else ("delegation",)),
        *(() if condenser_enabled else ("context.compaction",)),
        *(() if browser_enabled else ("browser",)),
        *(() if completion_logs_enabled else ("provider.exchange",)),
    }
    unavailable = {
        "memory",
        *(() if container_enabled else ("container.lifecycle",)),
        *(() if evaluator_enabled else ("evaluator.lifecycle",)),
    }
    characteristics = {
        "agent.session": ("derived", "full", "derived"),
        "model.turn": ("derived", "partial", "derived"),
        "provider.exchange": ("captured", "partial", "native_wall"),
        "tool.invocation": ("captured", "full", "not_available"),
        "tool.result": ("captured", "full", "not_available"),
        "tool.timing": ("derived", "full", "derived"),
        "shell": ("captured", "full", "derived"),
        "file": ("captured", "full", "derived"),
        "search": ("captured", "full", "derived"),
        "browser": ("captured", "full", "derived"),
        "delegation": (
            "captured",
            "full" if child_archive_complete else "partial",
            "derived",
        ),
        "context.compaction": ("captured", "full", "derived"),
        "harness.lifecycle": ("derived", "full", "derived"),
        "container.lifecycle": ("captured", "metadata_only", "native_wall"),
        "evaluator.lifecycle": ("captured", "metadata_only", "native_wall"),
        "patch": ("derived", "full", "derived"),
        "native.evidence": (
            "captured",
            "full" if child_archive_complete else "partial",
            "native_wall",
        ),
    }
    limitations = {
        "agent.session": (
            "Session lifecycle timing is derived around the native conversation.",
        ),
        "model.turn": (
            "Model turns are derived from observable agent-loop boundaries; "
            "the exact provider request boundary is not exposed.",
        ),
        "provider.exchange": (
            "Completion logs are available only when OpenHands completion logging is enabled.",
        ),
        "delegation": (
            (
                "Child events are recovered at attempt finalization from the "
                "agent-server's durable conversation archive; occurred_at keeps "
                "source time while recorded_at reflects archival ingestion."
            )
            if child_archive_complete
            else (
                "The durable child conversation source was read, but one or more "
                "delegations could not be linked to complete child evidence."
            )
            if child_archive_ingested
            else (
                "The parent remote stream exposes delegation boundaries and results, "
                "but its durable child conversation archive was not ingested."
            ),
        ),
        "harness.lifecycle": (
            "Startup and shutdown are coarse harness-owned phases; "
            "benchmark-specific infrastructure is intentionally not subdivided.",
        ),
        "container.lifecycle": (
            "Harbor exposes task-container identity and image metadata, not "
            "operating-system activity below agent tools.",
        ),
        "evaluator.lifecycle": (
            "Evaluator lifecycle is reported only when the outer harness makes "
            "its boundaries observable to the framework adapter.",
        ),
        "native.evidence": (
            "Token ID events and token or cost accounting fields are intentionally excluded.",
            *(
                (
                    "Internal remote subagent events were recovered from the "
                    "agent-server's durable conversation archive.",
                )
                if child_archive_complete
                else (
                    "The durable child conversation source was read, but child "
                    "evidence was incomplete or could not all be linked.",
                )
                if child_archive_ingested
                else (
                    "Internal remote subagent event streams are not forwarded to "
                    "the parent callback and no durable child archive was ingested.",
                )
            ),
        ),
    }

    capabilities = []
    for category in _CAPABILITY_CATEGORIES:
        category_evidence = tuple(sorted(evidence.get(category, set())))
        if category in disabled:
            capabilities.append(
                Capability(
                    category=category,
                    state="disabled",
                    coverage="none",
                    timing="not_applicable",
                    limitations=limitations.get(category, ()),
                )
            )
            continue
        if category in unavailable:
            capabilities.append(
                Capability(
                    category=category,
                    state="not_exposed",
                    coverage="none",
                    timing="not_available",
                    limitations=limitations.get(category, ()),
                )
            )
            continue
        if not category_evidence:
            capabilities.append(
                Capability(
                    category=category,
                    state="not_observed",
                    coverage="none",
                    timing="not_available",
                    limitations=limitations.get(category, ()),
                )
            )
            continue
        state, coverage, timing = characteristics[category]
        capabilities.append(
            Capability(
                category=category,
                state=state,
                coverage=coverage,
                timing=timing,
                evidence=category_evidence,
                limitations=limitations.get(category, ()),
            )
        )
    return tuple(capabilities)


class OpenHandsTraceAdapter:
    """Translate one OpenHands conversation's native callback events."""

    def __init__(
        self,
        recorder: TraceRecorder,
        *,
        session_id: str,
        delegation_enabled: bool,
        condenser_enabled: bool,
        browser_enabled: bool = False,
        completion_logs_enabled: bool = False,
        container_enabled: bool = False,
        evaluator_enabled: bool = False,
        started_at: datetime | None = None,
        agent_id: str = "coordinator",
        parent_agent_id: str | None = None,
        session_parent_span: str | None = None,
        capture_method: str = "native_hook",
        component: str = "openhands.sdk.conversation.callback",
    ) -> None:
        self._recorder = recorder
        self._session_id = session_id
        self._delegation_enabled = delegation_enabled
        self._condenser_enabled = condenser_enabled
        self._browser_enabled = browser_enabled
        self._completion_logs_enabled = completion_logs_enabled
        self._container_enabled = container_enabled
        self._evaluator_enabled = evaluator_enabled
        self._agent_id = agent_id
        self._parent_agent_id = parent_agent_id
        self._session_parent_span_override = session_parent_span
        self._capture_method = capture_method
        self._component = component
        self._lock = threading.RLock()
        self._seen_native_ids: set[str] = set()
        self._model_response_ids: set[str] = set()
        self._model_turn_count = 0
        self._pending_tools: dict[str, _PendingSpan] = {}
        self._tool_call_actions: dict[str, str] = {}
        self._pending_acp: dict[str, _PendingSpan] = {}
        self._pending_compaction: _PendingSpan | None = None
        self._pending_model: _PendingSpan | None = None
        self._observed: dict[str, set[str]] = {}
        self._delegations: dict[str, list[_DelegationLink]] = {}
        self._child_archive_ingested = False
        self._child_archive_complete = False
        self._ingested_child_sessions: set[str] = set()
        self._unlinked_child_sessions: set[str] = set()
        self._initial_started_at = started_at
        self._attempt_started_at: datetime | None = None
        self._startup_started_at: datetime | None = None
        self._execution_started_at: datetime | None = None
        self._execution_ended_at: datetime | None = None
        self._shutdown_started_at: datetime | None = None
        self._session_started_at: datetime | None = None
        self._finished: FinalizationResult | None = None

    @property
    def callback(self) -> Callable[[Event], None]:
        """Return the synchronous callback expected by OpenHands Conversation."""

        return self.__call__

    @property
    def identity(self) -> TraceIdentity:
        return self._recorder.identity

    def start(self) -> None:
        """Record instance and attempt lifecycle before benchmark setup."""

        with self._lock:
            if self._attempt_started_at is not None:
                return
            self._attempt_started_at = self._initial_started_at or datetime.now(UTC)
            instance_span = self._instance_span
            attempt_span = self._attempt_span
            self._record(
                event_type="instance.start",
                event_family="instance",
                phase="start",
                status="started",
                span_id=instance_span,
                occurred_at=self._attempt_started_at,
                payload={},
            )
            self._record(
                event_type="attempt.start",
                event_family="attempt",
                phase="start",
                status="started",
                span_id=attempt_span,
                parent_span_id=instance_span,
                occurred_at=self._attempt_started_at,
                payload={
                    "agent_configuration": {
                        "delegation_enabled": self._delegation_enabled,
                        "coordination_mode": "framework_native",
                        "delegation_sequence": (
                            ["navigator", "patcher", "reviewer"]
                            if self._delegation_enabled
                            else []
                        ),
                        "sequence_enforcement": "prompt_guided",
                    }
                },
            )
            self._startup_started_at = self._attempt_started_at
            self._record(
                event_type="harness.startup_start",
                event_family="harness",
                phase="start",
                status="started",
                span_id=self._startup_span,
                parent_span_id=attempt_span,
                occurred_at=self._attempt_started_at,
                payload={},
                timing={"fidelity": "derived"},
            )

    def start_execution(
        self,
        *,
        occurred_at: datetime | None = None,
    ) -> None:
        """Transition from coarse setup into detailed agent execution."""

        with self._lock:
            self.start()
            if self._execution_started_at is not None:
                return
            boundary = occurred_at or datetime.now(UTC)
            self._end_startup("completed", boundary)
            self._execution_started_at = boundary
            self._record(
                event_type="agent.execution_start",
                event_family="agent",
                phase="start",
                status="started",
                span_id=self._execution_span,
                parent_span_id=self._attempt_span,
                occurred_at=boundary,
                payload={"entered": True},
                timing={"fidelity": "derived"},
            )
            self.start_session(occurred_at=boundary)

    def start_session(
        self,
        *,
        occurred_at: datetime | None = None,
    ) -> None:
        """Record the agent session when the native conversation is constructed."""

        with self._lock:
            self.start()
            if self._execution_started_at is None:
                self.start_execution(occurred_at=occurred_at)
            if self._session_started_at is not None:
                return
            self._session_started_at = occurred_at or datetime.now(UTC)
            self._record(
                event_type="agent.session_start",
                event_family="agent",
                phase="start",
                status="started",
                span_id=self._session_span,
                parent_span_id=self._session_parent_span,
                occurred_at=self._session_started_at,
                payload={"native_session_id": self._session_id},
            )

    def end_execution(
        self,
        status: AdapterStatus,
        *,
        error_message: str | None = None,
        occurred_at: datetime | None = None,
    ) -> None:
        """Close detailed agent work and begin coarse shutdown."""

        with self._lock:
            if self._execution_ended_at is not None:
                return
            boundary = occurred_at or datetime.now(UTC)
            if self._execution_started_at is None:
                self._start_unentered_execution(status, boundary)
            self._close_incomplete_spans(boundary)
            if self._pending_model is not None:
                self._end_model(
                    status,
                    boundary,
                    "execution_boundary",
                )
            error = _lifecycle_error(status, error_message)
            if self._session_started_at is not None:
                self._record(
                    event_type="agent.session_end",
                    event_family="agent",
                    phase="end",
                    status=status,
                    span_id=self._session_span,
                    parent_span_id=self._session_parent_span,
                    occurred_at=boundary,
                    payload={"native_session_id": self._session_id},
                    error=error,
                    timing=_duration_timing(self._session_started_at, boundary),
                )
                self._session_started_at = None
            assert self._execution_started_at is not None
            self._record(
                event_type="agent.execution_end",
                event_family="agent",
                phase="end",
                status=status,
                span_id=self._execution_span,
                parent_span_id=self._attempt_span,
                occurred_at=boundary,
                payload={},
                error=error,
                timing=_duration_timing(self._execution_started_at, boundary),
            )
            self._execution_ended_at = boundary
            self._shutdown_started_at = boundary
            self._record(
                event_type="harness.shutdown_start",
                event_family="harness",
                phase="start",
                status="started",
                span_id=self._shutdown_span,
                parent_span_id=self._attempt_span,
                occurred_at=boundary,
                payload={},
                timing={"fidelity": "derived"},
            )

    def container_observed(
        self,
        metadata: JsonObject,
        *,
        occurred_at: datetime | None = None,
    ) -> None:
        """Record task-container metadata exposed by Harbor."""

        with self._lock:
            self._record(
                event_type="container.observed",
                event_family="container",
                phase="instant",
                status="completed",
                span_id=f"openhands-container-{self.identity.trace_id}",
                parent_span_id=self._lifecycle_parent_span,
                occurred_at=occurred_at or datetime.now(UTC),
                payload=metadata,
                timing={"fidelity": "native_wall"},
            )
            self._observe("container.lifecycle", "container.observed")

    def __call__(self, event: Event) -> None:
        """Handle one native callback without raising into OpenHands."""

        with self._lock:
            if isinstance(event, TokenEvent):
                return
            self.start_execution(occurred_at=_event_time(event.timestamp))
            self._ingest_native_event(event)

    def ingest_conversation_archive(self, path: Path | None) -> int:
        """Recover durable child-agent events before finalizing the attempt."""

        with self._lock:
            if self._finished is not None:
                raise RuntimeError(
                    "Cannot ingest child events after trace finalization"
                )
            if path is None:
                if self._delegations:
                    self._recorder.report_issue(
                        "openhands.child_archive_unavailable",
                        "Delegation was observed but no durable child archive was captured",
                        severity="warning",
                    )
                return 0
            try:
                conversations = _read_archived_subagents(path)
            except (OSError, tarfile.TarError, TypeError, ValueError):
                if self._delegations:
                    self._recorder.report_issue(
                        "openhands.child_archive_invalid",
                        "The OpenHands child conversation archive could not be read safely",
                        severity="warning",
                    )
                return 0

            return self._ingest_durable_conversations(
                conversations,
                source="archive",
            )

    def ingest_conversation_directory(self, path: Path) -> int:
        """Recover child events from a local OpenHands persistence directory."""

        with self._lock:
            if self._finished is not None:
                raise RuntimeError(
                    "Cannot ingest child events after trace finalization"
                )
            try:
                conversations = _read_persisted_subagents(path)
            except (OSError, TypeError, ValueError):
                if self._delegations:
                    self._recorder.report_issue(
                        "openhands.child_persistence_invalid",
                        "The OpenHands child persistence directory could not be read safely",
                        severity="warning",
                    )
                return 0
            return self._ingest_durable_conversations(
                conversations,
                source="persistence",
            )

    def _ingest_durable_conversations(
        self,
        conversations: tuple[_ArchivedConversation, ...],
        *,
        source: Literal["archive", "persistence"],
    ) -> int:
        self._child_archive_ingested = True
        for conversation in conversations:
            self._ingest_archived_conversation(conversation)
        self._child_archive_complete = (
            not self._delegations
            and not self._unlinked_child_sessions
            and all(conversation.complete for conversation in conversations)
        )
        if self._delegations or any(
            not conversation.complete for conversation in conversations
        ):
            self._recorder.report_issue(
                f"openhands.child_{source}_incomplete",
                (
                    "Delegation was observed but the durable source had no child events"
                    if not conversations
                    else "One or more durable child conversations lacked a complete "
                    "state and contiguous event sequence"
                ),
                severity="warning",
            )
        if self._unlinked_child_sessions:
            self._recorder.report_issue(
                f"openhands.child_{source}_unlinked",
                "One or more durable child conversations could not be linked "
                "to a parent delegation",
                severity="warning",
            )
        return len(conversations)

    def _ingest_archived_conversation(
        self,
        conversation: _ArchivedConversation,
    ) -> None:
        if (
            not conversation.events
            or conversation.session_id in self._ingested_child_sessions
        ):
            return
        self._ingested_child_sessions.add(conversation.session_id)
        link = self._match_delegation(conversation.events)
        if link is None:
            self._unlinked_child_sessions.add(conversation.session_id)
        child = OpenHandsTraceAdapter(
            self._recorder,
            session_id=conversation.session_id,
            delegation_enabled=self._delegation_enabled,
            condenser_enabled=self._condenser_enabled,
            browser_enabled=self._browser_enabled,
            completion_logs_enabled=self._completion_logs_enabled,
            container_enabled=self._container_enabled,
            evaluator_enabled=self._evaluator_enabled,
            agent_id=link.agent_id if link is not None else conversation.session_id,
            parent_agent_id=self._agent_id,
            session_parent_span=(
                link.span_id if link is not None else self._session_span
            ),
            capture_method="native_export",
            component="openhands.sdk.archived_subagent",
        )
        started_at = _event_time(conversation.events[0].timestamp)
        finished_at = max(
            (_event_time(event.timestamp) for event in conversation.events),
            default=started_at,
        )
        child._execution_started_at = started_at
        child._session_started_at = started_at
        child._record(
            event_type="agent.session_start",
            event_family="agent",
            phase="start",
            status="started",
            span_id=child._session_span,
            parent_span_id=child._session_parent_span,
            occurred_at=started_at,
            payload={
                "native_session_id": conversation.session_id,
                "recovered_from": "conversation_archive",
            },
            timing={"fidelity": "native_wall"},
        )
        for event in conversation.events:
            if isinstance(event, TokenEvent):
                continue
            child._ingest_native_event(event)
        child._close_incomplete_spans(finished_at)
        if child._pending_model is not None:
            child._end_model(
                conversation.status,
                finished_at,
                "archived_session_boundary",
            )
        child._record(
            event_type="agent.session_end",
            event_family="agent",
            phase="end",
            status=conversation.status,
            span_id=child._session_span,
            parent_span_id=child._session_parent_span,
            occurred_at=finished_at,
            payload={
                "native_session_id": conversation.session_id,
                "recovered_from": "conversation_archive",
            },
            timing=_duration_timing(started_at, finished_at),
            error=_lifecycle_error(conversation.status, None),
        )
        for category, event_types in child._observed.items():
            self._observed.setdefault(category, set()).update(event_types)

    def _match_delegation(
        self,
        events: tuple[Event, ...],
    ) -> _DelegationLink | None:
        for event in events:
            if not isinstance(event, MessageEvent) or event.source != "user":
                continue
            prompt = _message_text(event)
            links = self._delegations.get(prompt)
            if not links:
                continue
            match = links.pop(0)
            if not links:
                self._delegations.pop(prompt, None)
            return match
        return None

    def _ingest_native_event(self, event: Event) -> None:
        native_id = str(event.id)
        if native_id in self._seen_native_ids:
            return
        self._seen_native_ids.add(native_id)

        try:
            native = _serialize_event(event)
        except Exception:
            self._recorder.report_issue(
                "openhands.native_serialization_failed",
                "An OpenHands event could not be serialized",
            )
            return

        try:
            normalized = self._normalize(event, native)
        except Exception:
            self._recorder.report_issue(
                "openhands.normalization_failed",
                f"OpenHands event normalization failed for {type(event).__name__}",
            )
            issue = self._record(
                event_type="trace.issue",
                event_family="trace",
                phase="instant",
                status="degraded",
                span_id=f"openhands-event-{event.id}",
                parent_span_id=self._session_span,
                occurred_at=_event_time(event.timestamp),
                payload={"native_event_type": type(event).__name__},
            )
            normalized = (issue,) if issue is not None else ()

        try:
            retained = self._recorder.record_native(
                source=(
                    f"openhands.sdk.archive.{type(event).__name__}"
                    if self._capture_method == "native_export"
                    else f"openhands.sdk.callback.{type(event).__name__}"
                ),
                content=native,
                media_type="application/json",
                role="native.openhands.event",
                event_ids=tuple(item.event_id for item in normalized),
                native_record_id=(f"openhands-{self._session_id}-{native_id}"),
            )
            if retained is not None:
                for item in normalized:
                    self._observe("native.evidence", item.event_type)
        except Exception:
            self._recorder.report_issue(
                "openhands.native_record_failed",
                "An OpenHands native event could not be retained",
            )

    def finish(
        self,
        status: AdapterStatus,
        *,
        error_message: str | None = None,
    ) -> FinalizationResult:
        """Close lifecycle spans, finalize capabilities, and finalize the trace."""

        with self._lock:
            if self._finished is not None:
                return self._finished
            if self._attempt_started_at is None:
                self.start()
            finished_at = datetime.now(UTC)
            self.end_execution(
                status,
                error_message=error_message,
                occurred_at=finished_at,
            )
            error = _lifecycle_error(status, error_message)
            assert self._shutdown_started_at is not None
            self._record(
                event_type="harness.shutdown_end",
                event_family="harness",
                phase="end",
                status=status,
                span_id=self._shutdown_span,
                parent_span_id=self._attempt_span,
                occurred_at=finished_at,
                payload={},
                error=error,
                timing=_duration_timing(self._shutdown_started_at, finished_at),
            )
            assert self._attempt_started_at is not None
            self._record(
                event_type="attempt.end",
                event_family="attempt",
                phase="end",
                status=status,
                span_id=self._attempt_span,
                parent_span_id=self._instance_span,
                occurred_at=finished_at,
                payload={},
                error=error,
                timing=_duration_timing(self._attempt_started_at, finished_at),
            )
            self._record(
                event_type="instance.end",
                event_family="instance",
                phase="end",
                status=status,
                span_id=self._instance_span,
                occurred_at=finished_at,
                payload={},
                error=error,
                timing=_duration_timing(self._attempt_started_at, finished_at),
            )
            self._recorder.update_capabilities(self.capabilities())
            self._finished = self._recorder.finalize()
            return self._finished

    def capabilities(self) -> tuple[Capability, ...]:
        with self._lock:
            return openhands_capabilities(
                observed=self._observed,
                delegation_enabled=self._delegation_enabled,
                condenser_enabled=self._condenser_enabled,
                browser_enabled=self._browser_enabled,
                completion_logs_enabled=self._completion_logs_enabled,
                container_enabled=self._container_enabled,
                evaluator_enabled=self._evaluator_enabled,
                child_archive_ingested=self._child_archive_ingested,
                child_archive_complete=self._child_archive_complete,
            )

    @property
    def _instance_span(self) -> str:
        return f"instance-{self._recorder.identity.trace_id}"

    @property
    def _attempt_span(self) -> str:
        return f"attempt-{self._recorder.identity.trace_id}"

    @property
    def _startup_span(self) -> str:
        return f"openhands-startup-{self._recorder.identity.trace_id}"

    @property
    def _execution_span(self) -> str:
        return f"openhands-execution-{self._recorder.identity.trace_id}"

    @property
    def _shutdown_span(self) -> str:
        return f"openhands-shutdown-{self._recorder.identity.trace_id}"

    @property
    def _session_span(self) -> str:
        return f"openhands-session-{self._session_id}"

    @property
    def _session_parent_span(self) -> str:
        return self._session_parent_span_override or self._execution_span

    @property
    def _lifecycle_parent_span(self) -> str:
        if self._execution_started_at is None:
            return self._startup_span
        if self._execution_ended_at is None:
            return self._execution_span
        return self._shutdown_span

    def _end_startup(self, status: AdapterStatus, boundary: datetime) -> None:
        if self._startup_started_at is None:
            return
        self._record(
            event_type="harness.startup_end",
            event_family="harness",
            phase="end",
            status=status,
            span_id=self._startup_span,
            parent_span_id=self._attempt_span,
            occurred_at=boundary,
            payload={},
            error=_lifecycle_error(status, None),
            timing=_duration_timing(self._startup_started_at, boundary),
        )
        self._startup_started_at = None

    def _start_unentered_execution(
        self,
        status: AdapterStatus,
        boundary: datetime,
    ) -> None:
        self._end_startup(status, boundary)
        self._execution_started_at = boundary
        self._record(
            event_type="agent.execution_start",
            event_family="agent",
            phase="start",
            status="started",
            span_id=self._execution_span,
            parent_span_id=self._attempt_span,
            occurred_at=boundary,
            payload={"entered": False},
            timing={"fidelity": "derived"},
        )

    def _normalize(
        self,
        event: Event,
        native: JsonObject,
    ) -> tuple[_ObservedEvent, ...]:
        if isinstance(event, ActionEvent):
            return self._action(event, native)
        if isinstance(
            event, ObservationEvent | UserRejectObservation | AgentErrorEvent
        ):
            return self._observation(event, native)
        if isinstance(event, ACPToolCallEvent):
            return self._acp_tool(event, native)
        if isinstance(event, MessageEvent):
            return self._message(event, native)
        if isinstance(event, SystemPromptEvent):
            return self._artifact_event(
                event,
                native,
                event_type="agent.system_prompt",
                event_family="agent",
                role="agent.system_prompt",
            )
        if isinstance(event, CondensationRequest):
            return self._compaction_start(event)
        if isinstance(event, Condensation):
            return self._compaction_end(event, native)
        if isinstance(event, CondensationSummaryEvent):
            return self._artifact_event(
                event,
                native,
                event_type="context.compaction_summary",
                event_family="context",
                role="context.compaction_summary",
            )
        if isinstance(event, LLMCompletionLogEvent):
            return self._completion_log(event, native)
        if isinstance(event, StreamingDeltaEvent):
            return self._artifact_event(
                event,
                native,
                event_type="model.stream_delta",
                event_family="model",
                role="model.stream_delta",
            )
        if isinstance(event, HookExecutionEvent):
            return self._hook(event, native)
        if isinstance(event, ConversationStateUpdateEvent):
            return self._artifact_event(
                event,
                native,
                event_type="agent.state_update",
                event_family="agent",
                role="agent.state_update",
                payload={"key": event.key},
            )
        if isinstance(event, ConversationErrorEvent):
            observed = self._record(
                event_type="agent.error",
                event_family="agent",
                phase="instant",
                status="failed",
                span_id=f"openhands-event-{event.id}",
                parent_span_id=self._session_span,
                occurred_at=_event_time(event.timestamp),
                payload={"native_code": event.code},
                error={"code": "openhands.conversation", "message": event.detail},
            )
            return (observed,) if observed is not None else ()
        if isinstance(event, PauseEvent | InterruptEvent):
            observed = self._record(
                event_type=(
                    "agent.interrupt"
                    if isinstance(event, InterruptEvent)
                    else "agent.pause"
                ),
                event_family="agent",
                phase="instant",
                status="cancelled",
                span_id=f"openhands-event-{event.id}",
                parent_span_id=self._session_span,
                occurred_at=_event_time(event.timestamp),
                payload={},
            )
            return (observed,) if observed is not None else ()
        return self._artifact_event(
            event,
            native,
            event_type="agent.event",
            event_family="agent",
            role="native.openhands.unclassified",
            payload={"native_event_type": type(event).__name__},
        )

    def _action(
        self,
        event: ActionEvent,
        native: JsonObject,
    ) -> tuple[_ObservedEvent, ...]:
        observed = list(self._model_response(event, native))
        action = native.get("action")
        if not isinstance(action, dict):
            rejected = self._record(
                event_type="tool.rejected",
                event_family="tool",
                phase="instant",
                status="failed",
                span_id=f"openhands-event-{event.id}",
                parent_span_id=self._session_span,
                occurred_at=_event_time(event.timestamp),
                payload={
                    "tool": {
                        "name": event.tool_name,
                        "tool_call_id": event.tool_call_id,
                    }
                },
            )
            if rejected is not None:
                observed.append(rejected)
            return tuple(observed)

        start_type, end_type, family = _classify_tool(event.tool_name, action)
        artifact = self._recorder.store_json_artifact(action, role="tool.input")
        payload: JsonObject = {
            "tool": {
                "name": event.tool_name,
                "tool_call_id": event.tool_call_id,
                "arguments": _tool_arguments(action),
            },
            "native_action_id": str(event.id),
            "llm_response_id": str(event.llm_response_id),
        }
        started_at = _event_time(event.timestamp)
        span_id = f"openhands-action-{event.id}"
        started = self._record(
            event_type=start_type,
            event_family=family,
            phase="start",
            status="started",
            span_id=span_id,
            parent_span_id=self._session_span,
            occurred_at=started_at,
            payload=payload,
            artifacts=(artifact,) if artifact is not None else (),
        )
        if started is not None:
            observed.append(started)
            self._pending_tools[str(event.id)] = _PendingSpan(
                span_id=span_id,
                start_event_id=started.event_id,
                start_type=start_type,
                end_type=end_type,
                event_family=family,
                started_at=started_at,
                tool_name=event.tool_name,
            )
            self._tool_call_actions[event.tool_call_id] = str(event.id)
            if family == "delegation":
                prompt = action.get("prompt")
                subagent_type = action.get("subagent_type")
                if isinstance(prompt, str):
                    self._delegations.setdefault(prompt, []).append(
                        _DelegationLink(
                            prompt=prompt,
                            agent_id=(
                                subagent_type
                                if isinstance(subagent_type, str)
                                else "subagent"
                            ),
                            span_id=span_id,
                        )
                    )
        return tuple(observed)

    def _observation(
        self,
        event: ObservationEvent | UserRejectObservation | AgentErrorEvent,
        native: JsonObject,
    ) -> tuple[_ObservedEvent, ...]:
        action_id = (
            str(event.action_id)
            if isinstance(event, ObservationEvent | UserRejectObservation)
            else self._tool_call_actions.get(event.tool_call_id, "")
        )
        self._tool_call_actions.pop(event.tool_call_id, None)
        pending = self._pending_tools.pop(action_id, None)
        output = native.get("observation", native)
        artifact = self._recorder.store_json_artifact(
            output,
            role="tool.output",
        )
        status = _observation_status(event)
        payload: JsonObject = {
            "tool": {
                "name": event.tool_name,
                "tool_call_id": event.tool_call_id,
            },
            "native_action_id": action_id,
        }
        if isinstance(output, dict):
            for key in ("command", "exit_code", "timeout", "task_id", "subagent"):
                value = output.get(key)
                if isinstance(value, None | bool | int | float | str):
                    payload[key] = value
        error: JsonObject | None = (
            {
                "code": "openhands.tool_error",
                "message": "OpenHands reported an unsuccessful tool result",
            }
            if status != "completed"
            else None
        )
        if pending is None:
            start_type, _, family = _classify_tool(event.tool_name, {})
            event_type = "tool.result" if family == "tool" else f"{family}.result"
            if family == "file":
                event_type = start_type
            result = self._record(
                event_type=event_type,
                event_family=family,
                phase="instant",
                status=status,
                span_id=f"openhands-event-{event.id}",
                parent_span_id=self._session_span,
                occurred_at=_event_time(event.timestamp),
                payload=payload,
                artifacts=(artifact,) if artifact is not None else (),
                error=error,
            )
            self._resume_model_after_tools(_event_time(event.timestamp))
            return (result,) if result is not None else ()

        finished_at = _event_time(event.timestamp)
        result = self._record(
            event_type=pending.end_type,
            event_family=pending.event_family,
            phase="end",
            status=status,
            span_id=pending.span_id,
            parent_span_id=self._session_span,
            occurred_at=finished_at,
            payload=payload,
            artifacts=(artifact,) if artifact is not None else (),
            error=error,
            relations=({"type": "caused_by", "event_id": pending.start_event_id},),
            timing=_duration_timing(pending.started_at, finished_at),
        )
        self._resume_model_after_tools(finished_at)
        return (result,) if result is not None else ()

    def _model_response(
        self,
        event: ActionEvent | MessageEvent,
        native: JsonObject,
    ) -> tuple[_ObservedEvent, ...]:
        response_id = str(event.llm_response_id or event.id)
        if response_id in self._model_response_ids:
            return ()
        self._model_response_ids.add(response_id)
        finished_at = _event_time(event.timestamp)
        if self._pending_model is None:
            self._start_model(self._execution_started_at or finished_at)
        artifact = self._recorder.store_json_artifact(
            native,
            role="model.response",
        )
        result = self._end_model(
            "completed",
            finished_at,
            "native_response",
            payload={
                "native_response_id": response_id,
                "source_event_type": type(event).__name__,
            },
            artifacts=(artifact,) if artifact is not None else (),
        )
        return (result,) if result is not None else ()

    def _start_model(self, started_at: datetime) -> None:
        if self._pending_model is not None:
            return
        self._model_turn_count += 1
        span_id = (
            f"openhands-model-{self.identity.trace_id}-"
            f"{self._session_id}-{self._model_turn_count}"
        )
        result = self._record(
            event_type="model.turn_start",
            event_family="model",
            phase="start",
            status="started",
            span_id=span_id,
            parent_span_id=self._session_span,
            occurred_at=started_at,
            payload={"boundary": "agent_loop_ready"},
            timing={"fidelity": "derived"},
        )
        if result is None:
            return
        self._pending_model = _PendingSpan(
            span_id=span_id,
            start_event_id=result.event_id,
            start_type="model.turn_start",
            end_type="model.turn_end",
            event_family="model",
            started_at=started_at,
            tool_name="model",
        )

    def _end_model(
        self,
        status: AdapterStatus,
        finished_at: datetime,
        boundary: str,
        *,
        payload: JsonObject | None = None,
        artifacts: tuple[JsonObject, ...] = (),
    ) -> _ObservedEvent | None:
        pending = self._pending_model
        if pending is None:
            return None
        result = self._record(
            event_type=pending.end_type,
            event_family=pending.event_family,
            phase="end",
            status=status,
            span_id=pending.span_id,
            parent_span_id=self._session_span,
            occurred_at=finished_at,
            payload={"boundary": boundary, **(payload or {})},
            artifacts=artifacts,
            relations=({"type": "caused_by", "event_id": pending.start_event_id},),
            timing=_duration_timing(pending.started_at, finished_at),
        )
        self._pending_model = None
        return result

    def _resume_model_after_tools(self, started_at: datetime) -> None:
        if self._pending_tools or self._pending_acp:
            return
        self._start_model(started_at)

    def _message(
        self,
        event: MessageEvent,
        native: JsonObject,
    ) -> tuple[_ObservedEvent, ...]:
        if event.source == "agent":
            return self._model_response(event, native)
        observed = self._artifact_event(
            event,
            native,
            event_type="agent.message",
            event_family="agent",
            role="agent.message",
            payload={
                "source": str(event.source),
                **({"sender": event.sender} if event.sender else {}),
            },
        )
        self._start_model(_event_time(event.timestamp))
        return observed

    def _compaction_start(
        self,
        event: CondensationRequest,
    ) -> tuple[_ObservedEvent, ...]:
        started_at = _event_time(event.timestamp)
        observed: list[_ObservedEvent] = []
        model_end = self._end_model(
            "completed",
            started_at,
            "context_compaction",
        )
        if model_end is not None:
            observed.append(model_end)
        result = self._record(
            event_type="context.compaction_start",
            event_family="context",
            phase="start",
            status="started",
            span_id=f"openhands-compaction-{event.id}",
            parent_span_id=self._session_span,
            occurred_at=started_at,
            payload={},
        )
        if result is None:
            return tuple(observed)
        observed.append(result)
        self._pending_compaction = _PendingSpan(
            span_id=f"openhands-compaction-{event.id}",
            start_event_id=result.event_id,
            start_type="context.compaction_start",
            end_type="context.compaction_end",
            event_family="context",
            started_at=started_at,
            tool_name="condenser",
        )
        return tuple(observed)

    def _compaction_end(
        self,
        event: Condensation,
        native: JsonObject,
    ) -> tuple[_ObservedEvent, ...]:
        artifact = self._recorder.store_json_artifact(
            native,
            role="context.compaction",
        )
        finished_at = _event_time(event.timestamp)
        pending = self._pending_compaction
        self._pending_compaction = None
        if pending is None:
            result = self._record(
                event_type="context.compaction",
                event_family="context",
                phase="instant",
                status="completed",
                span_id=f"openhands-compaction-{event.id}",
                parent_span_id=self._session_span,
                occurred_at=finished_at,
                payload={"forgotten_event_count": len(event.forgotten_event_ids)},
                artifacts=(artifact,) if artifact is not None else (),
            )
            self._resume_model_after_tools(finished_at)
            return (result,) if result is not None else ()
        result = self._record(
            event_type=pending.end_type,
            event_family="context",
            phase="end",
            status="completed",
            span_id=pending.span_id,
            parent_span_id=self._session_span,
            occurred_at=finished_at,
            payload={"forgotten_event_count": len(event.forgotten_event_ids)},
            artifacts=(artifact,) if artifact is not None else (),
            relations=({"type": "caused_by", "event_id": pending.start_event_id},),
            timing=_duration_timing(pending.started_at, finished_at),
        )
        self._resume_model_after_tools(finished_at)
        return (result,) if result is not None else ()

    def _completion_log(
        self,
        event: LLMCompletionLogEvent,
        native: JsonObject,
    ) -> tuple[_ObservedEvent, ...]:
        content = native.get("log_data")
        artifact = self._recorder.store_json_artifact(
            content,
            role="provider.exchange",
        )
        failed = isinstance(content, dict) and "error" in content
        timing: JsonObject = {"fidelity": "not_available"}
        if isinstance(content, dict):
            latency = content.get("latency_sec")
            if isinstance(latency, int | float) and latency >= 0:
                timing = {
                    "fidelity": "native_wall",
                    "duration_ms": float(latency) * 1000,
                }
        result = self._record(
            event_type="provider.response",
            event_family="provider",
            phase="instant",
            status="failed" if failed else "completed",
            span_id=f"openhands-provider-{event.id}",
            parent_span_id=self._session_span,
            occurred_at=_event_time(event.timestamp),
            payload={
                "model": event.model_name,
                "usage_id": event.usage_id,
                "filename": event.filename,
            },
            artifacts=(artifact,) if artifact is not None else (),
            error=(
                {
                    "code": "openhands.provider_error",
                    "message": "OpenHands completion telemetry reported an error",
                }
                if failed
                else None
            ),
            timing=timing,
        )
        return (result,) if result is not None else ()

    def _hook(
        self,
        event: HookExecutionEvent,
        native: JsonObject,
    ) -> tuple[_ObservedEvent, ...]:
        artifact = self._recorder.store_json_artifact(
            native,
            role="tool.hook",
        )
        result = self._record(
            event_type="tool.hook",
            event_family="tool",
            phase="instant",
            status=(
                "completed"
                if event.success and not event.blocked
                else "cancelled"
                if event.blocked
                else "failed"
            ),
            span_id=f"openhands-hook-{event.id}",
            parent_span_id=self._session_span,
            occurred_at=_event_time(event.timestamp),
            payload={
                "hook_event_type": event.hook_event_type,
                "tool_name": event.tool_name,
                "exit_code": event.exit_code,
            },
            artifacts=(artifact,) if artifact is not None else (),
        )
        return (result,) if result is not None else ()

    def _acp_tool(
        self,
        event: ACPToolCallEvent,
        native: JsonObject,
    ) -> tuple[_ObservedEvent, ...]:
        key = str(event.tool_call_id)
        terminal = bool(
            event.is_error
            or event.raw_output is not None
            or str(event.status).lower()
            in {"completed", "failed", "cancelled", "timeout"}
        )
        if key not in self._pending_acp and not terminal:
            action = event.raw_input if isinstance(event.raw_input, dict) else {}
            start_type, end_type, family = _classify_tool(event.title, action)
            artifact = self._recorder.store_json_artifact(
                event.raw_input,
                role="tool.input",
            )
            started_at = _event_time(event.timestamp)
            observed: list[_ObservedEvent] = []
            model_end = self._end_model(
                "completed",
                started_at,
                "tool_start",
            )
            if model_end is not None:
                observed.append(model_end)
            result = self._record(
                event_type=start_type,
                event_family=family,
                phase="start",
                status="started",
                span_id=f"openhands-acp-{key}",
                parent_span_id=self._session_span,
                occurred_at=started_at,
                payload={
                    "tool": {
                        "name": event.title,
                        "tool_call_id": key,
                        "arguments": _tool_arguments(action),
                    },
                    "tool_kind": event.tool_kind,
                },
                artifacts=(artifact,) if artifact is not None else (),
            )
            if result is None:
                return tuple(observed)
            observed.append(result)
            self._pending_acp[key] = _PendingSpan(
                span_id=f"openhands-acp-{key}",
                start_event_id=result.event_id,
                start_type=start_type,
                end_type=end_type,
                event_family=family,
                started_at=started_at,
                tool_name=event.title,
            )
            return tuple(observed)
        if not terminal:
            return self._artifact_event(
                event,
                native,
                event_type="tool.progress",
                event_family="tool",
                role="tool.progress",
                payload={"tool_call_id": key, "status": event.status},
            )

        pending = self._pending_acp.pop(key, None)
        artifact = self._recorder.store_json_artifact(
            event.raw_output if event.raw_output is not None else native,
            role="tool.output",
        )
        status = _acp_status(event)
        finished_at = _event_time(event.timestamp)
        if pending is None:
            result = self._record(
                event_type="tool.result",
                event_family="tool",
                phase="instant",
                status=status,
                span_id=f"openhands-acp-{key}",
                parent_span_id=self._session_span,
                occurred_at=finished_at,
                payload={"tool_call_id": key, "status": event.status},
                artifacts=(artifact,) if artifact is not None else (),
            )
            self._resume_model_after_tools(finished_at)
            return (result,) if result is not None else ()
        result = self._record(
            event_type=pending.end_type,
            event_family=pending.event_family,
            phase="end",
            status=status,
            span_id=pending.span_id,
            parent_span_id=self._session_span,
            occurred_at=finished_at,
            payload={"tool_call_id": key, "status": event.status},
            artifacts=(artifact,) if artifact is not None else (),
            relations=({"type": "caused_by", "event_id": pending.start_event_id},),
            timing=_duration_timing(pending.started_at, finished_at),
        )
        self._resume_model_after_tools(finished_at)
        return (result,) if result is not None else ()

    def _artifact_event(
        self,
        event: Event,
        native: JsonObject,
        *,
        event_type: str,
        event_family: str,
        role: str,
        payload: JsonObject | None = None,
    ) -> tuple[_ObservedEvent, ...]:
        artifact = self._recorder.store_json_artifact(native, role=role)
        result = self._record(
            event_type=event_type,
            event_family=event_family,
            phase="instant",
            status="completed",
            span_id=f"openhands-event-{event.id}",
            parent_span_id=self._session_span,
            occurred_at=_event_time(event.timestamp),
            payload=payload or {},
            artifacts=(artifact,) if artifact is not None else (),
        )
        return (result,) if result is not None else ()

    def _record(
        self,
        *,
        event_type: str,
        event_family: str,
        phase: Literal["start", "end", "instant"],
        status: str,
        span_id: str,
        occurred_at: datetime,
        payload: JsonObject,
        parent_span_id: str | None = None,
        artifacts: tuple[JsonObject, ...] = (),
        error: JsonObject | None = None,
        relations: tuple[JsonObject, ...] = (),
        timing: JsonObject | None = None,
    ) -> _ObservedEvent | None:
        harness_owned = event_family in {
            "instance",
            "attempt",
            "harness",
            "container",
            "evaluator",
        } or event_type.startswith("agent.execution")
        event_id = self._recorder.record_event(
            event_type=event_type,
            event_family=event_family,
            phase=phase,
            status=status,
            span_id=span_id,
            session_id=None if harness_owned else self._session_id,
            agent_id=None if harness_owned else self._agent_id,
            parent_agent_id=(None if harness_owned else self._parent_agent_id),
            parent_span_id=parent_span_id,
            occurred_at=occurred_at,
            origin={
                "component": (
                    "openhands.benchmark.harness" if harness_owned else self._component
                ),
                "capture_method": (
                    self._capture_method
                    if not harness_owned and not event_type.startswith("agent.session")
                    else "derived"
                ),
            },
            timing=timing or {"fidelity": "not_available"},
            payload=payload,
            artifacts=artifacts,
            error=error,
            relations=relations,
        )
        if event_id is None:
            return None
        self._observe_for_event(event_type, event_family, phase)
        return _ObservedEvent(event_id=event_id, event_type=event_type)

    def _observe_for_event(
        self,
        event_type: str,
        event_family: str,
        phase: str,
    ) -> None:
        if event_type.startswith("agent.session"):
            self._observe("agent.session", event_type)
        if event_family == "model":
            self._observe("model.turn", event_type)
        if event_family == "provider":
            self._observe("provider.exchange", event_type)
        if event_family in {
            "tool",
            "shell",
            "file",
            "search",
            "browser",
            "delegation",
        }:
            self._observe(
                "tool.invocation" if phase == "start" else "tool.result",
                event_type,
            )
            if phase == "end":
                self._observe("tool.timing", event_type)
        if event_family in {"shell", "file", "search", "browser", "delegation"}:
            self._observe(event_family, event_type)
        if event_family == "context":
            self._observe("context.compaction", event_type)
        if event_family == "harness":
            self._observe("harness.lifecycle", event_type)
        if event_family == "patch":
            self._observe("patch", event_type)

    def _observe(self, category: str, event_type: str) -> None:
        self._observed.setdefault(category, set()).add(event_type)

    def _close_incomplete_spans(self, finished_at: datetime) -> None:
        pending = [
            *self._pending_tools.values(),
            *self._pending_acp.values(),
            *((self._pending_compaction,) if self._pending_compaction else ()),
        ]
        if not pending:
            return
        self._recorder.report_issue(
            "openhands.incomplete_native_span",
            f"{len(pending)} OpenHands activities ended without a native result",
            severity="warning",
        )
        for span in pending:
            self._record(
                event_type=span.end_type,
                event_family=span.event_family,
                phase="end",
                status="degraded",
                span_id=span.span_id,
                parent_span_id=self._session_span,
                occurred_at=finished_at,
                payload={
                    "tool": {"name": span.tool_name},
                    "closure": "adapter_finalization",
                },
                relations=({"type": "caused_by", "event_id": span.start_event_id},),
                timing=_duration_timing(span.started_at, finished_at),
            )
        self._pending_tools.clear()
        self._tool_call_actions.clear()
        self._pending_acp.clear()
        self._pending_compaction = None


def _read_archived_subagents(path: Path) -> tuple[_ArchivedConversation, ...]:
    events: dict[str, list[tuple[int, Event]]] = {}
    statuses: dict[str, AdapterStatus] = {}
    sessions: set[str] = set()
    total_bytes = 0
    event_count = 0
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if "subagents" not in parts:
                continue
            index = parts.index("subagents")
            if len(parts) <= index + 2 or not member.isfile():
                continue
            session_id = parts[index + 1]
            if not session_id or len(session_id) > 512:
                raise ValueError("Invalid archived child session identity")
            if session_id not in sessions:
                if len(sessions) >= 128:
                    raise ValueError(
                        "OpenHands child archive exceeds the session limit"
                    )
                sessions.add(session_id)

            relative = parts[index + 2 :]
            if relative == ("base_state.json",):
                total_bytes += member.size
                if total_bytes > 512 * 1024 * 1024:
                    raise ValueError(
                        "OpenHands child archive exceeds the evidence limit"
                    )
                document = _read_archive_json(archive, member, 8 * 1024 * 1024)
                statuses[session_id] = _archived_status(
                    document.get("execution_status")
                )
                continue
            if (
                len(relative) != 2
                or relative[0] != "events"
                or not relative[1].startswith("event-")
                or not relative[1].endswith(".json")
            ):
                continue

            total_bytes += member.size
            if total_bytes > 512 * 1024 * 1024:
                raise ValueError("OpenHands child archive exceeds the evidence limit")
            document = _read_archive_json(archive, member, 64 * 1024 * 1024)
            events.setdefault(session_id, []).append(
                (_event_file_sequence(relative[1]), Event.model_validate(document))
            )
            event_count += 1
            if event_count > 100_000:
                raise ValueError("OpenHands child archive exceeds the event limit")

    conversations = []
    for session_id, values in sorted(events.items()):
        ordered = sorted(values)
        conversations.append(
            _ArchivedConversation(
                session_id=session_id,
                status=statuses.get(session_id, "degraded"),
                events=tuple(event for _, event in ordered),
                complete=session_id in statuses
                and all(
                    sequence == index for index, (sequence, _) in enumerate(ordered)
                ),
            )
        )
    return tuple(conversations)


def _read_persisted_subagents(path: Path) -> tuple[_ArchivedConversation, ...]:
    root = path.expanduser().resolve()
    if not root.is_dir() or path.is_symlink():
        raise ValueError("OpenHands persistence root is not a real directory")
    conversations = []
    total_bytes = 0
    event_count = 0
    session_count = 0
    session_ids: set[str] = set()
    for child in sorted(root.rglob("subagents/*")):
        if not child.is_dir() or child.is_symlink() or child.parent.name != "subagents":
            continue
        session_count += 1
        if session_count > 128:
            raise ValueError("OpenHands child persistence exceeds the session limit")
        session_id = child.name
        if not session_id or len(session_id) > 512:
            raise ValueError("Invalid persisted child session identity")
        if session_id in session_ids:
            raise ValueError("Duplicate persisted child session identity")
        session_ids.add(session_id)
        base_state = child / "base_state.json"
        base_state_available = base_state.is_file() and not base_state.is_symlink()
        status: AdapterStatus = "degraded"
        if base_state_available:
            total_bytes += base_state.stat().st_size
            if total_bytes > 512 * 1024 * 1024:
                raise ValueError(
                    "OpenHands child persistence exceeds the evidence limit"
                )
            status = _archived_status(
                _read_json_file(base_state, 8 * 1024 * 1024).get("execution_status")
            )
        event_dir = child / "events"
        if not event_dir.is_dir() or event_dir.is_symlink():
            continue
        persisted_events = []
        for event_path in sorted(event_dir.glob("event-*.json")):
            if not event_path.is_file() or event_path.is_symlink():
                raise ValueError("OpenHands child event is not a regular file")
            total_bytes += event_path.stat().st_size
            if total_bytes > 512 * 1024 * 1024:
                raise ValueError(
                    "OpenHands child persistence exceeds the evidence limit"
                )
            persisted_events.append(
                (
                    _event_file_sequence(event_path.name),
                    Event.model_validate(_read_json_file(event_path, 64 * 1024 * 1024)),
                )
            )
            event_count += 1
            if event_count > 100_000:
                raise ValueError("OpenHands child persistence exceeds the event limit")
        ordered = sorted(persisted_events)
        conversations.append(
            _ArchivedConversation(
                session_id=session_id,
                status=status,
                events=tuple(event for _, event in ordered),
                complete=base_state_available
                and all(
                    sequence == index for index, (sequence, _) in enumerate(ordered)
                ),
            )
        )
    return tuple(conversations)


def _read_archive_json(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    maximum_bytes: int,
) -> JsonObject:
    if member.size < 0 or member.size > maximum_bytes:
        raise ValueError("OpenHands archive member exceeds the size limit")
    source = archive.extractfile(member)
    if source is None:
        raise ValueError("OpenHands archive member is unreadable")
    value = json.loads(source.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("OpenHands archive member is not a JSON object")
    return value


def _read_json_file(path: Path, maximum_bytes: int) -> JsonObject:
    size = path.stat().st_size
    if size < 0 or size > maximum_bytes:
        raise ValueError("OpenHands persistence member exceeds the size limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("OpenHands persistence member is not a JSON object")
    return value


def _event_file_sequence(name: str) -> int:
    try:
        return int(name.split("-", 2)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError("Invalid OpenHands child event sequence") from exc


def _archived_status(value: object) -> AdapterStatus:
    status = str(value or "").lower()
    if status in {"finished", "idle", "stopped"}:
        return "completed"
    if status in {"paused", "cancelled"}:
        return "cancelled"
    if status == "error":
        return "failed"
    return "degraded"


def _message_text(event: MessageEvent) -> str:
    return "".join(
        content.text
        for content in event.llm_message.content
        if isinstance(content, TextContent)
    )


def _serialize_event(event: Event) -> JsonObject:
    value = json.loads(event.model_dump_json(exclude_none=True))
    if not isinstance(value, dict):
        raise TypeError("OpenHands event did not serialize to an object")
    if isinstance(event, LLMCompletionLogEvent):
        try:
            value["log_data"] = json.loads(event.log_data)
        except json.JSONDecodeError:
            value["log_data"] = event.log_data
    return value


def _event_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(UTC)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _duration_timing(started_at: datetime, finished_at: datetime) -> JsonObject:
    return {
        "fidelity": "derived",
        "duration_ms": max(
            0.0,
            (finished_at - started_at).total_seconds() * 1000,
        ),
    }


def _lifecycle_error(
    status: AdapterStatus,
    error_message: str | None,
) -> JsonObject | None:
    if status == "completed":
        return None
    return {
        "code": "agent.session_failed",
        "message": error_message or "OpenHands session did not complete",
    }


def _classify_tool(
    tool_name: str,
    action: Mapping[str, JsonValue],
) -> tuple[str, str, str]:
    normalized = tool_name.lower().replace("-", "_")
    command = action.get("command")
    command_name = str(command).lower() if isinstance(command, str) else ""
    if "terminal" in normalized or normalized in {"bash", "shell"}:
        return "shell.start", "shell.end", "shell"
    if any(name in normalized for name in ("grep", "glob", "search")):
        return "search.start", "search.end", "search"
    if "browser" in normalized:
        return "browser.start", "browser.end", "browser"
    if normalized in {"task", "delegate"} or any(
        name in normalized for name in ("task_tool", "delegate")
    ):
        return "delegation.start", "delegation.end", "delegation"
    if (
        "apply_patch" in normalized
        or normalized in {"edit", "write_file"}
        or command_name in {"create", "str_replace", "insert", "undo_edit"}
    ):
        return "file.patch", "file.patch", "file"
    if (
        normalized in {"read_file", "list_directory", "file_editor"}
        or "file" in normalized
    ):
        if command_name == "view" or normalized in {"read_file", "list_directory"}:
            return "file.read", "file.read", "file"
        return "file.write", "file.write", "file"
    return "tool.start", "tool.end", "tool"


def _tool_arguments(action: Mapping[str, JsonValue]) -> JsonObject:
    keys = (
        "command",
        "path",
        "pattern",
        "include",
        "description",
        "subagent_type",
        "resume",
        "timeout",
        "is_input",
    )
    return {key: action[key] for key in keys if key in action}


def _observation_status(
    event: ObservationEvent | UserRejectObservation | AgentErrorEvent,
) -> AdapterStatus:
    if isinstance(event, UserRejectObservation):
        return "cancelled"
    if isinstance(event, AgentErrorEvent):
        return "failed"
    observation = event.observation
    if bool(getattr(observation, "timeout", False)):
        return "timeout"
    if bool(getattr(observation, "is_error", False)):
        return "failed"
    exit_code = getattr(observation, "exit_code", None)
    if isinstance(exit_code, int) and exit_code not in {0, -1}:
        return "failed"
    return "completed"


def _acp_status(event: ACPToolCallEvent) -> AdapterStatus:
    value = str(event.status or "").lower()
    if event.is_error or value == "failed":
        return "failed"
    if value == "cancelled":
        return "cancelled"
    if value == "timeout":
        return "timeout"
    return "completed"
