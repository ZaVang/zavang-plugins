"""Harness — 系统中枢，Effect循环。

Harness不直接调用任何组件，而是yield Effect声明意图，
由基础设施执行后喂回结果。每个yield点自动记录事件到Session。
"""

from __future__ import annotations

import logging
import uuid
from pydantic import BaseModel
from typing import Any, Optional, AsyncGenerator, Generic, TypeVar

from .event import Event, SessionBase
from .gateway import GatewayBase
from .sandbox import SandboxBase
from .context import HarnessRuntimeContext, _set_harness_context, _set_harness_runtime_context

T = TypeVar("T")

logger = logging.getLogger(__name__)


# ── Effect 类型定义 ──


class Effect(BaseModel, Generic[T]):
    """所有Effect的基类。Harness通过yield Effect声明意图。"""


class ExecuteToolEffect(Effect[dict]):
    """请求Sandbox执行一个工具。"""

    tool_name: str
    payload: dict[str, Any]
    idempotency_key: Optional[str] = None
    correlation_id: Optional[str] = None
    parent_correlation_id: Optional[str] = None
    attempt: int = 1
    attempt_metadata: Optional[dict[str, Any]] = None


class EmitEventEffect(Effect[None]):
    """请求向Session写入一条事件。"""

    event: Event


class GetEventsEffect(Effect[list[Event]]):
    """请求从Session读取事件流。"""

    session_id: str
    since: Optional[str] = None


# ── Harness 本体 ──


class Harness:
    """系统中枢。不含业务逻辑，不知道"预测"是什么。

    职责：
    1. 被Orchestration指派处理一个session（wake）
    2. 运行Pipeline的Effect循环，自动记录事件到Session
    3. 通过Sandbox执行工具调用
    4. 处理错误恢复和重试

    resume_policy 控制崩溃在工具中途（留下孤儿 tool_request）时 wake 的行为：
      - "replay"（默认）：判为 resumable。下一次 run_effect_loop 从头重放 Pipeline，
        已完成的工具靠 idempotency_key 短路跳过，未完成的那个被重新执行。
        ⚠️ 仅当 Pipeline 为工具设置了**稳定的 idempotency_key** 时才安全——
        否则重放会重复触发副作用。
      - "unrecoverable"：判为 unrecoverable，停在原地等人工介入（保守，副作用安全）。
    """

    def __init__(
        self,
        session: SessionBase,
        sandbox: SandboxBase,
        gateway: Optional[GatewayBase] = None,
        *,
        resume_policy: str = "replay",
    ) -> None:
        self.session = session
        self.sandbox = sandbox
        self.gateway = gateway
        self._session_context: dict = {}  # session级别上下文（附加到每个 tool_response.metadata）
        if resume_policy not in ("replay", "unrecoverable"):
            raise ValueError(f"Unknown resume_policy: {resume_policy!r}")
        self._resume_policy = resume_policy

    def set_session_context(self, context: dict) -> None:
        """设置 session 级别的上下文，将附加到每个 tool_response 事件的 metadata 中。"""
        self._session_context = dict(context)

    async def wake(self, session_id: str) -> str:
        """被Orchestration指派处理一个session。"""
        existing_events = await self.session.get_events(session_id)
        if not existing_events:
            logger.info("Harness.wake: 新 session %s", session_id)
            return "new"

        logger.info(
            "Harness.wake: 恢复 session %s, 已有 %d 个事件",
            session_id,
            len(existing_events),
        )
        event_types = {e.event_type for e in existing_events}
        if "pipeline_end" in event_types or "session_completed" in event_types:
            logger.info("Harness.wake: session %s 已完成，跳过", session_id)
            return "completed"
        if "session_unrecoverable" in event_types:
            logger.info("Harness.wake: session %s 已标记为不可恢复", session_id)
            return "unrecoverable"

        incomplete = self._find_incomplete_effect(existing_events)
        if incomplete is not None:
            # [FIX Bug-2] 崩在工具中途是最常见的崩溃点。旧实现一律判 unrecoverable，
            # 等于把最该救活的 session 判死。默认改为 replay：发 session_resumable，
            # 让 run_effect_loop 从头重放、靠幂等键跳过已完成步骤。
            if self._resume_policy == "replay":
                await self.session.emit(
                    Event(
                        session_id=session_id,
                        event_type="session_resumable",
                        component="harness",
                        idempotency_key=(
                            f"{session_id}:wake:{incomplete.event_id}:resumable"
                        ),
                        correlation_id=incomplete.correlation_id,
                        payload_out={
                            "last_event_id": incomplete.event_id,
                            "strategy": "replay_incomplete_effect",
                            "incomplete_correlation_id": incomplete.correlation_id,
                        },
                    )
                )
                logger.info(
                    "Harness.wake: session %s 有未完成 effect，将重放", session_id
                )
                return "resumable"
            await self._mark_unrecoverable(
                session_id,
                reason="incomplete_effect",
                correlation_id=incomplete.correlation_id,
                last_event_id=incomplete.event_id,
            )
            return "unrecoverable"

        last_event = existing_events[-1]
        await self.session.emit(
            Event(
                session_id=session_id,
                event_type="session_resumable",
                component="harness",
                idempotency_key=f"{session_id}:wake:{last_event.event_id}:resumable",
                correlation_id=last_event.correlation_id,
                payload_out={
                    "last_event_id": last_event.event_id,
                    "strategy": "resume_from_last_event",
                },
            )
        )
        logger.info("Harness.wake: session %s 可从最近事件恢复", session_id)
        return "resumable"

    async def run_effect_loop(
        self,
        session_id: str,
        effects: AsyncGenerator[Effect, Any],
    ) -> None:
        """运行一个Effect循环。

        遍历Pipeline yield出来的Effect，逐个执行并喂回结果。
        每个Effect执行前后自动emit事件到Session。
        """
        result = None
        try:
            effect = await effects.asend(None)  # 启动generator
            while True:
                try:
                    result = await self._handle_effect(session_id, effect)
                except Exception as exc:
                    effect = await effects.athrow(exc)
                    continue
                effect = await effects.asend(result)
        except StopAsyncIteration:
            pass

    async def _handle_effect(self, session_id: str, effect: Effect) -> Any:
        """处理单个Effect。根据类型分发到对应的基础设施。"""
        if isinstance(effect, ExecuteToolEffect):
            return await self._handle_execute_tool(session_id, effect)
        elif isinstance(effect, EmitEventEffect):
            event = effect.event
            if event.session_id != session_id:
                event = event.model_copy(update={"session_id": session_id})
            await self.session.emit(event)
            return None
        elif isinstance(effect, GetEventsEffect):
            return await self.session.get_events(
                effect.session_id, since=effect.since
            )
        else:
            raise TypeError(f"Unknown effect type: {type(effect)}")

    async def _handle_execute_tool(
        self,
        session_id: str,
        effect: ExecuteToolEffect,
        parent_correlation_id: Optional[str] = None,
    ) -> Any:
        """执行工具调用，自动记录事件。"""
        correlation_id = effect.correlation_id or str(uuid.uuid4())
        parent_id = parent_correlation_id or effect.parent_correlation_id
        request_key = (
            f"{effect.idempotency_key}:request"
            if effect.idempotency_key is not None
            else None
        )
        response_key = (
            f"{effect.idempotency_key}:response"
            if effect.idempotency_key is not None
            else None
        )
        error_key = (
            f"{effect.idempotency_key}:error"
            if effect.idempotency_key is not None
            else None
        )

        if response_key is not None:
            # [FIX Bug-3] 走 session 的 find_by_idempotency_key（SQLite 走索引 O(log n)），
            # 不再每次工具调用都全量 get_events 线性扫，消除长 session 的 O(n²) 退化。
            existing_response = await self.session.find_by_idempotency_key(
                session_id, response_key
            )
            if existing_response is not None:
                # [FIX Bug-4] payload_out 现在存的是完整结果（见下），重放返回真数据。
                return existing_response.payload_out or {}

        await self.session.emit(
            Event(
                session_id=session_id,
                event_type="tool_request",
                component=effect.tool_name,
                idempotency_key=request_key,
                correlation_id=correlation_id,
                parent_correlation_id=parent_id,
                attempt=effect.attempt,
                attempt_metadata=effect.attempt_metadata,
                payload_in=effect.payload,
            )
        )

        try:
            token = _set_harness_context(session_id, self.session)
            runtime = HarnessRuntimeContext(
                session_id=session_id,
                session=self.session,
                execute_tool=(
                    lambda name, payload, idem, corr, attempt, attempt_metadata:
                    self._handle_execute_tool(
                        session_id,
                        ExecuteToolEffect(
                            tool_name=name,
                            payload=payload,
                            idempotency_key=idem,
                            correlation_id=corr,
                            attempt=attempt,
                            attempt_metadata=attempt_metadata,
                        ),
                        parent_correlation_id=correlation_id,
                    )
                ),
                correlation_id=correlation_id,
            )
            runtime_token = _set_harness_runtime_context(runtime)
            try:
                result = await self.sandbox.execute(effect.tool_name, effect.payload)
            finally:
                from .context import _current, _runtime
                _runtime.reset(runtime_token)
                _current.reset(token)

            metadata: dict = dict(self._session_context)
            if isinstance(result, dict) and "thinking" in result:
                metadata["thinking"] = result.get("thinking")
            # [FIX Bug-4] 可选的 "__log__" 摘要只作为展示用 metadata，绝不取代真结果。
            # 旧实现把摘要写进 payload_out，导致幂等重放命中后返回的是摘要而非真数据。
            if isinstance(result, dict) and "__log__" in result:
                metadata["log_summary"] = result.get("__log__")

            await self.session.emit(
                Event(
                    session_id=session_id,
                    event_type="tool_response",
                    component=effect.tool_name,
                    idempotency_key=response_key,
                    correlation_id=correlation_id,
                    parent_correlation_id=parent_id,
                    attempt=effect.attempt,
                    attempt_metadata=effect.attempt_metadata,
                    # [FIX Bug-4] 持久化完整结果，作为幂等重放的权威来源。
                    payload_out=result if isinstance(result, dict) else {"value": result},
                    metadata=metadata if metadata else None,
                )
            )
            return result

        except Exception as e:
            error_metadata = _exception_event_metadata(e)
            await self.session.emit(
                Event(
                    session_id=session_id,
                    event_type="tool_error",
                    component=effect.tool_name,
                    idempotency_key=error_key,
                    correlation_id=correlation_id,
                    parent_correlation_id=parent_id,
                    attempt=effect.attempt,
                    attempt_metadata=effect.attempt_metadata,
                    payload_in=effect.payload,
                    metadata=error_metadata if error_metadata else None,
                    error=str(e),
                )
            )
            raise

    @staticmethod
    def _find_incomplete_effect(events: list[Event]) -> Optional[Event]:
        requests: dict[str, Event] = {}
        completed: set[str] = set()
        for event in events:
            if event.event_type in {"tool_request", "llm_request"}:
                if event.correlation_id is None:
                    continue
                key = event.correlation_id or event.event_id
                requests[key] = event
            elif event.event_type in {
                "tool_response",
                "tool_error",
                "llm_response",
                "llm_error",
            }:
                if event.correlation_id is None:
                    continue
                key = event.correlation_id or event.event_id
                completed.add(key)

        for key, event in reversed(list(requests.items())):
            if key not in completed:
                return event
        return None

    async def _mark_unrecoverable(
        self,
        session_id: str,
        *,
        reason: str,
        correlation_id: Optional[str],
        last_event_id: str,
    ) -> None:
        await self.session.emit(
            Event(
                session_id=session_id,
                event_type="session_unrecoverable",
                component="harness",
                idempotency_key=f"{session_id}:wake:{last_event_id}:unrecoverable",
                correlation_id=correlation_id,
                payload_out={
                    "reason": reason,
                    "correlation_id": correlation_id,
                    "last_event_id": last_event_id,
                },
            )
        )


def _exception_event_metadata(exc: Exception) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    kind = getattr(exc, "kind", None)
    if kind is not None:
        metadata["exception_kind"] = getattr(kind, "value", str(kind))
    exception_metadata = getattr(exc, "metadata", None)
    if isinstance(exception_metadata, dict) and exception_metadata:
        metadata["exception_metadata"] = exception_metadata
    return metadata
