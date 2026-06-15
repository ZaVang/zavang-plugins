"""Orchestration — 通用调度器。

系统最高层级，也是最简单的。极度简单以确保可靠性。
只做一件事：确保session_id被某个Harness处理着。
"""

from __future__ import annotations

import uuid
from typing import Any, Optional, Tuple

from .event import Event
from .harness import Harness


class Orchestration:
    """通用调度器。不持有状态，不做业务判断。

    越高层越简单越不会坏——Orchestration是保证
    其他所有组件能被恢复的最后防线。
    """

    def __init__(self, harness: Harness) -> None:
        self.harness = harness

    async def wake(self, session_id: Optional[str] = None) -> Tuple[str, str]:
        """确保一个session正在被Harness处理。

        如果session_id为None，创建新session。
        如果Harness执行失败，可重试（起一个新Harness实例）。

        返回 (session_id, wake_status)。
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
            await self.harness.session.create_session(session_id)
        status = await self.harness.wake(session_id)
        return session_id, status

    async def trigger(
        self,
        session_id: Optional[str] = None,
        *,
        trigger_kind: str = "manual",
        trigger_metadata: Optional[dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """外部事件触发（人工输入/突发新闻/定时任务）。"""
        resolved_session_id, status = await self.wake(session_id)
        await self.harness.session.emit(
            Event(
                session_id=resolved_session_id,
                event_type="orchestration_triggered",
                component="orchestration",
                payload_out={
                    "trigger_kind": trigger_kind,
                    "trigger_metadata": trigger_metadata or {},
                    "wake_status": status,
                },
            )
        )
        return resolved_session_id, status
