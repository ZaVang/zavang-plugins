"""Event模型 + SessionBase接口。

Session = 一次会话 = 一条追加写入的事件流。
独立于Harness和Sandbox，是整个系统唯一持久的状态锚。
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class Event(BaseModel):
    """一条事件记录。所有系统行为的原子单位。

    event_type 是 str 而非 Enum —— 通用层不预设事件类型集合，
    具体的事件类型由项目层（Pipeline/Tools）定义。
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: str
    component: str
    idempotency_key: Optional[str] = None
    correlation_id: Optional[str] = None
    parent_correlation_id: Optional[str] = None
    attempt: int = 1
    attempt_metadata: Optional[dict[str, Any]] = None
    payload_in: Optional[dict[str, Any]] = None
    payload_out: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class SessionInfo(BaseModel):
    """Session元信息。"""

    session_id: str
    created_at: datetime
    last_event_id: Optional[str] = None
    last_event_at: Optional[datetime] = None
    event_count: int = 0


class SessionBase(ABC):
    """一次会话 = 一条追加写入的事件流。

    实现方式：任何支持按顺序消费、接受幂等追加的存储。
    轻量: JSON文件（单进程）  生产: SQLite（多进程安全）/ PostgreSQL
    """

    @abstractmethod
    async def emit(self, event: Event) -> None:
        """追加一条事件（幂等：重复event_id或idempotency_key不重复写入）。"""

    @abstractmethod
    async def get_events(
        self, session_id: str, since: Optional[str] = None
    ) -> list[Event]:
        """获取事件流。since=event_id 时返回该事件之后的未处理事件。"""

    @abstractmethod
    async def get_session(self, session_id: str) -> SessionInfo:
        """获取session元信息。"""

    @abstractmethod
    async def create_session(self, session_id: Optional[str] = None) -> str:
        """创建一个新session，返回session_id。"""

    async def find_by_idempotency_key(
        self, session_id: str, idempotency_key: str
    ) -> Optional[Event]:
        """按 idempotency_key 查找事件，找不到返回 None。

        [FIX Bug-3] 默认实现是 O(n) 线性扫描（够轻量后端用）。
        SQLiteSession 覆盖此方法走索引，把 Harness 每次工具调用的
        幂等查重从 O(n) 降到 O(log n)，消除长 session 的 O(n²) 退化。
        """
        for event in await self.get_events(session_id):
            if event.idempotency_key == idempotency_key:
                return event
        return None
