"""SQLite session implementation —— 默认后端，**多进程安全**。

跨进程安全不靠进程内锁，而靠 SQLite 本身：
- WAL + busy_timeout（见 ../sqlite.py）让并发写者串行而非报错；
- ``event_id`` UNIQUE + (session_id, idempotency_key) 部分 UNIQUE 索引，把幂等
  落在数据库约束上，配合 ``INSERT OR IGNORE`` 让重复写入静默去重。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from ..event import Event, SessionBase, SessionInfo
from ..sqlite import (
    get_sqlite_lock,
    record_schema_migration,
    sqlite_conn,
)

_SCHEMA_VERSION = 2


class SQLiteSession(SessionBase):
    """Persist harness events in SQLite (multi-process safe)."""

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)

    async def create_session(self, session_id: Optional[str] = None) -> str:
        if session_id is None:
            now = datetime.now(timezone.utc)
            session_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        lock = await get_sqlite_lock(self.db_path)
        async with lock:
            with sqlite_conn(self.db_path) as connection:
                _ensure_session_schema(connection)
                connection.execute(
                    """
                    INSERT OR IGNORE INTO session_infos(session_id, created_at)
                    VALUES (?, ?)
                    """,
                    (session_id, datetime.now(timezone.utc).isoformat()),
                )
        return session_id

    async def emit(self, event: Event) -> None:
        payload = event.model_dump(mode="json")
        lock = await get_sqlite_lock(self.db_path)
        async with lock:
            with sqlite_conn(self.db_path) as connection:
                _ensure_session_schema(connection)
                connection.execute(
                    """
                    INSERT OR IGNORE INTO session_infos(session_id, created_at)
                    VALUES (?, ?)
                    """,
                    (event.session_id, event.timestamp.isoformat()),
                )
                # [FIX Bug-5] INSERT OR IGNORE + 下面的 UNIQUE 约束让幂等去重落在
                # 数据库层。即使两个进程同时通过应用层检查，第二条插入也会被
                # event_id / (session_id, idempotency_key) 的唯一约束静默丢弃，
                # 不再抛 IntegrityError，也不再产生重复幂等行。
                connection.execute(
                    """
                    INSERT OR IGNORE INTO session_events(
                        event_id, session_id, timestamp, event_type, component,
                        idempotency_key, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.session_id,
                        event.timestamp.isoformat(),
                        event.event_type,
                        event.component,
                        event.idempotency_key,
                        json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    ),
                )

    async def get_events(
        self,
        session_id: str,
        since: Optional[str] = None,
    ) -> list[Event]:
        lock = await get_sqlite_lock(self.db_path)
        async with lock:
            with sqlite_conn(self.db_path) as connection:
                _ensure_session_schema(connection)
                rows = connection.execute(
                    """
                    SELECT payload_json FROM session_events
                    WHERE session_id = ?
                    ORDER BY sequence ASC
                    """,
                    (session_id,),
                ).fetchall()
        events = [Event.model_validate(json.loads(row["payload_json"])) for row in rows]
        if since is None:
            return events
        for index, event in enumerate(events):
            if event.event_id == since:
                return events[index + 1:]
        return []

    async def find_by_idempotency_key(
        self, session_id: str, idempotency_key: str
    ) -> Optional[Event]:
        # [FIX Bug-3] 索引查找（idx_session_events_idempotency），O(log n)。
        lock = await get_sqlite_lock(self.db_path)
        async with lock:
            with sqlite_conn(self.db_path) as connection:
                _ensure_session_schema(connection)
                row = connection.execute(
                    """
                    SELECT payload_json FROM session_events
                    WHERE session_id = ? AND idempotency_key = ?
                    ORDER BY sequence ASC LIMIT 1
                    """,
                    (session_id, idempotency_key),
                ).fetchone()
        if row is None:
            return None
        return Event.model_validate(json.loads(row["payload_json"]))

    async def get_session(self, session_id: str) -> SessionInfo:
        events = await self.get_events(session_id)
        if not events:
            return SessionInfo(
                session_id=session_id,
                created_at=datetime.now(timezone.utc),
            )
        return SessionInfo(
            session_id=session_id,
            created_at=events[0].timestamp,
            last_event_id=events[-1].event_id,
            last_event_at=events[-1].timestamp,
            event_count=len(events),
        )


def _ensure_session_schema(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS session_infos (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS session_events (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            component TEXT NOT NULL,
            idempotency_key TEXT,
            payload_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_session_events_session_sequence "
        "ON session_events(session_id, sequence)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_session_events_idempotency "
        "ON session_events(session_id, idempotency_key)"
    )
    # [FIX Bug-5] 部分唯一索引：同一 session 内 idempotency_key 不可重复
    # （NULL 不参与，多条无 key 的事件不受影响）。这是 INSERT OR IGNORE 去重的依据。
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_session_events_idempotency "
        "ON session_events(session_id, idempotency_key) "
        "WHERE idempotency_key IS NOT NULL"
    )
    record_schema_migration(connection, component="harness_session", version=_SCHEMA_VERSION)
