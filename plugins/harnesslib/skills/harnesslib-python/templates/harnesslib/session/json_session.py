"""JSON文件Session实现 —— 轻量、可读、**单进程**。

每个session一个JSON文件，追加写入事件。

⚠️ 跨进程不安全：幂等与追加只靠进程内内存锁保护，read-modify-write 在多进程下
会丢事件（last-writer-wins）。需要多进程（如 cron 触发 + 手动运行并存）时请用
SQLiteSession —— 它把幂等落在数据库 UNIQUE 约束上，跨进程安全。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from ..event import Event, SessionBase, SessionInfo
from ..lock_registry import RefCountedLockRegistry, RefCountedLockState


# Per-session threading locks that retire only after a terminal event marks the
# session complete (retire=True is passed solely for those events).
_SESSION_LOCKS: RefCountedLockRegistry[threading.Lock] = RefCountedLockRegistry(
    threading.Lock
)
_SESSION_COMPLETED_EVENTS = {"pipeline_end", "session_completed"}

# Trim oversized payload fields before persisting to keep session.json sane.
# NOTE: events carrying an idempotency_key are NEVER trimmed (see emit) because
# they are the authoritative source for idempotent replay.
_PAYLOAD_FIELD_MAX_BYTES = 64 * 1024
_PAYLOAD_TOP_KEYS = ("payload_in", "payload_out")


class JsonSession(SessionBase):
    """基于JSON文件的Session实现（单进程）。"""

    def __init__(
        self,
        data_dir: Union[str, Path],
        *,
        trim_large_payloads: bool = True,
        completed_event_types: Optional[Iterable[str]] = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        # Which event types mark a session complete (so its lock can retire).
        # Default is the generic pipeline events; pass your own terminal event
        # types if your pipeline never emits those — otherwise the per-session
        # lock would never retire (slow memory growth over many sessions).
        self._completed_event_types = (
            frozenset(completed_event_types)
            if completed_event_types is not None
            else _SESSION_COMPLETED_EVENTS
        )
        # Trimming oversized fields keeps session.json cheap to store but is
        # mutually exclusive with idempotent replay of those fields. Pass False
        # when this session is the authoritative store consumers read back.
        self._trim_large_payloads = trim_large_payloads

    def _session_path(self, session_id: str) -> Path:
        return self._data_dir / f"{session_id}.json"

    async def create_session(self, session_id: Optional[str] = None) -> str:
        import uuid

        if session_id is None:
            now = datetime.now(timezone.utc)
            short_uuid = uuid.uuid4().hex[:8]
            session_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{short_uuid}"

        path = self._session_path(session_id)
        if not path.exists():
            path.write_text(json.dumps([], ensure_ascii=False, indent=2))
        return session_id

    async def emit(self, event: Event) -> None:
        # Offload the blocking read-modify-write to a worker thread so the event
        # loop is not stalled while one session file is rewritten. Different
        # sessions use different locks, so concurrent runs no longer serialize
        # on each other's session writes.
        await asyncio.to_thread(self._emit_sync, event)

    def _emit_sync(self, event: Event) -> None:
        path = self._session_path(event.session_id)
        lock_state = _acquire_session_lock_state(event.session_id)
        retire_after_emit = False

        try:
            with lock_state.lock:
                # 读取已有事件
                events: list[dict] = []
                if path.exists():
                    events = json.loads(path.read_text(encoding="utf-8"))

                # 幂等：重复 event_id 或 idempotency_key 不重复写入
                existing_ids = {e["event_id"] for e in events}
                if event.event_id in existing_ids:
                    return
                if event.idempotency_key is not None:
                    existing_keys = {
                        e.get("idempotency_key")
                        for e in events
                        if e.get("idempotency_key") is not None
                    }
                    if event.idempotency_key in existing_keys:
                        return

                event_dict = event.model_dump(mode="json")
                # [FIX Bug-4] 带 idempotency_key 的事件是幂等重放的权威来源，绝不裁剪，
                # 否则重放会拿到 {_omitted: True} 占位而非真数据。
                if self._trim_large_payloads and event.idempotency_key is None:
                    event_dict = _trim_large_payload_fields(event_dict)
                events.append(event_dict)
                _write_json_atomic(path, events)
                retire_after_emit = event.event_type in self._completed_event_types
        finally:
            _release_session_lock_state(
                event.session_id,
                lock_state,
                retire=retire_after_emit,
            )

    async def get_events(
        self, session_id: str, since: Optional[str] = None
    ) -> list[Event]:
        return await asyncio.to_thread(self._get_events_sync, session_id, since)

    def _get_events_sync(
        self, session_id: str, since: Optional[str] = None
    ) -> list[Event]:
        path = self._session_path(session_id)
        # Read under the same per-session lock that emit() holds, so a read
        # never observes a partially written file (and on Windows never races
        # an in-flight os.replace, which would raise PermissionError).
        lock_state = _acquire_session_lock_state(session_id)
        try:
            with lock_state.lock:
                if not path.exists():
                    return []
                events_raw = json.loads(path.read_text(encoding="utf-8"))
        finally:
            _release_session_lock_state(session_id, lock_state, retire=False)

        events = [Event.model_validate(e) for e in events_raw]

        if since is not None:
            # 找到since event_id的位置，返回之后的事件
            for i, e in enumerate(events):
                if e.event_id == since:
                    return events[i + 1:]
            return []  # since event_id not found

        return events

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


def _acquire_session_lock_state(
    session_id: str,
) -> RefCountedLockState[threading.Lock]:
    return _SESSION_LOCKS.acquire(session_id)


def _release_session_lock_state(
    session_id: str,
    lock_state: RefCountedLockState[threading.Lock],
    *,
    retire: bool,
) -> None:
    _SESSION_LOCKS.release(session_id, lock_state, retire=retire)


def _trim_large_payload_fields(event_dict: dict[str, Any]) -> dict[str, Any]:
    """Replace top-level payload fields larger than ``_PAYLOAD_FIELD_MAX_BYTES``
    with a digest placeholder so session.json does not balloon.

    Only top-level keys inside ``payload_in`` / ``payload_out`` are inspected;
    nested structure is preserved when the field itself is small.
    """

    for payload_key in _PAYLOAD_TOP_KEYS:
        payload = event_dict.get(payload_key)
        if not isinstance(payload, dict):
            continue
        for field_name in list(payload.keys()):
            value = payload[field_name]
            try:
                serialized = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                continue
            size = len(serialized.encode("utf-8"))
            if size <= _PAYLOAD_FIELD_MAX_BYTES:
                continue
            digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
            payload[field_name] = {
                "_omitted": True,
                "size_bytes": size,
                "sha256_prefix": digest,
                "summary": _summarize_value(value),
            }
    return event_dict


def _summarize_value(value: Any) -> str:
    if isinstance(value, list):
        return "list[{0}]".format(len(value))
    if isinstance(value, dict):
        return "dict({0} keys)".format(len(value))
    return type(value).__name__


def _write_json_atomic(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(events, handle, ensure_ascii=False, indent=2)
            # [FIX durability] flush + fsync 临时文件，确保 os.replace 之后数据真的落盘。
            # 否则掉电/内核崩溃时 rename 可见但数据块未刷 -> 文件被截断/清空。
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
        # best-effort: fsync 父目录让 rename 本身持久（POSIX）。Windows 上会失败，忽略。
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError):
            pass
    finally:
        if temp_name is not None:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass
