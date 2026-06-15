"""Small SQLite helpers shared by local persistence implementations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Union
import asyncio
import sqlite3
import threading
from pathlib import Path


_SQLITE_LOCKS: dict[str, asyncio.Lock] = {}
_SQLITE_LOCK_GUARD = threading.Lock()

# [FIX Bug-7] How long (ms) a writer waits for a competing writer before
# raising "database is locked". With WAL + a real busy_timeout, concurrent
# writers from *different processes* serialize instead of erroring out.
_BUSY_TIMEOUT_MS = 5000


async def get_sqlite_lock(path: Union[str, Path]) -> asyncio.Lock:
    """Per-path in-process async lock.

    Note: this only serializes writers *within one process*. Cross-process
    safety comes from SQLite itself (WAL + busy_timeout + the UNIQUE
    constraints in the schema), not from this lock.
    """
    key = str(Path(path).resolve())
    with _SQLITE_LOCK_GUARD:
        lock = _SQLITE_LOCKS.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _SQLITE_LOCKS[key] = lock
        return lock


def connect_sqlite(path: Union[str, Path]) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    # [FIX Bug-7] Without this a second concurrent writer fails immediately
    # with sqlite3.OperationalError: database is locked.
    connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    return connection


@contextmanager
def sqlite_conn(path: Union[str, Path]) -> Iterator[sqlite3.Connection]:
    """Open a connection, run inside a transaction, and ALWAYS close it.

    [FIX Bug-9] ``with sqlite3.connect(...)`` only manages the transaction, not
    the connection's lifetime — leaving it open leaks the handle (one per op
    here) and, on Windows, holds a file lock that blocks deletion. This wrapper
    commits on success / rolls back on error AND closes the connection.
    """
    connection = connect_sqlite(path)
    try:
        with connection:  # commit on success, rollback on exception
            yield connection
    finally:
        connection.close()


def ensure_schema_migrations(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            component TEXT PRIMARY KEY,
            version INTEGER NOT NULL,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def record_schema_migration(
    connection: sqlite3.Connection,
    *,
    component: str,
    version: int,
) -> None:
    ensure_schema_migrations(connection)
    connection.execute(
        """
        INSERT INTO schema_migrations(component, version)
        VALUES (?, ?)
        ON CONFLICT(component) DO UPDATE SET
            version = excluded.version,
            applied_at = CURRENT_TIMESTAMP
        WHERE excluded.version >= schema_migrations.version
        """,
        (component, version),
    )
