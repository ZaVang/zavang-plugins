"""harnesslib.session — Session实现。

- JsonSession: 轻量、可读，**单进程**用。每个 session 一个 JSON 文件。
- SQLiteSession: 默认、**多进程安全**（WAL + busy_timeout + UNIQUE 幂等约束）。
"""

from .json_session import JsonSession
from .sqlite_session import SQLiteSession

__all__ = ["JsonSession", "SQLiteSession"]
