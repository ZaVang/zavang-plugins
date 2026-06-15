"""Harness runtime context — per-execution session access via ContextVar.

Tools that need to emit events directly (instead of via Generator Tool pattern)
call get_harness_context() to retrieve the current (session_id, session) pair.
ContextVar values are inherited by asyncio.create_task(), so parallel subtasks
automatically see the same session.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional, Awaitable, Callable, Protocol, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .event import SessionBase

_current: ContextVar[Optional[Tuple[str, "SessionBase"]]] = ContextVar(
    "harness_current", default=None
)


class NestedToolExecutor(Protocol):
    """Execute a child tool with explicit effect metadata.

    Parameters are:
    name, payload, idempotency_key, correlation_id, attempt, attempt_metadata.
    """

    def __call__(
        self,
        name: str,
        payload: dict[str, Any],
        idempotency_key: Optional[str],
        correlation_id: Optional[str],
        attempt: int,
        attempt_metadata: Optional[dict[str, Any]],
    ) -> Awaitable[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class HarnessRuntimeContext:
    """Runtime services available to code currently executing under Harness."""

    session_id: str
    session: "SessionBase"
    execute_tool: NestedToolExecutor
    correlation_id: Optional[str] = None


_runtime: ContextVar[Optional[HarnessRuntimeContext]] = ContextVar(
    "harness_runtime", default=None
)


def get_harness_context() -> Optional[Tuple[str, "SessionBase"]]:
    """Return (session_id, session) for the currently executing tool, or None."""
    return _current.get()


def get_harness_runtime_context() -> Optional[HarnessRuntimeContext]:
    """Return the full Harness runtime context for nested effects, if present."""
    return _runtime.get()


async def execute_tool(
    name: str,
    payload: dict[str, Any],
    *,
    idempotency_key: Optional[str] = None,
    correlation_id: Optional[str] = None,
    attempt: int = 1,
    attempt_metadata: Optional[dict[str, Any]] = None,
    fallback: Optional[Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]] = None,
) -> dict[str, Any]:
    """Execute a nested tool through Harness when running inside a session."""
    runtime = _runtime.get()
    if runtime is not None:
        return await runtime.execute_tool(
            name,
            payload,
            idempotency_key,
            correlation_id,
            attempt,
            attempt_metadata,
        )
    if fallback is not None:
        return await fallback(name, payload)
    raise RuntimeError("No Harness runtime context is active")


def _set_harness_context(session_id: str, session: "SessionBase"):
    """Internal: set by Harness before each tool execution. Returns the reset token."""
    return _current.set((session_id, session))


def _set_harness_runtime_context(
    runtime: HarnessRuntimeContext,
):
    """Internal: set by Harness before each tool execution. Returns reset token."""
    return _runtime.set(runtime)
