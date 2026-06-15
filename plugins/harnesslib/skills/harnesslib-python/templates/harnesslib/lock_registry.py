"""Reference-counted registry of per-key locks that retires idle entries.

Shared by the JSON session store (``threading.Lock`` keyed by session id) and
any path-keyed stores (``asyncio.Lock`` keyed by resolved path), so the
refcount/retire bookkeeping lives in one place instead of being duplicated.

The registry only manages the lock objects and their lifetimes; callers acquire
and release the returned lock themselves (a sync ``with`` or an
``async with``), since the two stores use different lock types.

Retirement is sticky: an entry is removed once no holder/waiter remains AND
retirement has been requested at least once. A store that wants idle retirement
(path locks) passes ``retire=True`` on every release; a store that only retires
on a terminal signal (session locks) passes ``retire=True`` solely for that
signal.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, Generic, Hashable, TypeVar

LockT = TypeVar("LockT")


class RefCountedLockState(Generic[LockT]):
    """A lock plus the bookkeeping the registry uses to retire it."""

    def __init__(self, lock: LockT) -> None:
        self.lock = lock
        self.users = 0
        self.retire_after_release = False


class RefCountedLockRegistry(Generic[LockT]):
    """Hand out per-key locks and retire them once idle.

    ``lock_factory`` builds a fresh lock for a new key (e.g. ``asyncio.Lock`` or
    ``threading.Lock``). It is called while the internal guard is held, so it
    must not block or await.
    """

    def __init__(self, lock_factory: Callable[[], LockT]) -> None:
        self._lock_factory = lock_factory
        self._states: Dict[Hashable, RefCountedLockState[LockT]] = {}
        self._guard = threading.Lock()

    def acquire(self, key: Hashable) -> RefCountedLockState[LockT]:
        with self._guard:
            state = self._states.get(key)
            if state is None:
                state = RefCountedLockState(self._lock_factory())
                self._states[key] = state
            state.users += 1
            return state

    def release(
        self,
        key: Hashable,
        state: RefCountedLockState[LockT],
        *,
        retire: bool,
    ) -> None:
        with self._guard:
            if retire:
                state.retire_after_release = True
            state.users -= 1
            if (
                state.users <= 0
                and state.retire_after_release
                and self._states.get(key) is state
            ):
                self._states.pop(key, None)

    def __contains__(self, key: Hashable) -> bool:
        with self._guard:
            return key in self._states

    def __len__(self) -> int:
        with self._guard:
            return len(self._states)
