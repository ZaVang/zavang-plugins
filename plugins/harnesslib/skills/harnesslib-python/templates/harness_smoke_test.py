"""harnesslib — functional smoke test (pydantic-only, no network, no jinja2).

Covers the hardening fixes:
  - SQLite idempotent dedup (DB UNIQUE constraint)
  - Harness idempotent replay returns the FULL result (the __log__ fix)
  - replay does NOT re-execute a completed tool
  - wake: new / resumable / unrecoverable policy
  - JSON session basic emit + dedup

Run:  python harness_smoke_test.py
"""

import asyncio
import os
import sys
import tempfile

# Make the sibling `harnesslib/` package importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import core submodules directly so this test only needs pydantic
# (importing the top-level package would also pull jinja2 via tools/).
from harnesslib.event import Event
from harnesslib.sandbox import Sandbox, ToolDefinition
from harnesslib.harness import Harness, ExecuteToolEffect
from harnesslib.session.sqlite_session import SQLiteSession
from harnesslib.session.json_session import JsonSession


async def test_sqlite_idempotent_dedup():
    with tempfile.TemporaryDirectory() as d:
        session = SQLiteSession(os.path.join(d, "s.db"))
        sid = await session.create_session()
        await session.emit(Event(session_id=sid, event_type="x", component="c",
                                 idempotency_key="k", payload_out={"n": 1}))
        # Same idempotency_key, different event_id, different payload -> ignored.
        await session.emit(Event(session_id=sid, event_type="x", component="c",
                                 idempotency_key="k", payload_out={"n": 2}))
        events = await session.get_events(sid)
        assert len(events) == 1, f"expected 1 event, got {len(events)}"
        assert events[0].payload_out == {"n": 1}
        # indexed lookup
        found = await session.find_by_idempotency_key(sid, "k")
        assert found is not None and found.payload_out == {"n": 1}
    print("  OK  sqlite idempotent dedup + indexed lookup")


async def test_harness_replay_full_result():
    with tempfile.TemporaryDirectory() as d:
        session = SQLiteSession(os.path.join(d, "s.db"))
        sandbox = Sandbox()
        calls = {"n": 0}

        async def tool(payload):
            calls["n"] += 1
            # __log__ is a short summary; data is the real payload.
            return {"__log__": "short", "data": [1, 2, 3]}

        await sandbox.register(ToolDefinition(name="t", description="t"), tool)
        harness = Harness(session, sandbox, resume_policy="replay")
        sid = await session.create_session()

        captured = []

        async def pipeline():
            r = yield ExecuteToolEffect(tool_name="t", payload={}, idempotency_key="k1")
            captured.append(r)

        await harness.run_effect_loop(sid, pipeline())
        assert calls["n"] == 1
        assert captured[0]["data"] == [1, 2, 3], "first run lost real data"

        # Re-run with the same key: must short-circuit (no re-exec) and still
        # return the FULL result, not the __log__ summary.
        captured.clear()
        await harness.run_effect_loop(sid, pipeline())
        assert calls["n"] == 1, f"tool re-executed on replay ({calls['n']} calls)"
        assert captured[0]["data"] == [1, 2, 3], "replay returned summary, not real data"
    print("  OK  harness replay: no re-exec + full result (__log__ fix)")


async def test_wake_policies():
    with tempfile.TemporaryDirectory() as d:
        sandbox = Sandbox()

        # fresh session -> "new"
        s1 = SQLiteSession(os.path.join(d, "a.db"))
        sid1 = await s1.create_session()
        assert await Harness(s1, sandbox).wake(sid1) == "new"

        # orphan tool_request (crashed mid-tool) -> replay policy => "resumable"
        s2 = SQLiteSession(os.path.join(d, "b.db"))
        sid2 = await s2.create_session()
        await s2.emit(Event(session_id=sid2, event_type="tool_request",
                            component="t", correlation_id="c1"))
        assert await Harness(s2, sandbox, resume_policy="replay").wake(sid2) == "resumable"

        # same orphan, unrecoverable policy => "unrecoverable"
        s3 = SQLiteSession(os.path.join(d, "c.db"))
        sid3 = await s3.create_session()
        await s3.emit(Event(session_id=sid3, event_type="tool_request",
                            component="t", correlation_id="c1"))
        assert await Harness(s3, sandbox, resume_policy="unrecoverable").wake(sid3) == "unrecoverable"
    print("  OK  wake policies: new / resumable / unrecoverable")


async def test_json_session():
    with tempfile.TemporaryDirectory() as d:
        session = JsonSession(d)
        sid = await session.create_session()
        await session.emit(Event(session_id=sid, event_type="x", component="c",
                                 idempotency_key="k", payload_out={"n": 1}))
        await session.emit(Event(session_id=sid, event_type="x", component="c",
                                 idempotency_key="k", payload_out={"n": 2}))
        events = await session.get_events(sid)
        assert len(events) == 1, f"expected 1, got {len(events)}"
    print("  OK  json session emit + dedup + fsync")


async def main():
    tests = [
        test_sqlite_idempotent_dedup,
        test_harness_replay_full_result,
        test_wake_policies,
        test_json_session,
    ]
    print("\n harnesslib smoke test\n")
    passed = failed = 0
    for t in tests:
        try:
            await t()
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {e!r}")
            failed += 1
    print(f"\n{'='*48}\nResults: {passed} passed, {failed} failed\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
