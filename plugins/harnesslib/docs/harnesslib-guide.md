# harnesslib — Agent Harness Plugin Guide

A reusable **agent harness** you vendor into any Python project, the same way the
**llm-bridge** plugin vendors `llm_bridge`. It is a faithful, hardened implementation of
Anthropic's *Managed Agents* architecture — *"decouple the brain from the hands."*

## The mental model

```
   Orchestration   "ensure a session is handled by some Harness"  (top, simplest)
        │
   ┌────┴─────────────── Brain ───────────────┐
   │  Harness  — Effect loop: Pipeline yields  │      ┌──────── Hands ────────┐
   │  intent, harness interprets + records     │◄────►│ Sandbox: execute(     │
   └───────────────────────────────────────────┘      │   name, payload)->dict│
                  │ every step                          └───────────────────────┘
                  ▼
            Session  — append-only event log (the one durable state anchor)
```

- **Session** is the only thing that persists. Everything is reconstructable from its event log.
- **Harness** holds no business logic. The *Pipeline* (an `async generator` that yields
  `Effect`s) is the domain "brain"; the harness just executes effects and journals them.
- **Sandbox** is a uniform tool-execution surface (`execute(name, payload)`). It is a
  dispatcher, **not** a security boundary — tools run in-process.
- **Gateway** adapts your `llm_bridge` so LLM calls are journaled like any other effect.

## Why a Pipeline yields Effects

The Pipeline never calls the sandbox/session directly. It *declares* intent:

```python
async def pipeline():
    data = yield ExecuteToolEffect(tool_name="fetch", payload={...}, idempotency_key="fetch-1")
    answer = yield ExecuteToolEffect(tool_name="reason", payload={"data": data}, idempotency_key="reason-1")
    yield EmitEventEffect(event=Event(session_id="", event_type="pipeline_end", component="pipeline"))
```

Because the brain is pure/declarative, the harness can journal every step, dedupe by
`idempotency_key`, and **replay** the whole pipeline after a crash — completed steps
short-circuit, the crashed one re-runs.

## Session backends

| | `SQLiteSession` (default) | `JsonSession` |
|---|---|---|
| Multi-process | ✅ WAL + busy_timeout + UNIQUE idempotency | ❌ single-process only |
| Durability | DB-managed | fsync + atomic replace |
| Readability | query with SQL | one JSON file per session |
| Use when | cron + manual coexist, production, scale | local dev / debugging |

Pick SQLite the moment two processes can touch the same session (e.g. you add a cron
trigger while a manual run is going) — that is exactly where a JSON file loses events.

## Hardening built in (vs a naive port)

1. **Crash-replay, not give-up** — a crash mid-tool is the *common* case; the harness replays
   and skips completed steps instead of marking the session dead.
2. **O(log n) idempotency lookup** — indexed in SQLite; removes the per-tool-call full scan
   that makes long sessions O(n²).
3. **Multi-process-safe SQLite** — dedup lives in DB UNIQUE constraints + `INSERT OR IGNORE`,
   not an in-memory lock that two processes can't share.
4. **Replay returns real data** — `tool_response.payload_out` stores the full result; the
   optional `__log__` summary is display-only metadata.
5. **fsync-durable JSON** + read-under-lock (also dodges a Windows `os.replace`-vs-open race).
6. **No connection leak** — every SQLite op opens *and closes* its connection.

## Install / use

This is a **Claude Code plugin**. Trigger the `harnesslib-python` skill in a target project
("add an agent harness", "vendor harnesslib here"); it asks for a location + module name +
default session backend and copies `templates/harnesslib/` in. All internal imports are
relative, so the package is rename-safe.

Verify after copy:
```
python harness_smoke_test.py   # emit / replay / wake / __log__ fix  (pydantic only)
python test_import.py          # full public API + structure         (pydantic + jinja2)
```

## Dependencies

- Core: `pydantic>=2.0`
- `prompt_engine.py` / `LLMTool`: `jinja2>=3.1`
- SQLite: stdlib `sqlite3` (no dependency)

## Pairs with

- **llm-bridge** — provides `llm_bridge.LLMBridge`; wrap it with `BridgeGateway` so all LLM
  calls flow through the harness and land in the session log.
