---
name: harnesslib-python
description: Generate a production-ready agent harness module in any Python project. A reusable Session/Harness/Sandbox/Orchestration framework inspired by Anthropic's Managed Agents — append-only session log, effect-loop harness, idempotent replay, crash resume, multi-process-safe SQLite persistence. Use when the user wants to build an agent loop, orchestrate LLM tool-use with durable state, add crash-resumable long-running agents, or needs a harness layer separate from business logic.
---

# Harnesslib Python Skill

Auto-generate a complete, reusable **agent harness** module in any Python project. The
generated module is a faithful, hardened implementation of Anthropic's *Managed Agents*
architecture ("decouple the brain from the hands"):

- **Session** — an append-only event log; the single durable state anchor (`emit` / `get_events`)
- **Harness** — an Effect loop: the Pipeline yields intent (`ExecuteToolEffect`, …), the harness interprets it and records every step to the session
- **Sandbox** — a uniform `execute(name, payload) -> dict` tool-execution surface
- **Gateway** — a thin LLM entry point that adapts your `llm_bridge` (see the companion **llm-bridge** plugin)
- **Orchestration** — the top, simplest layer: "ensure a session is being handled by some Harness"

It carries built-in fixes over a naive harness: **idempotent crash-replay**, **O(log n) idempotency lookup**, **multi-process-safe SQLite persistence**, and **fsync-durable JSON sessions**.

## When to Use This Skill

Activate when the user wants to:
- Build an agent loop / orchestration layer that is separate from business logic
- Run LLM tool-use with **durable, replayable** state (survive crashes / restarts)
- Add long-running, crash-resumable agents (cron-triggered, multi-process)
- Vendor a shared harness into several projects (like the `llm_gateway` / llm-bridge pattern)

## Architecture (what gets generated)

```
$MODULE_NAME/
├── __init__.py          # public API (Harness, Orchestration, SQLiteSession, Effects, …)
├── event.py             # Event model + SessionBase interface (find_by_idempotency_key)
├── context.py           # ContextVar runtime: nested tool exec + session access
├── sandbox.py           # SandboxBase + in-memory Sandbox (execute(name, payload))
├── gateway.py           # GatewayBase + BridgeGateway (adapts llm_bridge)
├── harness.py           # Harness: Effect loop, wake/replay, idempotent tool exec
├── orchestration.py     # Orchestration: wake / trigger
├── prompt_engine.py     # Jinja2 template render + version tracking
├── resources.py         # ResourceManager (source_ref -> path)
├── lock_registry.py     # ref-counted per-key locks (retire when idle)
├── sqlite.py            # shared SQLite helpers (WAL + busy_timeout)
├── session/
│   ├── __init__.py
│   ├── json_session.py   # single-process, fsync-durable JSON files
│   └── sqlite_session.py # DEFAULT: multi-process-safe (WAL + UNIQUE idempotency)
└── tools/
    ├── __init__.py
    ├── llm_tool.py       # "render template -> call LLM -> result" pattern
    └── data_tool.py      # "call adapter -> data" pattern
```

## Workflow

### Step 1: Determine Target Location and Module Name

Ask the user two things:

1. **Where** the harness module should be placed (parent directory). Common choices:
   - `src/` — flat layout (default)
   - `src/services/` — service-oriented
   - `infra/` — infrastructure-heavy

2. **What to name** the module directory (becomes the Python import name). Common choices:
   - `harnesslib` (default)
   - `harness`
   - `agent_harness`

Record parent as `$PARENT_DIR`, module name as `$MODULE_NAME`, full path `$TARGET_DIR = $PARENT_DIR/$MODULE_NAME`, and import path `$MODULE_IMPORT_PATH` (e.g. `src.harnesslib`).

**All internal imports are relative**, so the package is rename-safe — copying it under any
`$MODULE_NAME` just works, no import rewrites needed.

### Step 2: Choose the Session Backend

Ask which Session backend to wire as default:

1. **SQLite** (default, recommended) — multi-process safe (WAL + `busy_timeout` + UNIQUE
   idempotency constraints). Use when more than one process may touch the same session
   (e.g. a cron trigger running alongside a manual run), or you want a durable store.
2. **JSON** — one file per session, human-readable. **Single-process only.** Good for local
   dev / debugging. Hardened with fsync + read-under-lock, but cross-process writes can still
   lose events — do not use it from multiple processes.

Both are always generated; this only sets which one the usage examples and app wiring default to.

### Step 3: Copy Template Files

Copy everything under this skill's `templates/harnesslib/` to `$TARGET_DIR`, preserving the
subdirectory layout (`session/`, `tools/`). Do not rename internal files.

### Step 4: Update Dependencies

The harness core needs only **pydantic**; `prompt_engine.py` / `LLMTool` additionally need
**jinja2**. SQLite uses the stdlib `sqlite3` (no dep).

**For `requirements.txt`**, append (skip if present):
```
pydantic>=2.0
jinja2>=3.1   # only if you use prompt_engine / LLMTool
```
**For `pyproject.toml`**, add the equivalents to `[project.dependencies]`.

### Step 5: Wire the Gateway (optional, if doing LLM tool-use)

If the project uses the **llm-bridge** plugin's module, adapt it:
```python
from $MODULE_IMPORT_PATH import BridgeGateway
from llm_bridge import LLMBridge   # or wherever llm_bridge was vendored

gateway = BridgeGateway(LLMBridge.from_config("llm_bridge_config.yaml"))
```

### Step 6: Display Usage Guide

Use the actual `$MODULE_IMPORT_PATH`:

```python
import asyncio
from $MODULE_IMPORT_PATH import (
    Harness, Orchestration, Sandbox, SQLiteSession,
    ExecuteToolEffect, ToolDefinition,
)

async def main():
    # 1) Session (durable, multi-process-safe) + Sandbox (tools)
    session = SQLiteSession("data/sessions.db")
    sandbox = Sandbox()

    async def greet(payload):
        return {"text": f"hello {payload['name']}"}
    await sandbox.register(ToolDefinition(name="greet", description="greet"), greet)

    # 2) Harness + Orchestration. resume_policy="replay" => crash-resumable.
    harness = Harness(session, sandbox, resume_policy="replay")
    orch = Orchestration(harness)

    # 3) A Pipeline is an async generator that yields Effects (declarative "brain").
    async def pipeline():
        result = yield ExecuteToolEffect(
            tool_name="greet",
            payload={"name": "world"},
            idempotency_key="greet-world",   # stable key => safe replay
        )
        print(result)   # {'text': 'hello world'}

    # 4) Trigger + run. Re-running after a crash replays and skips done steps.
    session_id, status = await orch.trigger(trigger_kind="manual")
    await harness.run_effect_loop(session_id, pipeline())

asyncio.run(main())
```

### Step 7: Run the Smoke Test

From `$TARGET_DIR`'s parent, run the copied `harness_smoke_test.py` to verify
emit / idempotent-replay / wake all work:
```
python harness_smoke_test.py
```

## Important Rules for AI

1. **The harness must stay business-agnostic** — it knows nothing about the domain. Put
   domain logic in the Pipeline (the async generator) and in tool handlers, never in `harness.py`.
2. **Give tools stable `idempotency_key`s** whenever a re-run must not duplicate side effects.
   Crash-replay (`resume_policy="replay"`) only short-circuits completed steps when keys are stable.
3. **Use SQLiteSession for anything multi-process.** JSON is single-process only.
4. **Do not trim payloads you need to replay.** Events with an `idempotency_key` are the
   authoritative replay source and are never trimmed; for very large outputs, store them in a
   canonical repository and put only a reference in the tool result.
5. **The "Sandbox" is a dispatcher, not a security boundary** — tools run in-process with full
   privileges. For real isolation, have a handler forward to a subprocess / container / remote.

## Notable Hardening (vs a naive port)

- **Crash-replay over give-up** (`harness.py` wake): a crash mid-tool no longer marks the
  session unrecoverable; it replays and skips completed steps via idempotency keys.
- **O(log n) idempotency lookup** (`SessionBase.find_by_idempotency_key`, indexed in SQLite):
  removes the O(n²) full-scan on every tool call in long sessions.
- **Multi-process-safe SQLite**: WAL + `busy_timeout` + a partial UNIQUE index on
  `(session_id, idempotency_key)` + `INSERT OR IGNORE` push dedup into the DB, not an
  in-memory lock.
- **Replay correctness**: `tool_response.payload_out` stores the *full* result (the optional
  `__log__` summary is display-only metadata), so an idempotent replay returns real data.
- **fsync-durable JSON** + read-under-lock (also fixes a Windows `os.replace`-vs-open race).
