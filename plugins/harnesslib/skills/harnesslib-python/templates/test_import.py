"""harnesslib — Import & Structure Test.

Validates the full public API imports and is wired correctly.
Requires pydantic and jinja2 (the latter for prompt_engine / LLMTool).
No API keys or network access required.

Run:  python test_import.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_public_api_imports():
    """All advertised public symbols import from the top-level package."""
    from harnesslib import (  # noqa: F401
        Event, SessionBase, SessionInfo, JsonSession, SQLiteSession,
        Harness, Effect, ExecuteToolEffect, EmitEventEffect, GetEventsEffect,
        Orchestration, Sandbox, SandboxBase, ToolDefinition,
        GatewayBase, GatewayResponse, BridgeGateway, chat_with_gateway,
        ResourceManager, ResourceRef, PromptEngine,
        HarnessRuntimeContext, get_harness_context,
        get_harness_runtime_context, execute_tool,
        RefCountedLockRegistry, RefCountedLockState,
        LLMTool, DataTool, set_log_prompts,
    )
    print("  OK  public API imports")


def test_all_exports_resolve():
    """Everything named in __all__ is actually present on the package."""
    import harnesslib

    missing = [name for name in harnesslib.__all__ if not hasattr(harnesslib, name)]
    assert not missing, f"__all__ names missing from module: {missing}"
    print(f"  OK  __all__ resolves ({len(harnesslib.__all__)} symbols)")


def test_effect_hierarchy():
    """Effect subclasses carry the documented fields."""
    from harnesslib import ExecuteToolEffect, Effect

    e = ExecuteToolEffect(tool_name="t", payload={"a": 1}, idempotency_key="k")
    assert isinstance(e, Effect)
    assert e.tool_name == "t" and e.payload == {"a": 1} and e.attempt == 1
    print("  OK  Effect hierarchy + fields")


def test_harness_resume_policy_validation():
    """Harness rejects an unknown resume_policy and accepts the valid ones."""
    from harnesslib import Harness, Sandbox, SQLiteSession

    sandbox = Sandbox()
    session = SQLiteSession("unused.db")  # not touched (no I/O in __init__)
    for policy in ("replay", "unrecoverable"):
        Harness(session, sandbox, resume_policy=policy)
    try:
        Harness(session, sandbox, resume_policy="nope")
    except ValueError:
        print("  OK  resume_policy validation")
    else:
        raise AssertionError("expected ValueError for bad resume_policy")


def test_event_idempotency_key_default_scan():
    """SessionBase.find_by_idempotency_key has a concrete default (the scan)."""
    from harnesslib import SessionBase

    assert "find_by_idempotency_key" in SessionBase.__dict__, (
        "find_by_idempotency_key should be a concrete default on SessionBase"
    )
    print("  OK  SessionBase.find_by_idempotency_key default present")


if __name__ == "__main__":
    print("\n harnesslib import & structure test\n")
    tests = [
        test_public_api_imports,
        test_all_exports_resolve,
        test_effect_hierarchy,
        test_harness_resume_policy_validation,
        test_event_idempotency_key_default_scan,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {e!r}")
            failed += 1
    print(f"\n{'='*48}\nResults: {passed} passed, {failed} failed\n")
    raise SystemExit(1 if failed else 0)
