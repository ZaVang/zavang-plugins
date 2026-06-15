"""harnesslib — 通用 Agent 编排框架（借鉴 Anthropic Managed Agents）。

把一个 agent 拆成可独立替换的三块 + 调度层，跨项目复用、不含业务假设：

    Session  —— 只追加的事件流，唯一持久状态锚（JsonSession / SQLiteSession）
    Harness  —— Effect 循环：Pipeline yield 意图，Harness 解释执行并落事件
    Sandbox  —— execute(name, payload) -> dict 的统一工具执行接口
    Gateway  —— LLM 统一调用入口（适配你的 llm_bridge）
    Orchestration —— 最高层调度，只负责"确保 session 被某个 Harness 处理"

Public API::

    from harnesslib import (
        Harness, Orchestration, Sandbox, SQLiteSession,
        Effect, ExecuteToolEffect, EmitEventEffect, GetEventsEffect,
        Event, BridgeGateway,
    )

    session = SQLiteSession("data/sessions.db")
    sandbox = Sandbox()
    harness = Harness(session, sandbox, resume_policy="replay")
    orch = Orchestration(harness)
    session_id, status = await orch.trigger(trigger_kind="cron")
    await harness.run_effect_loop(session_id, my_pipeline())
"""

from __future__ import annotations

from .event import Event, SessionBase, SessionInfo
from .harness import (
    Harness,
    Effect,
    ExecuteToolEffect,
    EmitEventEffect,
    GetEventsEffect,
)
from .orchestration import Orchestration
from .sandbox import Sandbox, SandboxBase, ToolDefinition
from .gateway import (
    GatewayBase,
    GatewayResponse,
    BridgeGateway,
    chat_with_gateway,
)
from .resources import ResourceManager, ResourceRef
from .prompt_engine import PromptEngine
from .context import (
    HarnessRuntimeContext,
    get_harness_context,
    get_harness_runtime_context,
    execute_tool,
)
from .lock_registry import RefCountedLockRegistry, RefCountedLockState
from .session import JsonSession, SQLiteSession
from .tools import LLMTool, DataTool, set_log_prompts

__all__ = [
    # Session / Event
    "Event",
    "SessionBase",
    "SessionInfo",
    "JsonSession",
    "SQLiteSession",
    # Harness + Effects
    "Harness",
    "Effect",
    "ExecuteToolEffect",
    "EmitEventEffect",
    "GetEventsEffect",
    # Orchestration
    "Orchestration",
    # Sandbox
    "Sandbox",
    "SandboxBase",
    "ToolDefinition",
    # Gateway
    "GatewayBase",
    "GatewayResponse",
    "BridgeGateway",
    "chat_with_gateway",
    # Resources / Prompt
    "ResourceManager",
    "ResourceRef",
    "PromptEngine",
    # Runtime context
    "HarnessRuntimeContext",
    "get_harness_context",
    "get_harness_runtime_context",
    "execute_tool",
    # Locks
    "RefCountedLockRegistry",
    "RefCountedLockState",
    # Tools
    "LLMTool",
    "DataTool",
    "set_log_prompts",
]
