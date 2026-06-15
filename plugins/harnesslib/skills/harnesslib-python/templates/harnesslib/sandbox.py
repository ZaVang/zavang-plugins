"""Sandbox — 通用执行环境。

所有工具通过 execute(name, payload) -> result 统一接口注册和调用。
不预设具体有哪些工具、每个工具的Schema是什么。

注意：这是一个"调度型 sandbox"——工具在同进程内执行，本身不提供
进程/网络隔离或资源限额。需要真正隔离时，在 handler 内部把调用转发到
子进程 / 容器 / 远程服务（execute(name, input) 这个接口形状正是为此设计的）。
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Optional, Awaitable, Callable

from pydantic import BaseModel, ConfigDict


class ToolDefinition(BaseModel):
    """工具定义——任何可以描述为名称和输入形状的能力。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    input_schema: Optional[dict[str, Any]] = None


class SandboxBase(ABC):
    """通用执行环境。meta-sandbox——不对具体实现做假设。"""

    @abstractmethod
    async def register(
        self,
        definition: ToolDefinition,
        handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> None:
        """注册一个工具及其handler。"""

    @abstractmethod
    async def execute(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """执行一个已注册的工具。"""

    @abstractmethod
    def list_tools(self) -> list[ToolDefinition]:
        """列出所有已注册的工具。"""


class Sandbox(SandboxBase):
    """Sandbox的默认内存实现。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[
            str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
        ] = {}

    async def register(
        self,
        definition: ToolDefinition,
        handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> None:
        # [FIX] async-generator handler（流式工具）在本版本不受支持。
        # 早返回一个清晰错误，而不是等到 `await` 一个 async_generator 时
        # 抛出难以理解的 TypeError。需要流式时，让 handler 自己聚合后返回 dict。
        if inspect.isasyncgenfunction(handler):
            raise NotImplementedError(
                f"Tool '{definition.name}' is an async generator function, which "
                "this Sandbox does not drive. Wrap it in a coroutine that consumes "
                "the generator and returns a dict."
            )
        self._tools[definition.name] = definition
        self._handlers[definition.name] = handler

    async def execute(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if name not in self._handlers:
            raise KeyError(f"Tool not registered: {name}")
        return await self._handlers[name](payload)

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())
