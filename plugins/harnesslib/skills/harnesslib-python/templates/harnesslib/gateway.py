"""Gateway — LLM统一调用入口。

harnesslib的Gateway是一个轻量接口层。
实际的多Provider支持、重试、熔断等由你的 LLM gateway 提供
（例如 llm_bridge.LLMBridge，参见配套的 llm-bridge 插件）。

两层关系：
- harnesslib.gateway: 定义 GatewayBase 接口（通用抽象）
- llm_bridge.LLMBridge: 完整实现（多Provider/重试/熔断/工具调用）
- 项目层通过 BridgeGateway 适配器连接两者
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field

from .context import get_harness_runtime_context
from .event import Event


class GatewayResponse(BaseModel):
    """LLM调用响应（通用层的简化视图）。"""

    content: Optional[str] = None
    model: str
    usage: dict[str, Any]
    latency_ms: int
    stop_reason: Optional[str] = None
    tool_events: list[dict[str, Any]] = Field(default_factory=list)
    content_blocks: Optional[list[dict[str, Any]]] = None
    thinking: Optional[str] = None


class GatewayBase(ABC):
    """LLM统一调用入口（通用层接口）。

    harnesslib只定义这个最小接口。
    具体实现由你的 LLM gateway 通过适配器提供。
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        model_tier: str = "smart",
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> GatewayResponse:
        """发送消息，返回响应。

        model_tier: "fast"（降噪/摘要）或 "smart"（深度推理）
        tools: Claude API tools列表（如 web_search），透传给底层 bridge
        """


class BridgeGateway(GatewayBase):
    """将一个 LLM bridge（如 llm_bridge.LLMBridge）适配为 harnesslib.GatewayBase。

    这是连接 harnesslib 通用层和具体 LLM gateway 的适配器。
    bridge 只需提供一个 ``async chat(model, messages, **kwargs)`` 方法，
    其响应对象带有 content / model_id / usage / latency_ms 等属性。
    """

    def __init__(self, bridge: Any) -> None:
        """bridge: 你的 LLM gateway 实例（如 llm_bridge.LLMBridge）。"""
        self._bridge = bridge

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model_tier: str = "smart",
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> GatewayResponse:
        runtime = get_harness_runtime_context()
        correlation_id = str(uuid.uuid4())
        parent_correlation_id = (
            runtime.correlation_id if runtime is not None else None
        )
        bridge_tools = kwargs.get("bridge_tools")
        if runtime is not None:
            await runtime.session.emit(
                Event(
                    session_id=runtime.session_id,
                    event_type="llm_request",
                    component="gateway.chat",
                    correlation_id=correlation_id,
                    parent_correlation_id=parent_correlation_id,
                    payload_in={
                        "messages_count": len(messages),
                        "model_tier": model_tier,
                        "tools_count": len(tools or []),
                        "bridge_tools_count": len(bridge_tools or []),
                    },
                )
            )

        call_kwargs: dict[str, Any] = {**kwargs}
        bridge_tools = call_kwargs.pop("bridge_tools", None)
        if bridge_tools is not None:
            call_kwargs["tools"] = bridge_tools
        if tools is not None:
            # Pass raw API tool dicts (e.g. web_search) as api_tools
            # to avoid conflict with bridge's BridgeTool-based tools parameter.
            # api_tools flows through **kwargs -> merged_kwargs -> provider sdk_kwargs.
            call_kwargs["api_tools"] = tools

        try:
            response = await self._bridge.chat(
                model=model_tier,
                messages=messages,
                **call_kwargs,
            )
        except Exception as exc:
            if runtime is not None:
                await runtime.session.emit(
                    Event(
                        session_id=runtime.session_id,
                        event_type="llm_error",
                        component="gateway.chat",
                        correlation_id=correlation_id,
                        parent_correlation_id=parent_correlation_id,
                        payload_in={
                            "messages_count": len(messages),
                            "model_tier": model_tier,
                            "tools_count": len(tools or []),
                            "bridge_tools_count": len(bridge_tools or []),
                        },
                        error=str(exc),
                    )
                )
            raise

        # Extract content_blocks from response if available
        content_blocks: Optional[list[dict[str, Any]]] = getattr(
            response, "content_blocks", None
        )

        gateway_response = GatewayResponse(
            content=response.content,
            model=response.model_id or model_tier,
            usage=response.usage.model_dump() if response.usage else {},
            latency_ms=int(getattr(response, "latency_ms", 0)),
            stop_reason=getattr(response, "stop_reason", None),
            tool_events=getattr(response, "tool_events", []),
            content_blocks=content_blocks,
            thinking=getattr(response, "thinking", None),
        )
        if runtime is not None:
            await runtime.session.emit(
                Event(
                    session_id=runtime.session_id,
                    event_type="llm_response",
                    component="gateway.chat",
                    correlation_id=correlation_id,
                    parent_correlation_id=parent_correlation_id,
                    payload_out={
                        "content": gateway_response.content,
                        "model": gateway_response.model,
                        "usage": gateway_response.usage,
                        "latency_ms": gateway_response.latency_ms,
                        "stop_reason": gateway_response.stop_reason,
                        "tool_events": gateway_response.tool_events,
                    },
                )
            )
        return gateway_response


async def chat_with_gateway(
    gateway: GatewayBase,
    messages: list[dict[str, Any]],
    model_tier: str = "smart",
    tools: Optional[list[dict[str, Any]]] = None,
    **kwargs: Any,
) -> GatewayResponse:
    """Call a GatewayBase implementation through the harnesslib boundary."""
    return await gateway.chat(
        messages=messages,
        model_tier=model_tier,
        tools=tools,
        **kwargs,
    )
