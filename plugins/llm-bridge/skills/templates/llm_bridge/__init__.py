"""
LLM Bridge — Lightweight multi-provider LLM gateway.

Public API::

    from llm_bridge import LLMBridge, ChatParameters, UnifiedResponse

    bridge = LLMBridge.from_config("llm_bridge_config.yaml")
    response = await bridge.chat(
        "openai/gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
        params=ChatParameters(temperature=0.5, max_tokens=1024),
    )
"""

from .bridge import LLMBridge
from .models import (
    BridgeConfig,
    ChatParameters,
    ModelConfig,
    RetryConfig,
    UnifiedResponse,
    UsageInfo,
)

__all__ = [
    "LLMBridge",
    "ChatParameters",
    "UnifiedResponse",
    "UsageInfo",
    "BridgeConfig",
    "ModelConfig",
    "RetryConfig",
]
