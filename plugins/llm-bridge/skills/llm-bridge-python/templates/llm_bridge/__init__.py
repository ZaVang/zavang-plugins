"""
LLM Bridge — Lightweight multi-provider LLM gateway.

Quick start::

    from llm_bridge import LLMBridge, ChatParameters

    bridge = LLMBridge.from_config("llm_bridge_config.yaml")
    response = await bridge.chat(
        "openai/gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
        params=ChatParameters(temperature=0.5, max_tokens=1024),
    )

Skill loading::

    response = await bridge.chat(
        "openai/gpt-4o",
        messages=[{"role": "user", "content": "Review this code: ..."}],
        skills=["code-reviewer"],
        skill_vars={"language": "Python"},
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
from .skills import LLMBridgeError, SkillLoader, SkillNotFoundError, SkillRenderError

__all__ = [
    # Core gateway
    "LLMBridge",
    "ChatParameters",
    "UnifiedResponse",
    "UsageInfo",
    "BridgeConfig",
    "ModelConfig",
    "RetryConfig",
    # Skill loader
    "SkillLoader",
    "LLMBridgeError",
    "SkillNotFoundError",
    "SkillRenderError",
]
