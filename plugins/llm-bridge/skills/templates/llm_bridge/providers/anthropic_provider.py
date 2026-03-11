"""
LLM Bridge — Anthropic provider adapter.

Uses the ``anthropic`` SDK.

**Parameter mapping (2026):**
- ``max_tokens`` → ``max_tokens``  (REQUIRED by Anthropic, defaults to 4096)
- ``temperature`` → ``temperature``  (0.0–1.0, default 1.0)
- ``top_k`` → ``top_k``  (natively supported)
- ``top_p`` → ``top_p``  (natively supported)
- ``stop_sequences`` → ``stop_sequences``
- ``system_prompt`` → ``system`` (top-level parameter, NOT a message role)
- ``frequency_penalty`` / ``presence_penalty`` / ``seed`` → NOT supported

**Structured output:**
Uses ``client.beta.messages.parse(response_format=…)`` with the
``anthropic-beta: structured-outputs-2025-11-13`` header.
Returns ``response.parsed_output``.

**Response mapping:**
- ``usage.input_tokens`` → ``input_tokens``
- ``usage.output_tokens`` → ``output_tokens``
- ``usage.cache_creation_input_tokens`` → ``cache_creation_input_tokens``
- ``usage.cache_read_input_tokens`` → ``cache_read_input_tokens``
- ``stop_reason`` → ``stop_reason``  (values: "end_turn", "max_tokens", "stop_sequence")
- ``id`` → ``model_id``  (format: "msg_…")
"""

from __future__ import annotations

from typing import Optional, Type

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from ..models import ChatParameters, UnifiedResponse, UsageInfo
from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Adapter for the Anthropic API (Claude Sonnet 4, Claude Opus, etc.)."""

    # Beta header required for structured output support.
    _BETA_HEADER = "structured-outputs-2025-11-13"

    async def chat(
        self,
        model: str,
        messages: list[dict],
        api_key: str,
        *,
        response_model: Optional[Type[BaseModel]] = None,
        params: Optional[ChatParameters] = None,
        **kwargs,
    ) -> UnifiedResponse:
        client = AsyncAnthropic(api_key=api_key)
        p = params or ChatParameters()

        # ── Build SDK kwargs ──────────────────────────────────────
        sdk_kwargs: dict = {**kwargs}

        # max_tokens is REQUIRED by Anthropic
        sdk_kwargs["max_tokens"] = p.max_tokens if p.max_tokens is not None else 4096

        if p.temperature is not None:
            sdk_kwargs["temperature"] = p.temperature
        if p.top_k is not None:
            sdk_kwargs["top_k"] = p.top_k
        if p.top_p is not None:
            sdk_kwargs["top_p"] = p.top_p
        if p.stop_sequences is not None:
            sdk_kwargs["stop_sequences"] = p.stop_sequences
        # system prompt is a top-level parameter (NOT a message role)
        if p.system_prompt is not None:
            sdk_kwargs["system"] = p.system_prompt
        # frequency_penalty, presence_penalty, seed → NOT supported by Anthropic

        # ── Call SDK ──────────────────────────────────────────────
        if response_model is not None:
            response = await client.beta.messages.parse(
                model=model,
                messages=messages,
                response_format=response_model,
                extra_headers={"anthropic-beta": self._BETA_HEADER},
                **sdk_kwargs,
            )
            parsed = response.parsed_output
            # Anthropic structured output may also carry text blocks
            content = None
            for block in response.content:
                if hasattr(block, "text"):
                    content = block.text
                    break
        else:
            response = await client.messages.create(
                model=model,
                messages=messages,
                **sdk_kwargs,
            )
            parsed = None
            content_text = None
            thinking_text = None
            
            if hasattr(response, "content") and isinstance(response.content, list):
                text_blocks = []
                thinking_blocks = []
                for block in response.content:
                    if getattr(block, "type", "") == "text":
                        text_blocks.append(getattr(block, "text", ""))
                    elif getattr(block, "type", "") == "thinking":
                        thinking_blocks.append(getattr(block, "thinking", ""))
                
                if text_blocks:
                    content_text = "\n\n".join(text_blocks)
                if thinking_blocks:
                    thinking_text = "\n\n".join(thinking_blocks)
            elif response.content:
                content_text = response.content[0].text if hasattr(response.content[0], "text") else None

        # ── Map response ──────────────────────────────────────────
        usage = response.usage
        input_tok = getattr(usage, "input_tokens", 0) or 0
        output_tok = getattr(usage, "output_tokens", 0) or 0
        return UnifiedResponse(
            provider="anthropic",
            model=model,
            content=content_text,
            parsed=parsed,
            thinking=thinking_text,
            usage=UsageInfo(
                input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
                output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
                total_tokens=(
                    getattr(response.usage, "input_tokens", 0)
                    + getattr(response.usage, "output_tokens", 0)
                ),
                cache_creation_input_tokens=getattr(
                    response.usage, "cache_creation_input_tokens", 0
                )
                or 0,
                cache_read_input_tokens=getattr(
                    response.usage, "cache_read_input_tokens", 0
                )
                or 0,
            ),
            stop_reason=getattr(response, "stop_reason", None),
            model_id=getattr(response, "id", None),
        )

    async def list_models(self, api_key: str, **kwargs) -> list[str]:
        client = AsyncAnthropic(api_key=api_key)
        response = await client.models.list()
        return [m.id for m in response.data]
