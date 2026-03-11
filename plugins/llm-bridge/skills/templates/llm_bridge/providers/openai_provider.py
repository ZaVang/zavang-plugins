"""
LLM Bridge — OpenAI provider adapter.

Uses the ``openai`` SDK v1.x.

**Parameter mapping (2026):**
- ``max_tokens`` → ``max_completion_tokens``  (``max_tokens`` is deprecated
  for o-series models and newer).
- ``stop_sequences`` → ``stop``
- ``top_k`` is NOT supported — silently ignored.
- ``system_prompt`` → injected as a message with ``role='system'``.

**Structured output:**
Uses ``client.beta.chat.completions.parse(response_format=…)``
which returns ``response.choices[0].message.parsed``.

**Response mapping:**
- ``usage.prompt_tokens`` → ``input_tokens``
- ``usage.completion_tokens`` → ``output_tokens``
- ``usage.total_tokens`` → ``total_tokens``
- ``usage.completion_tokens_details.reasoning_tokens`` → ``reasoning_tokens``
- ``choices[0].finish_reason`` → ``stop_reason``
- ``id`` → ``model_id``
"""

from __future__ import annotations

from typing import Optional, Type

from openai import AsyncOpenAI
from pydantic import BaseModel

from ..models import ChatParameters, UnifiedResponse, UsageInfo
from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """Adapter for the OpenAI API (GPT-4o, GPT-5.1, o-series, etc.)."""

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
        client = AsyncOpenAI(api_key=api_key)
        p = params or ChatParameters()

        # ── Build SDK kwargs ──────────────────────────────────────
        sdk_kwargs: dict = {**kwargs}

        # max_completion_tokens replaces deprecated max_tokens for newer models
        if p.max_tokens is not None:
            sdk_kwargs["max_completion_tokens"] = p.max_tokens

        if p.temperature is not None:
            sdk_kwargs["temperature"] = p.temperature
        if p.top_p is not None:
            sdk_kwargs["top_p"] = p.top_p
        # top_k is NOT supported by OpenAI — intentionally omitted
        if p.frequency_penalty is not None:
            sdk_kwargs["frequency_penalty"] = p.frequency_penalty
        if p.presence_penalty is not None:
            sdk_kwargs["presence_penalty"] = p.presence_penalty
        if p.stop_sequences is not None:
            sdk_kwargs["stop"] = p.stop_sequences
        if p.seed is not None:
            sdk_kwargs["seed"] = p.seed

        # Prepend system prompt as a system message if provided
        effective_messages = list(messages)
        if p.system_prompt:
            effective_messages.insert(
                0, {"role": "system", "content": p.system_prompt}
            )

        # ── Call SDK ──────────────────────────────────────────────
        if response_model is not None:
            response = await client.beta.chat.completions.parse(
                model=model,
                messages=effective_messages,
                response_format=response_model,
                **sdk_kwargs,
            )
            parsed = response.choices[0].message.parsed
            content = response.choices[0].message.content
        else:
            response = await client.chat.completions.create(
                model=model,
                messages=effective_messages,
                **sdk_kwargs,
            )
            parsed = None
            content = response.choices[0].message.content
            content_text = response.choices[0].message.content

        # ── Map response ──────────────────────────────────────────
           # Finish reason
        stop_reason = None
        if len(response.choices) > 0:
            stop_reason = response.choices[0].finish_reason

        return UnifiedResponse(
            provider="openai",
            model=model,
            content=content_text,
            parsed=parsed,
            thinking=None,
            usage=UsageInfo(
                input_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(response.usage, "total_tokens", 0) or 0,
                reasoning_tokens=getattr(
                    getattr(response.usage, "completion_tokens_details", None),
                    "reasoning_tokens",
                    0,
                )
                or 0,
            ),
            stop_reason=stop_reason,
            model_id=getattr(response, "id", None),
        )

    async def list_models(self, api_key: str, **kwargs) -> list[str]:
        client = AsyncOpenAI(api_key=api_key)
        response = await client.models.list()
        return [m.id for m in response.data]
