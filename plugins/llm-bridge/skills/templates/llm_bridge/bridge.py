"""
LLM Bridge — Main entry point.

``LLMBridge`` is the **only** class external code needs to interact with.
It dispatches calls to the correct provider, handles key rotation,
retry with exponential backoff, and dynamic fallback.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Optional, Type, Union

from pydantic import BaseModel

from .config import load_config
from .models import BridgeConfig, ChatParameters, ModelConfig, UnifiedResponse
from .providers import PROVIDER_REGISTRY, BaseProvider
from .router import KeyRotator, fallback_chain, retry_with_backoff

logger = logging.getLogger("llm_bridge")


class LLMBridge:
    """Unified, multi-provider LLM gateway.

    ``LLMBridge`` is the single entry-point for all LLM interactions.
    It routes requests to OpenAI, Anthropic, or Google based on the
    model string, handles API-key rotation, exponential-backoff retry,
    and dynamic fallback across models.

    Examples
    --------
    **Basic text completion:**

    .. code-block:: python

        import asyncio
        from llm_bridge import LLMBridge

        async def main():
            bridge = LLMBridge.from_config("llm_bridge_config.yaml")

            # Simple text response
            response = await bridge.chat(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello!"}],
            )
            print(response.content)        # "Hello! How can I help you?"
            print(response.usage)          # UsageInfo(input_tokens=10, ...)
            print(response.stop_reason)    # "stop"
            print(response.model_id)       # "chatcmpl-abc123..."

        asyncio.run(main())

    **Structured output with Pydantic:**

    .. code-block:: python

        from pydantic import BaseModel
        from llm_bridge import LLMBridge, ChatParameters

        class MovieReview(BaseModel):
            title: str
            rating: float
            pros: list[str]
            cons: list[str]

        async def main():
            bridge = LLMBridge.from_config("llm_bridge_config.yaml")

            response = await bridge.chat(
                model="google/gemini-2.0-flash",
                messages=[{"role": "user", "content": "Review the movie Inception"}],
                response_model=MovieReview,
                params=ChatParameters(
                    temperature=0.3,
                    max_tokens=2048,
                    system_prompt="You are a professional film critic.",
                ),
            )
            review = response.parsed  # MovieReview instance
            print(f"{review.title}: {review.rating}/10")
            print(f"Pros: {review.pros}")

    **Using a config alias with fallback:**

    .. code-block:: python

        # In llm_bridge_config.yaml:
        #   models:
        #     fast:
        #       provider: openai
        #       model: gpt-4o
        #       fallback_models: [gemini_flash]

        response = await bridge.chat(
            model="fast",  # uses config alias
            messages=[{"role": "user", "content": "Summarize this text..."}],
            params=ChatParameters(max_tokens=500, top_p=0.9),
        )
    """

    def __init__(self, config: BridgeConfig) -> None:
        self._config = config
        self._providers: dict[str, BaseProvider] = {}
        self._key_rotators: dict[str, KeyRotator] = {}

        # Pre-instantiate providers and key rotators from config.
        for alias, model_cfg in config.models.items():
            provider_name = model_cfg.provider.lower()
            if provider_name not in PROVIDER_REGISTRY:
                raise ValueError(
                    f"Unknown provider '{provider_name}' for model '{alias}'. "
                    f"Available: {list(PROVIDER_REGISTRY.keys())}"
                )
            self._providers[alias] = PROVIDER_REGISTRY[provider_name]()
            if model_cfg.api_keys:
                self._key_rotators[alias] = KeyRotator(model_cfg.api_keys)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, path: Union[str, Path]) -> "LLMBridge":
        """Create an ``LLMBridge`` from a YAML configuration file.

        Parameters
        ----------
        path :
            Path to a ``llm_bridge_config.yaml`` file.  Environment
            variables like ``${OPENAI_API_KEY}`` are resolved automatically.
        """
        return cls(load_config(path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        response_model: Optional[Type[BaseModel]] = None,
        params: Optional[ChatParameters] = None,
        **kwargs,
    ) -> UnifiedResponse:
        """Send a chat request through the bridge.

        This is the **primary method** for all LLM interactions.

        Parameters
        ----------
        model :
            Either a config alias (e.g. ``"gpt4o"``, ``"fast"``) defined
            in your YAML config, or an inline ``"provider/model"`` string
            (e.g. ``"openai/gpt-4o"``, ``"anthropic/claude-sonnet-4"``).

        messages :
            OpenAI-style message list::

                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello!"},
                ]

            Note: for Anthropic, system messages should be passed via
            ``params.system_prompt`` instead of inline messages.

        response_model :
            Optional Pydantic model class.  When provided, the bridge
            requests structured (JSON) output from the provider and
            returns the parsed model in ``response.parsed``.

        params :
            Optional ``ChatParameters`` controlling generation behaviour.
            All fields are optional and provider-agnostic — the bridge
            maps them to the correct SDK-specific names automatically::

                params = ChatParameters(
                    max_tokens=2048,          # → OpenAI: max_completion_tokens
                                              # → Anthropic: max_tokens
                                              # → Google: max_output_tokens
                    temperature=0.5,
                    top_p=0.9,
                    top_k=40,                 # Anthropic & Google only
                    frequency_penalty=0.3,    # OpenAI & Google only
                    presence_penalty=0.3,     # OpenAI & Google only
                    stop_sequences=["END"],
                    seed=42,                  # OpenAI & Google only
                    system_prompt="Be concise.",
                )

        **kwargs :
            Additional provider-specific arguments forwarded directly
            to the SDK call.

        Returns
        -------
        UnifiedResponse
            Contains ``content``, ``parsed``, ``usage``, ``stop_reason``,
            and ``model_id`` regardless of which provider was used.

        Raises
        ------
        RuntimeError
            If all retry attempts and fallback models are exhausted.
        ValueError
            If the model string cannot be resolved.

        Example
        -------
        >>> response = await bridge.chat(
        ...     model="openai/gpt-4o",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     params=ChatParameters(temperature=0.2, max_tokens=1024),
        ... )
        >>> print(response.content)
        >>> print(response.usage.total_tokens)
        """

        model_cfg = self._resolve_model(model)
        alias = model  # keep for logging

        # Build the primary call
        primary = functools.partial(
            self._call_model, model_cfg, messages, response_model, params, **kwargs
        )

        # Build fallback callables
        fallback_fns = []
        for fb_model_str in model_cfg.fallback_models:
            fb_cfg = self._resolve_model(fb_model_str)
            fallback_fns.append(
                functools.partial(
                    self._call_model, fb_cfg, messages, response_model, params, **kwargs
                )
            )

        if fallback_fns:
            response = await fallback_chain(primary, fallback_fns)
        else:
            response = await primary()

        # Optional token-usage logging
        if self._config.log_usage:
            logger.info(
                "[%s] %s/%s — in: %d  out: %d  total: %d  reason: %s",
                alias,
                response.provider,
                response.model,
                response.usage.input_tokens,
                response.usage.output_tokens,
                response.usage.total_tokens,
                response.stop_reason,
            )

        return response

    async def get_available_models(self) -> dict[str, list[str]]:
        """Query all configured providers for their available remote models.

        Returns
        -------
        dict[str, list[str]]
            A mapping of provider name (e.g., 'openai', 'anthropic', 'google')
            to a list of their supported model IDs.
        """
        results = {}
        for alias, model_cfg in self._config.models.items():
            provider_name = model_cfg.provider.lower()
            if provider_name in results:
                continue  # Already queried this provider

            # Get an API key for this provider to make the query
            rotator = self._key_rotators.get(alias)
            api_key = rotator.next() if rotator else (model_cfg.api_keys[0] if model_cfg.api_keys else None)
            
            if not api_key:
                continue

            provider = self._providers[alias]
            try:
                models = await provider.list_models(api_key=api_key, **model_cfg.extra)
                results[provider_name] = models
            except Exception as e:
                logger.warning("Failed to list models for '%s': %s", provider_name, e)
                results[provider_name] = []

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model(self, model: str) -> ModelConfig:
        """Resolve a model alias or ``provider/model`` string."""
        # 1. Check config aliases
        if model in self._config.models:
            return self._config.models[model]

        # 2. Parse inline "provider/model"
        if "/" in model:
            provider, model_name = model.split("/", 1)
            return ModelConfig(provider=provider, model=model_name)

        # 3. Try default
        if (
            self._config.default_model
            and self._config.default_model in self._config.models
        ):
            return self._config.models[self._config.default_model]

        raise ValueError(
            f"Cannot resolve model '{model}'. "
            f"Known aliases: {list(self._config.models.keys())}"
        )

    async def _call_model(
        self,
        model_cfg: ModelConfig,
        messages: list[dict],
        response_model: Optional[Type[BaseModel]],
        params: Optional[ChatParameters],
        **kwargs,
    ) -> UnifiedResponse:
        """Execute a single model call with retry + key rotation."""
        provider_name = model_cfg.provider.lower()
        provider_cls = PROVIDER_REGISTRY.get(provider_name)
        if provider_cls is None:
            raise ValueError(f"No provider registered for '{provider_name}'.")

        adapter = provider_cls()

        # Determine API key (rotate if multiple)
        alias = f"{model_cfg.provider}/{model_cfg.model}"
        rotator = self._key_rotators.get(alias)
        if rotator:
            api_key = rotator.next()
        elif model_cfg.api_keys:
            api_key = model_cfg.api_keys[0]
        else:
            api_key = ""

        # Build default ChatParameters from config if caller didn't provide
        effective_params = params or ChatParameters()
        if effective_params.max_tokens is None and model_cfg.max_tokens:
            effective_params = effective_params.model_copy(
                update={"max_tokens": model_cfg.max_tokens}
            )
        if effective_params.temperature is None and model_cfg.temperature is not None:
            effective_params = effective_params.model_copy(
                update={"temperature": model_cfg.temperature}
            )

        # Merge extra kwargs from config
        merged_kwargs = {**model_cfg.extra, **kwargs}

        async def _do_call() -> UnifiedResponse:
            return await adapter.chat(
                model=model_cfg.model,
                messages=messages,
                api_key=api_key,
                response_model=response_model,
                params=effective_params,
                **merged_kwargs,
            )

        return await retry_with_backoff(
            _do_call,
            retry_config=self._config.retry,
        )
