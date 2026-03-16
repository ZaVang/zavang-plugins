"""
LLM Bridge — Abstract base provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Type

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..models import ChatParameters, UnifiedResponse


class BaseProvider(ABC):
    """Interface that every LLM provider adapter must implement."""

    @abstractmethod
    async def chat(
        self,
        model: str,
        messages: list[dict],
        api_key: str,
        *,
        response_model: Optional[Type[BaseModel]] = None,
        params: Optional["ChatParameters"] = None,
        **kwargs,
    ) -> "UnifiedResponse":
        """Send a chat completion request and return a ``UnifiedResponse``.

        Parameters
        ----------
        model :
            Model identifier passed to the provider SDK (e.g. ``"gpt-4o"``).
        messages :
            OpenAI-style message list: ``[{"role": "user", "content": "…"}]``.
        api_key :
            API key for authentication.
        response_model :
            Optional Pydantic model class for structured (JSON) output.
        params :
            Optional ``ChatParameters`` containing temperature, top_p, etc.
        **kwargs :
            Any extra provider-specific arguments.
        """
        ...

    @abstractmethod
    async def list_models(self, api_key: str, **kwargs) -> list[str]:
        """Fetch a list of available model IDs from the provider.
        
        Parameters
        ----------
        api_key : str
            API key for authentication.
        **kwargs :
            Extra optional arguments (e.g. project, location for Vertex AI).
        """
        ...
