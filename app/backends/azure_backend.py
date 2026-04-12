"""
Azure OpenAI inference backend (fallback).

Uses the official openai Python SDK in Azure mode.
Activated by the smart router when the local GPU is saturated
(either queue full or GPU utilization above threshold).
"""

import logging
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from app.backends.base import InferenceBackend

logger = logging.getLogger(__name__)


class AzureBackend(InferenceBackend):
    """
    Azure OpenAI backend using the official SDK.
    
    Requires:
      - AZURE_OPENAI_ENDPOINT
      - AZURE_OPENAI_API_KEY
      - AZURE_OPENAI_DEPLOYMENT_NAME
      - AZURE_OPENAI_API_VERSION
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment_name: str,
        api_version: str = "2024-12-01-preview",
    ):
        self._deployment_name = deployment_name
        self._client = None

        try:
            from openai import AsyncAzureOpenAI

            self._client = AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
            logger.info(
                f"Azure OpenAI backend initialized: "
                f"deployment='{deployment_name}', endpoint='{endpoint}'"
            )
        except ImportError:
            logger.error(
                "openai package not installed — Azure fallback unavailable. "
                "Install with: pip install openai"
            )

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """
        Send a chat completion request to Azure OpenAI.
        
        The model parameter is ignored — Azure uses the deployment name instead.
        """
        if not self._client:
            raise RuntimeError("Azure OpenAI client not initialized")

        if stream:
            return self._stream_completion(messages, max_tokens, temperature)
        else:
            return await self._non_stream_completion(messages, max_tokens, temperature)

    async def _non_stream_completion(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Non-streaming Azure OpenAI completion."""
        response = await self._client.chat.completions.create(
            model=self._deployment_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )

        # Convert the SDK response object to a dict
        return {
            "id": response.id,
            "object": "chat.completion",
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content or "",
                    },
                    "finish_reason": choice.finish_reason,
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

    async def _stream_completion(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Streaming Azure OpenAI completion yielding OpenAI-format chunks."""
        stream = await self._client.chat.completions.create(
            model=self._deployment_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            yield {
                "id": chunk.id,
                "object": "chat.completion.chunk",
                "created": chunk.created,
                "model": chunk.model,
                "choices": [
                    {
                        "index": choice.index,
                        "delta": {
                            "role": getattr(choice.delta, "role", None),
                            "content": getattr(choice.delta, "content", None),
                        },
                        "finish_reason": choice.finish_reason,
                    }
                    for choice in chunk.choices
                ],
                **(
                    {
                        "usage": {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens,
                        }
                    }
                    if chunk.usage
                    else {}
                ),
            }

    async def health_check(self) -> bool:
        """
        Check Azure OpenAI availability.
        
        We can't easily ping Azure without making an API call,
        so we check if the client is initialized and credentials are set.
        """
        return self._client is not None

    async def close(self) -> None:
        """Close the Azure OpenAI client."""
        if self._client:
            await self._client.close()
