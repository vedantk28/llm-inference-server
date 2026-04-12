"""
vLLM inference backend.

vLLM exposes an OpenAI-compatible API out of the box, so this backend
is essentially a thin proxy client that forwards requests.

What makes vLLM special (learning objectives):
  - PagedAttention: Pages KV cache like virtual memory — eliminates fragmentation,
    allows parallel requests without OOM.
  - Continuous batching: Dynamically batches incoming requests on the GPU,
    serving multiple users concurrently.
  - KV cache quantization: Using --kv-cache-dtype fp8 stores attention cache
    in 8-bit float, saving ~50% cache memory.
  - Weight quantization: AWQ/GPTQ stores model weights in INT4, 4× smaller
    than fp16.
"""

import logging
import time
import uuid
from typing import Any, AsyncGenerator

import httpx

from app.backends.base import InferenceBackend

logger = logging.getLogger(__name__)


class VLLMBackend(InferenceBackend):
    """
    vLLM OpenAI-compatible API client.
    
    Since vLLM serves an OpenAI-compatible API, this backend simply
    forwards requests in OpenAI format — no translation needed.
    """

    def __init__(self, base_url: str = "http://localhost:8001"):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=120.0,
                write=10.0,
                pool=10.0,
            ),
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
        Send a chat completion request to vLLM.
        
        vLLM already speaks OpenAI format, so we pass-through directly.
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        if stream:
            return self._stream_completion(payload)
        else:
            return await self._non_stream_completion(payload)

    async def _non_stream_completion(self, payload: dict) -> dict[str, Any]:
        """Forward non-streaming request to vLLM."""
        response = await self._client.post(
            "/v1/chat/completions", json=payload
        )
        response.raise_for_status()
        return response.json()

    async def _stream_completion(
        self, payload: dict
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Forward streaming request to vLLM and yield chunks.
        
        vLLM streams SSE (Server-Sent Events) in OpenAI format:
          data: {"id": "...", "choices": [...], ...}
          data: [DONE]
        """
        import json

        async with self._client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue

                # SSE format: "data: {json}" or "data: [DONE]"
                if line.startswith("data: "):
                    data_str = line[6:]  # Strip "data: " prefix
                    if data_str == "[DONE]":
                        return
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse vLLM chunk: {data_str}")
                        continue

    async def health_check(self) -> bool:
        """Check if vLLM server is running via its health endpoint."""
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
