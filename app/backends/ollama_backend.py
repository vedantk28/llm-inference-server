"""
Ollama inference backend.

Communicates with the Ollama server via its HTTP API.
Translates between OpenAI's chat completion format and Ollama's native format.
Supports both streaming (SSE) and non-streaming responses.

Ollama API reference: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import logging
import time
import uuid
from typing import Any, AsyncGenerator

import httpx

from app.backends.base import InferenceBackend

logger = logging.getLogger(__name__)


class OllamaBackend(InferenceBackend):
    """
    Ollama HTTP client backend.
    
    Ollama uses its own API format, so this class handles the translation
    between OpenAI-compatible requests and Ollama's /api/chat endpoint.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=120.0,  # LLM generation can be slow
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
        Send a chat completion request to Ollama.
        
        Translates OpenAI format -> Ollama format -> OpenAI format.
        """
        # Build Ollama request payload
        ollama_payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        if stream:
            return self._stream_completion(ollama_payload, model)
        else:
            return await self._non_stream_completion(ollama_payload, model)

    async def _non_stream_completion(
        self, payload: dict, model: str
    ) -> dict[str, Any]:
        """Handle non-streaming completion — wait for full response."""
        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        # Translate Ollama response to OpenAI format
        return self._to_openai_response(data, model)

    async def _stream_completion(
        self, payload: dict, model: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Handle streaming completion — yield chunks as they arrive.
        
        Ollama streams NDJSON (newline-delimited JSON).
        We translate each chunk to OpenAI's SSE chunk format.
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                import json

                chunk_data = json.loads(line)

                # Ollama sends {"message": {"role": "assistant", "content": "..."}, "done": false}
                message = chunk_data.get("message", {})
                content = message.get("content", "")
                done = chunk_data.get("done", False)

                # Build OpenAI-compatible chunk
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content} if content else {},
                            "finish_reason": "stop" if done else None,
                        }
                    ],
                }

                # Include usage info on the final chunk if Ollama provides it
                if done and "eval_count" in chunk_data:
                    chunk["usage"] = {
                        "prompt_tokens": chunk_data.get("prompt_eval_count", 0),
                        "completion_tokens": chunk_data.get("eval_count", 0),
                        "total_tokens": (
                            chunk_data.get("prompt_eval_count", 0)
                            + chunk_data.get("eval_count", 0)
                        ),
                    }

                yield chunk

    def _to_openai_response(self, ollama_data: dict, model: str) -> dict[str, Any]:
        """Translate Ollama's response format to OpenAI's chat completion format."""
        message = ollama_data.get("message", {})

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": message.get("role", "assistant"),
                        "content": message.get("content", ""),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": ollama_data.get("prompt_eval_count", 0),
                "completion_tokens": ollama_data.get("eval_count", 0),
                "total_tokens": (
                    ollama_data.get("prompt_eval_count", 0)
                    + ollama_data.get("eval_count", 0)
                ),
            },
        }

    async def health_check(self) -> bool:
        """Check if Ollama is running by hitting its root endpoint."""
        try:
            response = await self._client.get("/")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
