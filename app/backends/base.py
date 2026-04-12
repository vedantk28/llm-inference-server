"""
Abstract base class for inference backends.

All backends (Ollama, vLLM, Azure) implement this interface.
GPU monitoring is intentionally NOT part of this interface — it's handled
separately by app.gpu.monitor.GPUMonitor to avoid coupling inference with
hardware monitoring.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Optional


class InferenceBackend(ABC):
    """
    Abstract interface for LLM inference backends.
    
    Each backend translates the unified request format into its native API,
    handles both streaming and non-streaming responses, and exposes a
    health check endpoint.
    """

    @abstractmethod
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """
        Generate a chat completion.
        
        Args:
            messages: List of {"role": "...", "content": "..."} messages
            model: Model identifier (e.g. "mistral:latest", "gpt-4o-mini")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            stream: If True, return an async generator yielding chunks
            
        Returns:
            If stream=False: dict with OpenAI-compatible completion response
            If stream=True: AsyncGenerator yielding OpenAI-compatible chunk dicts
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the backend is reachable and ready to serve.
        
        Returns:
            True if healthy, False otherwise
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (HTTP clients, connections, etc.)."""
        ...
