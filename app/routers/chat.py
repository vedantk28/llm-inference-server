"""
Chat completions endpoint — OpenAI-compatible.

POST /v1/chat/completions

Supports:
  - Non-streaming: Returns full JSON response
  - Streaming (SSE): Returns text/event-stream with data: {chunk}\n\n format
  
Headers added:
  - X-Tokens-Used: total tokens consumed
  - X-Tokens-Remaining: budget remaining
  - X-Tokens-Budget: configured budget
  - X-Backend: which backend handled the request (ollama/vllm/azure)
  - X-Queue-Depth: current active local requests
"""

import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.metrics.prometheus import (
    COST_DOLLARS,
    REQUEST_DURATION,
    REQUEST_TOTAL,
    TOKENS_INPUT,
    TOKENS_OUTPUT,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Request/Response Models ---


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request body."""

    model: Optional[str] = Field(
        default=None,
        description="Model to use. Defaults to server's DEFAULT_MODEL.",
    )
    messages: list[ChatMessage] = Field(
        ..., description="List of chat messages", min_length=1
    )
    max_tokens: int = Field(
        default=256, ge=1, le=16384, description="Max tokens to generate"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    stream: bool = Field(
        default=False, description="Enable SSE streaming"
    )


# --- Endpoints ---


@router.post("/v1/chat/completions")
async def chat_completion(request: Request, body: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint.
    
    Routes through:
      1. Token budget check → 400 if over budget
      2. Smart router → selects backend (local or Azure)
      3. Backend inference → streaming or non-streaming
      4. Metrics recording
    """
    # Resolve model
    model = body.model or settings.default_model

    # Get shared components from app state
    token_checker = request.app.state.token_checker
    smart_router = request.app.state.smart_router

    # --- Token Budget Check ---
    messages_dicts = [m.model_dump() for m in body.messages]
    within_budget, input_tokens, error_msg = token_checker.check(
        messages_dicts, body.max_tokens
    )

    if not within_budget:
        from app.metrics.prometheus import REQUEST_REJECTED

        REQUEST_REJECTED.labels(reason="budget_exceeded").inc()
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Token budget exceeded",
                "message": error_msg,
                "requested_input_tokens": input_tokens,
                "requested_max_tokens": body.max_tokens,
                "total_requested": input_tokens + body.max_tokens,
                "budget": token_checker.budget,
            },
            headers={
                "X-Tokens-Budget": str(token_checker.budget),
            },
        )

    # --- Route to Backend ---
    try:
        backend, backend_name = await smart_router.route()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    queue_depth = smart_router.active_local_requests

    # --- Execute Inference ---
    if body.stream:
        return await _handle_streaming(
            backend,
            backend_name,
            messages_dicts,
            model,
            body.max_tokens,
            body.temperature,
            input_tokens,
            token_checker.budget,
            queue_depth,
            smart_router,
        )
    else:
        return await _handle_non_streaming(
            backend,
            backend_name,
            messages_dicts,
            model,
            body.max_tokens,
            body.temperature,
            input_tokens,
            token_checker.budget,
            queue_depth,
            smart_router,
        )


async def _handle_non_streaming(
    backend,
    backend_name: str,
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    input_tokens: int,
    budget: int,
    queue_depth: int,
    smart_router,
) -> dict:
    """Handle non-streaming chat completion with metrics tracking."""
    start_time = time.monotonic()

    try:
        # Use smart_router.execute_local for local backends (tracks active count)
        if backend_name != "azure":
            result = await smart_router.execute_local(
                backend.chat_completion(
                    messages, model, max_tokens, temperature, stream=False
                )
            )
        else:
            result = await backend.chat_completion(
                messages, model, max_tokens, temperature, stream=False
            )

        duration = time.monotonic() - start_time

        # Extract token usage from response
        usage = result.get("usage", {})
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        # Record metrics
        REQUEST_TOTAL.labels(
            backend=backend_name, model=model, status="success"
        ).inc()
        REQUEST_DURATION.labels(backend=backend_name, model=model).observe(duration)
        TOKENS_INPUT.labels(backend=backend_name, model=model).inc(input_tokens)
        TOKENS_OUTPUT.labels(backend=backend_name, model=model).inc(output_tokens)

        # Record cost
        _record_cost(backend_name, model, input_tokens, output_tokens)

        # Add custom headers to response
        from fastapi.responses import JSONResponse

        response = JSONResponse(
            content=result,
            headers={
                "X-Tokens-Used": str(total_tokens),
                "X-Tokens-Remaining": str(budget - total_tokens),
                "X-Tokens-Budget": str(budget),
                "X-Backend": backend_name,
                "X-Queue-Depth": str(queue_depth),
                "X-Duration-Ms": str(int(duration * 1000)),
            },
        )
        return response

    except Exception as e:
        duration = time.monotonic() - start_time
        REQUEST_TOTAL.labels(
            backend=backend_name, model=model, status="error"
        ).inc()
        REQUEST_DURATION.labels(backend=backend_name, model=model).observe(duration)
        logger.error(f"Inference error ({backend_name}): {e}")
        raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")


async def _handle_streaming(
    backend,
    backend_name: str,
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    input_tokens: int,
    budget: int,
    queue_depth: int,
    smart_router,
) -> StreamingResponse:
    """
    Handle streaming chat completion via Server-Sent Events.
    
    Streams tokens as they're generated:
      data: {"id": "...", "choices": [{"delta": {"content": "Hello"}}]}
      data: {"id": "...", "choices": [{"delta": {}, "finish_reason": "stop"}]}
      data: [DONE]
    """

    async def event_generator():
        start_time = time.monotonic()
        output_tokens = 0

        try:
            # Get the streaming generator
            if backend_name != "azure":
                await smart_router._increment()

            try:
                stream = await backend.chat_completion(
                    messages, model, max_tokens, temperature, stream=True
                )

                async for chunk in stream:
                    # Count output tokens from streamed chunks
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            # Rough count: ~1 token per chunk for most backends
                            output_tokens += 1

                    yield f"data: {json.dumps(chunk)}\n\n"

                # Send the [DONE] sentinel
                yield "data: [DONE]\n\n"

            finally:
                if backend_name != "azure":
                    await smart_router._decrement()

            # Record metrics after stream completes
            duration = time.monotonic() - start_time
            REQUEST_TOTAL.labels(
                backend=backend_name, model=model, status="success"
            ).inc()
            REQUEST_DURATION.labels(backend=backend_name, model=model).observe(
                duration
            )
            TOKENS_INPUT.labels(backend=backend_name, model=model).inc(input_tokens)
            TOKENS_OUTPUT.labels(backend=backend_name, model=model).inc(output_tokens)
            _record_cost(backend_name, model, input_tokens, output_tokens)

        except Exception as e:
            duration = time.monotonic() - start_time
            REQUEST_TOTAL.labels(
                backend=backend_name, model=model, status="error"
            ).inc()
            REQUEST_DURATION.labels(backend=backend_name, model=model).observe(
                duration
            )
            logger.error(f"Streaming error ({backend_name}): {e}")
            error_chunk = {
                "error": {"message": str(e), "type": "backend_error"}
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Backend": backend_name,
            "X-Queue-Depth": str(queue_depth),
            "X-Tokens-Budget": str(budget),
        },
    )


def _record_cost(
    backend_name: str, model: str, input_tokens: int, output_tokens: int
) -> None:
    """Record estimated cost based on backend pricing configuration."""
    if backend_name == "azure":
        cost = (
            (input_tokens / 1000) * settings.azure_cost_per_1k_input
            + (output_tokens / 1000) * settings.azure_cost_per_1k_output
        )
    else:
        cost = (
            (input_tokens / 1000) * settings.local_cost_per_1k_input
            + (output_tokens / 1000) * settings.local_cost_per_1k_output
        )

    if cost > 0:
        COST_DOLLARS.labels(backend=backend_name, model=model).inc(cost)
