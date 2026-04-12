"""
FastAPI application entry point.

Handles:
  - Application lifespan (startup/shutdown for GPU monitor, backends, tokenizer)
  - Middleware registration (CORS, Prometheus instrumentation)
  - Router mounting (chat, benchmark)
  - Health check endpoints
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import BackendType, settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Startup: Initialize GPU monitor, backends, tokenizer, smart router.
    Shutdown: Clean up resources.
    """
    logger.info("=" * 60)
    logger.info("LLM Inference Server starting up...")
    logger.info(f"  Active backend: {settings.active_backend.value}")
    logger.info(f"  Default model:  {settings.default_model}")
    logger.info(f"  Token budget:   {settings.max_tokens_budget}")
    logger.info(f"  Max concurrent: {settings.max_concurrent_local}")
    logger.info(f"  GPU threshold:  {settings.gpu_saturation_threshold:.0%}")
    logger.info(f"  Azure fallback: {'configured' if settings.azure_configured else 'not configured'}")
    logger.info("=" * 60)

    # --- Initialize GPU Monitor ---
    from app.gpu.monitor import GPUMonitor

    gpu_monitor = GPUMonitor(cache_ttl=settings.gpu_util_cache_ttl)
    app.state.gpu_monitor = gpu_monitor

    # --- Initialize Local Backend ---
    if settings.active_backend == BackendType.OLLAMA:
        from app.backends.ollama_backend import OllamaBackend

        local_backend = OllamaBackend(base_url=settings.ollama_base_url)
        logger.info(f"Ollama backend: {settings.ollama_base_url}")
    else:
        from app.backends.vllm_backend import VLLMBackend

        local_backend = VLLMBackend(base_url=settings.vllm_base_url)
        logger.info(f"vLLM backend: {settings.vllm_base_url}")

    app.state.local_backend = local_backend

    # --- Initialize Azure Backend (if configured) ---
    azure_backend = None
    if settings.azure_configured:
        from app.backends.azure_backend import AzureBackend

        azure_backend = AzureBackend(
            endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            deployment_name=settings.azure_openai_deployment_name,
            api_version=settings.azure_openai_api_version,
        )
        logger.info("Azure OpenAI fallback: configured")
    else:
        logger.info("Azure OpenAI fallback: not configured (will 503 if GPU saturated)")

    app.state.azure_backend = azure_backend

    # --- Initialize Tokenizer ---
    from app.utils.tokenizer import HybridTokenizer

    tokenizer = HybridTokenizer(
        model_name=settings.hf_tokenizer_model,
        backend=settings.active_backend.value,
    )
    app.state.tokenizer = tokenizer
    logger.info(f"Tokenizer mode: {tokenizer.mode}")

    # --- Initialize Token Budget Checker ---
    from app.middleware.token_budget import TokenBudgetChecker

    token_checker = TokenBudgetChecker(tokenizer=tokenizer)
    app.state.token_checker = token_checker

    # --- Initialize Smart Router ---
    from app.routing.smart_router import SmartRouter

    smart_router = SmartRouter(
        local_backend=local_backend,
        azure_backend=azure_backend,
        gpu_monitor=gpu_monitor,
        max_concurrent=settings.max_concurrent_local,
        gpu_threshold=settings.gpu_saturation_threshold,
    )
    app.state.smart_router = smart_router

    # Check backend health on startup
    health = await local_backend.health_check()
    if health:
        logger.info(f"✅ {settings.active_backend.value} backend is healthy")
    else:
        logger.warning(
            f"⚠️  {settings.active_backend.value} backend health check failed — "
            f"is it running?"
        )

    yield  # App is running

    # --- Shutdown ---
    logger.info("Shutting down...")
    await local_backend.close()
    if azure_backend:
        await azure_backend.close()
    gpu_monitor.shutdown()
    logger.info("Shutdown complete")


# --- Create Application ---

app = FastAPI(
    title="LLM Inference Server",
    description=(
        "Production-grade LLM inference gateway with Ollama/vLLM backends, "
        "Azure OpenAI fallback, token budgeting, and Prometheus observability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# --- Middleware ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentation — exposes /metrics endpoint
Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    excluded_handlers=["/metrics", "/health", "/docs", "/openapi.json"],
).instrument(app).expose(app)

# --- Routers ---

from app.routers.chat import router as chat_router
from app.routers.benchmark import router as benchmark_router

app.include_router(chat_router, tags=["Chat"])
app.include_router(benchmark_router, tags=["Benchmark"])


# --- Health & Info Endpoints ---


@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint."""
    local_health = await app.state.local_backend.health_check()
    azure_health = (
        await app.state.azure_backend.health_check()
        if app.state.azure_backend
        else None
    )

    gpu_util = app.state.gpu_monitor.get_utilization()
    gpu_mem = app.state.gpu_monitor.get_memory()

    # Update GPU Prometheus metrics
    from app.metrics.prometheus import (
        GPU_MEMORY_TOTAL_MB,
        GPU_MEMORY_USED_MB,
        GPU_UTILIZATION,
    )

    GPU_UTILIZATION.set(gpu_util * 100)
    GPU_MEMORY_USED_MB.set(gpu_mem["used_mb"])
    GPU_MEMORY_TOTAL_MB.set(gpu_mem["total_mb"])

    return {
        "status": "healthy" if local_health else "degraded",
        "backends": {
            settings.active_backend.value: {
                "healthy": local_health,
            },
            "azure": {
                "configured": settings.azure_configured,
                "healthy": azure_health,
            },
        },
        "gpu": {
            "utilization": f"{gpu_util:.1%}",
            "memory": gpu_mem,
        },
        "config": {
            "default_model": settings.default_model,
            "token_budget": settings.max_tokens_budget,
            "max_concurrent_local": settings.max_concurrent_local,
            "gpu_threshold": f"{settings.gpu_saturation_threshold:.0%}",
            "tokenizer_mode": app.state.tokenizer.mode,
        },
        "routing": {
            "active_local_requests": app.state.smart_router.active_local_requests,
        },
    }


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "LLM Inference Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "POST /v1/chat/completions",
            "benchmark": "POST /benchmark",
            "benchmark_results": "GET /benchmark/results",
            "metrics": "GET /metrics",
            "health": "GET /health",
        },
    }
