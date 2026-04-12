"""
Benchmark endpoint — structured load testing with metrics export.

POST /benchmark     — run a benchmark
GET  /benchmark/results — get last benchmark results

Runs configurable concurrent requests against the gateway itself,
measures latency/throughput, and pushes results to Prometheus.
"""

import asyncio
import json
import logging
import statistics
import time
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import settings
from app.metrics.prometheus import BENCHMARK_LATENCY_P95, BENCHMARK_THROUGHPUT

logger = logging.getLogger(__name__)

router = APIRouter()

# Store last benchmark result in memory
_last_result: Optional[dict] = None


class BenchmarkRequest(BaseModel):
    """Benchmark configuration."""

    prompt: str = Field(
        default="Explain the concept of recursion in programming.",
        description="Prompt to use for benchmark requests",
    )
    num_requests: int = Field(
        default=10, ge=1, le=100, description="Total number of requests"
    )
    concurrency: int = Field(
        default=2, ge=1, le=20, description="Number of concurrent requests"
    )
    max_tokens: int = Field(
        default=128, ge=1, le=4096, description="Max tokens per request"
    )
    model: Optional[str] = Field(
        default=None, description="Model to benchmark (defaults to server default)"
    )


class BenchmarkResult(BaseModel):
    """Benchmark results."""

    total_requests: int
    successful: int
    failed: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_requests_per_sec: float
    throughput_tokens_per_sec: float
    total_input_tokens: int
    total_output_tokens: int
    cost_per_1k_tokens_usd: float
    backend_used: str
    model: str
    duration_seconds: float


@router.post("/benchmark")
async def run_benchmark(request: Request, body: BenchmarkRequest):
    """
    Run a structured benchmark against the gateway.
    
    Sends num_requests concurrent requests and measures latency,
    throughput, and cost. Results are also pushed to Prometheus.
    """
    global _last_result

    model = body.model or settings.default_model
    port = settings.port

    # Build the request payload
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": body.prompt}],
        "max_tokens": body.max_tokens,
        "temperature": 0.1,  # Low temp for consistent benchmarks
        "stream": False,
    }

    # Semaphore to control concurrency
    semaphore = asyncio.Semaphore(body.concurrency)
    results = []

    async def single_request(client: httpx.AsyncClient, idx: int) -> dict:
        """Execute a single benchmark request."""
        async with semaphore:
            start = time.monotonic()
            try:
                response = await client.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json=payload,
                    timeout=120.0,
                )
                duration_ms = (time.monotonic() - start) * 1000

                if response.status_code == 200:
                    data = response.json()
                    usage = data.get("usage", {})
                    backend = response.headers.get("x-backend", "unknown")
                    return {
                        "idx": idx,
                        "status": "success",
                        "latency_ms": duration_ms,
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "backend": backend,
                    }
                else:
                    return {
                        "idx": idx,
                        "status": "error",
                        "latency_ms": duration_ms,
                        "error": response.text,
                        "status_code": response.status_code,
                    }
            except Exception as e:
                duration_ms = (time.monotonic() - start) * 1000
                return {
                    "idx": idx,
                    "status": "error",
                    "latency_ms": duration_ms,
                    "error": str(e),
                }

    # Run all requests
    logger.info(
        f"Starting benchmark: {body.num_requests} requests, "
        f"concurrency={body.concurrency}, model={model}"
    )

    overall_start = time.monotonic()

    async with httpx.AsyncClient() as client:
        tasks = [single_request(client, i) for i in range(body.num_requests)]
        results = await asyncio.gather(*tasks)

    overall_duration = time.monotonic() - overall_start

    # Analyze results
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]

    if not successful:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "All benchmark requests failed",
                "failures": [r.get("error", "unknown") for r in failed[:5]],
            },
        )

    latencies = [r["latency_ms"] for r in successful]
    latencies.sort()

    total_input = sum(r.get("input_tokens", 0) for r in successful)
    total_output = sum(r.get("output_tokens", 0) for r in successful)
    total_tokens = total_input + total_output

    # Determine predominant backend
    backends_used = [r.get("backend", "unknown") for r in successful]
    backend_used = max(set(backends_used), key=backends_used.count)

    # Calculate cost
    if backend_used == "azure":
        total_cost = (
            (total_input / 1000) * settings.azure_cost_per_1k_input
            + (total_output / 1000) * settings.azure_cost_per_1k_output
        )
    else:
        total_cost = (
            (total_input / 1000) * settings.local_cost_per_1k_input
            + (total_output / 1000) * settings.local_cost_per_1k_output
        )

    cost_per_1k = (total_cost / total_tokens * 1000) if total_tokens > 0 else 0

    # Percentile helpers
    def percentile(data, pct):
        idx = int(len(data) * pct / 100)
        return data[min(idx, len(data) - 1)]

    result = BenchmarkResult(
        total_requests=body.num_requests,
        successful=len(successful),
        failed=len(failed),
        avg_latency_ms=round(statistics.mean(latencies), 1),
        p50_latency_ms=round(percentile(latencies, 50), 1),
        p95_latency_ms=round(percentile(latencies, 95), 1),
        p99_latency_ms=round(percentile(latencies, 99), 1),
        min_latency_ms=round(min(latencies), 1),
        max_latency_ms=round(max(latencies), 1),
        throughput_requests_per_sec=round(len(successful) / overall_duration, 2),
        throughput_tokens_per_sec=round(total_tokens / overall_duration, 2),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        cost_per_1k_tokens_usd=round(cost_per_1k, 6),
        backend_used=backend_used,
        model=model,
        duration_seconds=round(overall_duration, 2),
    )

    # Push to Prometheus
    BENCHMARK_THROUGHPUT.set(result.throughput_tokens_per_sec)
    BENCHMARK_LATENCY_P95.set(result.p95_latency_ms)

    # Cache the result
    _last_result = result.model_dump()

    logger.info(
        f"Benchmark complete: {result.successful}/{result.total_requests} successful, "
        f"avg={result.avg_latency_ms}ms, p95={result.p95_latency_ms}ms, "
        f"throughput={result.throughput_tokens_per_sec} tok/s"
    )

    return result


@router.get("/benchmark/results")
async def get_benchmark_results():
    """Return the last benchmark results, or 404 if none have been run."""
    if _last_result is None:
        raise HTTPException(
            status_code=404,
            detail="No benchmark results available. Run POST /benchmark first.",
        )
    return _last_result
