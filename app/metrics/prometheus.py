"""
Custom Prometheus metrics for LLM inference monitoring.

These metrics are registered once at import time and updated by the
gateway's middleware and routers. They're exposed at /metrics by
prometheus-fastapi-instrumentator.

Metric naming follows Prometheus conventions:
  - Counter: monotonically increasing (total requests, total tokens)
  - Histogram: distributions (latency)
  - Gauge: point-in-time values (GPU utilization, active requests)
"""

from prometheus_client import Counter, Gauge, Histogram

# --- Request Metrics ---

REQUEST_TOTAL = Counter(
    "llm_request_total",
    "Total LLM inference requests",
    labelnames=["backend", "model", "status"],
)

REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "LLM inference request duration in seconds",
    labelnames=["backend", "model"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

# --- Token Metrics ---

TOKENS_INPUT = Counter(
    "llm_tokens_input_total",
    "Total input tokens processed",
    labelnames=["backend", "model"],
)

TOKENS_OUTPUT = Counter(
    "llm_tokens_output_total",
    "Total output tokens generated",
    labelnames=["backend", "model"],
)

# --- Cost Metrics ---

COST_DOLLARS = Counter(
    "llm_cost_dollars_total",
    "Total estimated cost in USD",
    labelnames=["backend", "model"],
)

# --- GPU Metrics ---

GPU_UTILIZATION = Gauge(
    "llm_gpu_utilization_percent",
    "Current GPU compute utilization (0-100)",
)

GPU_MEMORY_USED_MB = Gauge(
    "llm_gpu_memory_used_mb",
    "GPU memory used in megabytes",
)

GPU_MEMORY_TOTAL_MB = Gauge(
    "llm_gpu_memory_total_mb",
    "GPU total memory in megabytes",
)

# --- Routing Metrics ---

ACTIVE_REQUESTS = Gauge(
    "llm_active_requests",
    "Current number of active inference requests",
    labelnames=["backend"],
)

REQUEST_REJECTED = Counter(
    "llm_request_rejected_total",
    "Total rejected requests",
    labelnames=["reason"],
)

# --- Benchmark Metrics ---

BENCHMARK_THROUGHPUT = Gauge(
    "llm_benchmark_throughput_tokens_per_sec",
    "Last benchmark throughput in tokens per second",
)

BENCHMARK_LATENCY_P95 = Gauge(
    "llm_benchmark_latency_p95_ms",
    "Last benchmark P95 latency in milliseconds",
)
