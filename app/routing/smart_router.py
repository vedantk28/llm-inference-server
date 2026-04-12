"""
Smart Router — dual-signal GPU saturation detection.

Uses two complementary signals to decide whether to route locally or to Azure:

1. ACTIVE REQUEST COUNTER (primary, zero-cost):
   In-memory counter tracking how many requests are currently being processed
   by the local backend. If this exceeds MAX_CONCURRENT_LOCAL, route to Azure.
   This is instant — no I/O, no system calls.

2. GPU UTILIZATION via pynvml (secondary, cached):
   Catches cases where the GPU is hot from *other* processes (e.g., a training
   job running alongside inference). Uses the 5-second TTL-cached reading from
   GPUMonitor to avoid hammering pynvml.

The combination covers both scenarios:
  - Many inference requests → counter catches it
  - External GPU load → pynvml catches it
"""

import asyncio
import logging
from typing import Optional

from app.backends.azure_backend import AzureBackend
from app.backends.base import InferenceBackend
from app.config import settings
from app.gpu.monitor import GPUMonitor
from app.metrics.prometheus import ACTIVE_REQUESTS, REQUEST_REJECTED

logger = logging.getLogger(__name__)


class SmartRouter:
    """
    Routes inference requests to the best available backend.
    
    Decision flow:
      1. Queue full? → Azure
      2. GPU hot?    → Azure
      3. Backend down? → Azure
      4. Otherwise    → Local
    """

    def __init__(
        self,
        local_backend: InferenceBackend,
        azure_backend: Optional[AzureBackend],
        gpu_monitor: GPUMonitor,
        max_concurrent: int = 4,
        gpu_threshold: float = 0.85,
    ):
        self._local = local_backend
        self._azure = azure_backend
        self._gpu_monitor = gpu_monitor
        self._max_concurrent = max_concurrent
        self._gpu_threshold = gpu_threshold

        # Atomic counter for active local requests
        self._active_count = 0
        self._lock = asyncio.Lock()

    @property
    def active_local_requests(self) -> int:
        """Current number of active requests on the local backend."""
        return self._active_count

    async def _increment(self) -> None:
        """Thread-safe increment of active request counter."""
        async with self._lock:
            self._active_count += 1
            ACTIVE_REQUESTS.labels(backend="local").set(self._active_count)

    async def _decrement(self) -> None:
        """Thread-safe decrement of active request counter."""
        async with self._lock:
            self._active_count = max(0, self._active_count - 1)
            ACTIVE_REQUESTS.labels(backend="local").set(self._active_count)

    async def route(self, **kwargs) -> tuple[InferenceBackend, str]:
        """
        Determine which backend should handle the request.
        
        Returns:
            Tuple of (backend_instance, backend_name)
            backend_name is one of: "ollama", "vllm", "azure"
        """
        # Signal 1: Check active request queue (zero-cost)
        if self._active_count >= self._max_concurrent:
            logger.info(
                f"Local queue full ({self._active_count}/{self._max_concurrent}), "
                f"routing to Azure"
            )
            if self._azure and await self._azure.health_check():
                REQUEST_REJECTED.labels(reason="queue_full").inc()
                return self._azure, "azure"
            else:
                logger.warning("Azure fallback not available — queuing locally anyway")

        # Signal 2: Check GPU utilization (cached, ~1ms worst case)
        gpu_util = self._gpu_monitor.get_utilization()
        if gpu_util > self._gpu_threshold:
            logger.info(
                f"GPU utilization {gpu_util:.1%} > threshold {self._gpu_threshold:.1%}, "
                f"routing to Azure"
            )
            if self._azure and await self._azure.health_check():
                return self._azure, "azure"
            else:
                logger.warning(
                    "GPU saturated but Azure unavailable — proceeding locally"
                )

        # Signal 3: Check local backend health
        if not await self._local.health_check():
            logger.warning("Local backend health check failed, routing to Azure")
            if self._azure and await self._azure.health_check():
                REQUEST_REJECTED.labels(reason="backend_unavailable").inc()
                return self._azure, "azure"
            else:
                raise RuntimeError(
                    "Both local and Azure backends are unavailable"
                )

        # All clear — use local backend
        return self._local, settings.active_backend.value

    async def execute_local(self, coro):
        """
        Execute a coroutine against the local backend with request counting.
        
        Wraps the call with increment/decrement to track active requests.
        Guarantees decrement even if the coroutine raises an exception.
        """
        await self._increment()
        try:
            return await coro
        finally:
            await self._decrement()
