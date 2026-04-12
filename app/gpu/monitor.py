"""
GPU monitoring via pynvml — NVIDIA's direct C bindings.

Why pynvml instead of nvidia-smi subprocess:
  - nvidia-smi: ~200ms per call (spawns a process, parses XML output)
  - pynvml:     ~1ms per call (direct C library call via ctypes)

Under load with hundreds of requests/sec, 200ms per GPU check would
bottleneck the entire gateway. pynvml eliminates this.

The 5-second TTL cache further reduces calls — under sustained load,
we read the GPU at most once every 5 seconds regardless of request volume.
"""

import logging
from time import monotonic
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import pynvml; if unavailable (e.g. no NVIDIA GPU), degrade gracefully
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning(
        "pynvml not available — GPU monitoring disabled. "
        "Install with: pip install pynvml"
    )


class GPUMonitor:
    """
    Cached GPU utilization monitor using NVIDIA's pynvml library.
    
    Reads GPU utilization and memory via direct C bindings (~1ms per call),
    then caches the result for `cache_ttl` seconds to avoid hammering
    the driver under high request load.
    
    Usage:
        monitor = GPUMonitor(cache_ttl=5.0)
        util = monitor.get_utilization()   # 0.0 - 1.0
        mem = monitor.get_memory()         # {"used_mb": ..., "total_mb": ..., "percent": ...}
    """

    def __init__(self, cache_ttl: float = 5.0, gpu_index: int = 0):
        self._cache_ttl = cache_ttl
        self._gpu_index = gpu_index
        self._initialized = False
        self._handle = None

        # Cached values
        self._cached_utilization: float = 0.0
        self._cached_memory: dict = {"used_mb": 0, "total_mb": 0, "percent": 0.0}
        self._last_util_time: float = 0.0
        self._last_mem_time: float = 0.0

        self._initialize()

    def _initialize(self) -> None:
        """Initialize pynvml and get a handle to the GPU device."""
        if not PYNVML_AVAILABLE:
            logger.warning("GPUMonitor: pynvml not available, returning dummy values")
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if self._gpu_index >= device_count:
                logger.error(
                    f"GPU index {self._gpu_index} out of range "
                    f"(found {device_count} GPUs)"
                )
                return

            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            name = pynvml.nvmlDeviceGetName(self._handle)
            # pynvml may return bytes or str depending on version
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            logger.info(f"GPUMonitor initialized: {name} (GPU {self._gpu_index})")
            self._initialized = True
        except pynvml.NVMLError as e:
            logger.error(f"Failed to initialize pynvml: {e}")

    def get_utilization(self) -> float:
        """
        Get GPU compute utilization as a float between 0.0 and 1.0.
        
        Returns a cached value if the reading is younger than cache_ttl seconds.
        Falls back to 0.0 if pynvml is unavailable.
        """
        if not self._initialized:
            return 0.0

        now = monotonic()
        if now - self._last_util_time < self._cache_ttl:
            return self._cached_utilization

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            self._cached_utilization = util.gpu / 100.0
            self._last_util_time = now
        except pynvml.NVMLError as e:
            logger.warning(f"Failed to read GPU utilization: {e}")
            # Return last known value rather than failing

        return self._cached_utilization

    def get_memory(self) -> dict:
        """
        Get GPU memory usage.
        
        Returns:
            dict with keys: used_mb, total_mb, percent
        """
        if not self._initialized:
            return self._cached_memory

        now = monotonic()
        if now - self._last_mem_time < self._cache_ttl:
            return self._cached_memory

        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._cached_memory = {
                "used_mb": round(mem_info.used / (1024 * 1024)),
                "total_mb": round(mem_info.total / (1024 * 1024)),
                "percent": round(mem_info.used / mem_info.total, 4),
            }
            self._last_mem_time = now
        except pynvml.NVMLError as e:
            logger.warning(f"Failed to read GPU memory: {e}")

        return self._cached_memory

    def shutdown(self) -> None:
        """Clean up pynvml resources."""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
                logger.info("GPUMonitor shutdown complete")
            except pynvml.NVMLError:
                pass
            self._initialized = False
