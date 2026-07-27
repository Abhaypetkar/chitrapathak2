"""
System metrics collection for Chitrapathak-2 OCR API.

Tracks request counters, timing averages, and system resource usage
(CPU, RAM, GPU) using psutil and pynvml.
"""

import threading
from typing import Dict, Optional

import psutil

# Optional GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


class MetricsCollector:
    """
    Thread-safe metrics collector for tracking request statistics
    and system resource usage.
    """

    def __init__(self) -> None:
        """Initialize counters and lock."""
        self._lock = threading.Lock()
        self._total_requests: int = 0
        self._success: int = 0
        self._failed: int = 0
        self._total_inference_time: float = 0.0
        self._total_request_time: float = 0.0

    def record_success(self, inference_time: float, total_time: float) -> None:
        """
        Record a successful request.

        Args:
            inference_time: Model inference duration in seconds.
            total_time: Total request duration in seconds.
        """
        with self._lock:
            self._total_requests += 1
            self._success += 1
            self._total_inference_time += inference_time
            self._total_request_time += total_time

    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self._total_requests += 1
            self._failed += 1

    def get_stats(self) -> Dict[str, float]:
        """
        Get current request statistics.

        Returns:
            Dictionary with total, success, failed counts and averages.
        """
        with self._lock:
            total = self._total_requests
            success = self._success
            failed = self._failed
            avg_inference = (
                round(self._total_inference_time / success, 4)
                if success > 0 else 0.0
            )
            avg_total = (
                round(self._total_request_time / success, 4)
                if success > 0 else 0.0
            )

        return {
            "total_requests": total,
            "success": success,
            "failed": failed,
            "avg_inference_time": avg_inference,
            "avg_total_time": avg_total,
        }


def get_cpu_usage() -> float:
    """
    Get current CPU usage percentage.

    Returns:
        CPU usage as a percentage (0-100).
    """
    return psutil.cpu_percent(interval=0.1)


def get_ram_usage() -> float:
    """
    Get current RAM usage percentage.

    Returns:
        RAM usage as a percentage (0-100).
    """
    return psutil.virtual_memory().percent


def get_gpu_usage() -> Optional[float]:
    """
    Get current GPU utilization percentage.

    Returns:
        GPU utilization percentage, or None if GPU is not available.
    """
    if not GPU_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(utilization.gpu)
    except Exception:
        return None


def get_gpu_memory() -> Optional[float]:
    """
    Get current GPU memory usage percentage.

    Returns:
        GPU memory usage percentage, or None if GPU is not available.
    """
    if not GPU_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return round((mem_info.used / mem_info.total) * 100, 1)
    except Exception:
        return None


def get_system_metrics() -> Dict[str, Optional[float]]:
    """
    Collect all system metrics in one call.

    Returns:
        Dictionary with cpu, ram, gpu, and gpu_memory percentages.
    """
    return {
        "cpu": get_cpu_usage(),
        "ram": get_ram_usage(),
        "gpu": get_gpu_usage(),
        "gpu_memory": get_gpu_memory(),
    }


# Singleton metrics collector
metrics_collector = MetricsCollector()
