"""
Queue manager for Chitrapathak-2 OCR API.

Manages a ThreadPoolExecutor for concurrent request handling
with a GPU lock to serialize model.generate() calls.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import torch

from app.config import settings
from app.logger import get_logger

logger = get_logger()


class QueueManager:
    """
    Manages concurrent OCR request processing.

    Uses a ThreadPoolExecutor for parallelism and a threading.Lock
    to serialize GPU inference calls (preventing CUDA OOM errors).
    """

    def __init__(self, max_workers: int = None) -> None:
        """
        Initialize the queue manager.

        Args:
            max_workers: Maximum number of concurrent workers.
                         Defaults to settings.MAX_WORKERS.
        """
        self._max_workers = max_workers or settings.MAX_WORKERS
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="ocr-worker",
        )
        self._gpu_lock = threading.Lock()
        self._queue_size: int = 0
        self._queue_lock = threading.Lock()

        logger.info(f"Queue manager initialized with {self._max_workers} workers")

    @property
    def queue_size(self) -> int:
        """Get the current number of requests in the queue."""
        with self._queue_lock:
            return self._queue_size

    @property
    def gpu_lock(self) -> threading.Lock:
        """Get the GPU inference lock."""
        return self._gpu_lock

    def _increment_queue(self) -> None:
        """Increment the queue counter."""
        with self._queue_lock:
            self._queue_size += 1

    def _decrement_queue(self) -> None:
        """Decrement the queue counter."""
        with self._queue_lock:
            self._queue_size = max(0, self._queue_size - 1)

    async def submit(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Submit a task to the executor and await its result.

        Wraps the synchronous function in asyncio.run_in_executor()
        for non-blocking execution in the FastAPI event loop.

        Args:
            func: Synchronous callable to execute.
            *args: Positional arguments for the callable.
            **kwargs: Keyword arguments for the callable.

        Returns:
            The result of the callable.
        """
        self._increment_queue()
        try:
            loop = asyncio.get_event_loop()
            # Wrap to pass kwargs
            result = await loop.run_in_executor(
                self._executor,
                lambda: func(*args, **kwargs),
            )
            return result
        finally:
            self._decrement_queue()

    def shutdown(self) -> None:
        """Shutdown the executor, waiting for pending tasks."""
        logger.info("Shutting down queue manager...")
        self._executor.shutdown(wait=True)
        logger.info("Queue manager shut down")


# Singleton queue manager instance
queue_manager = QueueManager()
