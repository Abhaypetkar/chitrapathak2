"""
Logging module for Chitrapathak-2 OCR API.

Provides JSON-formatted logging with both console output
and daily rotating file handler.
"""

import logging
import json
import os
import sys
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from typing import Any, Dict, Optional

from app.config import settings


class JSONFormatter(logging.Formatter):
    """Custom JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        extra_fields = [
            "request_id", "filename", "device", "input_tokens",
            "output_tokens", "queue_wait", "inference_time",
            "total_time", "status", "error",
        ]
        for field in extra_fields:
            value = getattr(record, field, None)
            if value is not None:
                log_entry[field] = value

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logger(name: str = "chitrapathak-ocr") -> logging.Logger:
    """
    Configure and return the application logger.

    Sets up:
    - Console handler (stdout) with JSON formatting
    - Daily rotating file handler with JSON formatting

    Args:
        name: Logger name identifier.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers on re-initialization
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    json_formatter = JSONFormatter()

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(json_formatter)
    logger.addHandler(console_handler)

    # --- File Handler (Daily Rotation) ---
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    log_file = os.path.join(settings.LOG_DIR, "ocr_api.log")

    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Get or create the application logger."""
    return setup_logger()


def log_request(
    logger: logging.Logger,
    request_id: str,
    filename: str,
    device: str,
    input_tokens: int,
    output_tokens: int,
    queue_wait: float,
    inference_time: float,
    total_time: float,
    status: str,
    error: Optional[str] = None,
) -> None:
    """
    Log a structured OCR request entry.

    Args:
        logger: Logger instance.
        request_id: Unique request identifier.
        filename: Uploaded file name.
        device: Compute device used (cpu/cuda).
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        queue_wait: Time spent waiting in queue (seconds).
        inference_time: Model inference time (seconds).
        total_time: Total request processing time (seconds).
        status: Request status (success/failed).
        error: Error message if status is failed.
    """
    extra = {
        "request_id": request_id,
        "filename": filename,
        "device": device,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "queue_wait": round(queue_wait, 4),
        "inference_time": round(inference_time, 4),
        "total_time": round(total_time, 4),
        "status": status,
    }
    if error:
        extra["error"] = error

    if status == "success":
        logger.info(
            f"OCR completed: {request_id} | {filename} | "
            f"inference={inference_time:.2f}s | total={total_time:.2f}s",
            extra=extra,
        )
    else:
        logger.error(
            f"OCR failed: {request_id} | {filename} | error={error}",
            extra=extra,
        )
