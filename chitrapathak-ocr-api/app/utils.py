"""
Utility functions for Chitrapathak-2 OCR API.

Provides request ID generation, file validation, and timestamp helpers.
"""

import uuid
import os
import time
from datetime import datetime, timezone
from typing import Optional


# Supported image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def generate_request_id() -> str:
    """
    Generate a unique 8-character hex request ID.

    Returns:
        8-character hexadecimal string.
    """
    return uuid.uuid4().hex[:8]


def validate_image_file(filename: str) -> bool:
    """
    Validate that the uploaded file has a supported image extension.

    Args:
        filename: Original filename from the upload.

    Returns:
        True if the file extension is supported, False otherwise.
    """
    if not filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def get_file_extension(filename: str) -> str:
    """
    Extract the file extension from a filename.

    Args:
        filename: Original filename.

    Returns:
        Lowercase file extension including the dot (e.g., '.jpg').
    """
    return os.path.splitext(filename)[1].lower()


def get_timestamp() -> str:
    """
    Get current UTC timestamp in ISO format.

    Returns:
        ISO-formatted UTC timestamp string.
    """
    return datetime.now(timezone.utc).isoformat()


def get_uptime(start_time: float) -> float:
    """
    Calculate application uptime in seconds.

    Args:
        start_time: Application start time from time.time().

    Returns:
        Uptime in seconds, rounded to 2 decimal places.
    """
    return round(time.time() - start_time, 2)


def safe_filename(request_id: str, original_filename: str) -> str:
    """
    Generate a safe filename using the request ID and original extension.

    Args:
        request_id: Unique request identifier.
        original_filename: Original uploaded filename.

    Returns:
        Safe filename in format '{request_id}{extension}'.
    """
    ext = get_file_extension(original_filename)
    return f"{request_id}{ext}"


def ensure_directories(*dirs: str) -> None:
    """
    Create directories if they don't exist.

    Args:
        *dirs: Variable number of directory paths to create.
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
