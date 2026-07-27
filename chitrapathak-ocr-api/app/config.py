"""
Configuration module for Chitrapathak-2 OCR API.

Loads settings from environment variables / .env file using Pydantic BaseSettings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # --- Model ---
    MODEL_NAME: str = Field(
        default="krutrim-ai-labs/Chitrapathak-2",
        description="HuggingFace model identifier for Chitrapathak-2 VLM",
    )
    MAX_NEW_TOKENS: int = Field(
        default=2048,
        description="Maximum number of new tokens to generate per inference",
    )
    DEFAULT_LANGUAGE: str = Field(
        default="sanskrit",
        description="Default language hint for OCR",
    )

    # --- Concurrency ---
    MAX_WORKERS: int = Field(
        default=2,
        description="Number of ThreadPoolExecutor workers",
    )

    # --- Directories ---
    UPLOAD_DIR: str = Field(
        default="uploads",
        description="Directory to save uploaded images",
    )
    OUTPUT_DIR: str = Field(
        default="outputs",
        description="Directory to save OCR result text files",
    )
    LOG_DIR: str = Field(
        default="logs",
        description="Directory for rotating log files",
    )

    # --- Logging ---
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # --- Server ---
    HOST: str = Field(
        default="0.0.0.0",
        description="Server bind host",
    )
    PORT: int = Field(
        default=8000,
        description="Server bind port",
    )

    # --- HuggingFace ---
    HF_TOKEN: Optional[str] = Field(
        default=None,
        description="HuggingFace API token (if model requires authentication)",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


# Singleton settings instance
settings = Settings()
