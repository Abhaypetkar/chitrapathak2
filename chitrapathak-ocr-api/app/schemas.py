"""
Pydantic schemas for Chitrapathak-2 OCR API.

Defines request/response models for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


class MetricsData(BaseModel):
    """Per-request performance metrics returned with OCR results."""

    device: str = Field(description="Compute device used (cpu/cuda)")
    queue_wait: float = Field(description="Time waiting in queue (seconds)")
    image_load: float = Field(description="Image loading time (seconds)")
    tokenization: float = Field(description="Tokenization time (seconds)")
    inference: float = Field(description="Model inference time (seconds)")
    decode: float = Field(description="Token decoding time (seconds)")
    total: float = Field(description="Total request processing time (seconds)")
    input_tokens: int = Field(description="Number of input tokens")
    output_tokens: int = Field(description="Number of output tokens generated")
    tokens_per_second: float = Field(description="Output tokens per second during inference")
    cpu: float = Field(description="CPU usage percentage at time of request")
    ram: float = Field(description="RAM usage percentage at time of request")
    gpu: Optional[float] = Field(default=None, description="GPU usage percentage (if available)")
    gpu_memory: Optional[float] = Field(default=None, description="GPU memory usage percentage (if available)")


class OCRResponse(BaseModel):
    """Response model for POST /ocr endpoint."""

    request_id: str = Field(description="Unique request identifier")
    status: str = Field(description="Request status (success/failed)")
    text: str = Field(default="", description="Extracted OCR text")
    metrics: Optional[MetricsData] = Field(default=None, description="Performance metrics")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class HealthResponse(BaseModel):
    """Response model for GET /health endpoint."""

    status: str = Field(description="Service health status")
    model_loaded: bool = Field(description="Whether the model is loaded and ready")
    device: str = Field(description="Active compute device")
    queue_size: int = Field(description="Current queue size")
    uptime: float = Field(description="Service uptime in seconds")
    model_name: str = Field(description="Loaded model identifier")


class SystemMetricsResponse(BaseModel):
    """Response model for GET /metrics endpoint."""

    total_requests: int = Field(description="Total requests received")
    success: int = Field(description="Successful requests count")
    failed: int = Field(description="Failed requests count")
    avg_inference_time: float = Field(description="Average inference time (seconds)")
    avg_total_time: float = Field(description="Average total processing time (seconds)")
    cpu: float = Field(description="Current CPU usage percentage")
    ram: float = Field(description="Current RAM usage percentage")
    gpu: Optional[float] = Field(default=None, description="Current GPU usage percentage")
    gpu_memory: Optional[float] = Field(default=None, description="Current GPU memory usage percentage")
    queue_size: int = Field(description="Current queue size")
    device: str = Field(description="Active compute device")
