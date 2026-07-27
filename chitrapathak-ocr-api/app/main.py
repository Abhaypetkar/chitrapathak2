"""
FastAPI application for Chitrapathak-2 OCR API.

Production-ready OCR microservice for historical Sanskrit/Devanagari
manuscript transcription using the Chitrapathak-2 VLM.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.inference import run_ocr
from app.logger import get_logger, log_request
from app.metrics import get_system_metrics, metrics_collector
from app.model import model_manager
from app.queue_manager import queue_manager
from app.schemas import HealthResponse, MetricsData, OCRResponse, SystemMetricsResponse
from app.utils import (
    ensure_directories,
    generate_request_id,
    get_uptime,
    safe_filename,
    validate_image_file,
)

logger = get_logger()

# Application start time for uptime tracking
APP_START_TIME: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Startup: Create directories, load model.
    Shutdown: Unload model, shutdown queue.
    """
    global APP_START_TIME
    APP_START_TIME = time.time()

    logger.info("=" * 60)
    logger.info("Chitrapathak-2 OCR API starting up...")
    logger.info("=" * 60)

    # Create required directories
    ensure_directories(
        settings.UPLOAD_DIR,
        settings.OUTPUT_DIR,
        settings.LOG_DIR,
    )

    # Load model at startup (singleton)
    try:
        model_manager.load()
        logger.info("Model loaded and ready for inference")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        raise

    logger.info(f"Server ready on {settings.HOST}:{settings.PORT}")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutting down Chitrapathak-2 OCR API...")
    queue_manager.shutdown()
    model_manager.unload()
    logger.info("Shutdown complete")


# --- FastAPI App ---
app = FastAPI(
    title="Chitrapathak-2 OCR API",
    description=(
        "Production-ready OCR microservice using Chitrapathak-2 "
        "Vision-Language Model for historical Sanskrit/Devanagari "
        "manuscript transcription."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# POST /ocr
# ──────────────────────────────────────────────

@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(
    file: UploadFile = File(..., description="Image file to transcribe"),
    language: Optional[str] = Form(
        default=None,
        description="Language hint (default: sanskrit)",
    ),
    max_new_tokens: Optional[int] = Form(
        default=None,
        description="Maximum new tokens to generate",
    ),
) -> OCRResponse:
    """
    Perform OCR on an uploaded manuscript image.

    Accepts a multipart image upload and returns the extracted
    Devanagari text along with detailed performance metrics.
    """
    request_id = generate_request_id()
    request_start = time.perf_counter()
    queue_enter = time.perf_counter()

    logger.info(
        f"Received OCR request: {request_id} | file={file.filename}",
        extra={"request_id": request_id, "filename": file.filename},
    )

    # Validate file
    if not validate_image_file(file.filename or ""):
        raise HTTPException(
            status_code=400,
            detail={
                "request_id": request_id,
                "error": (
                    "Unsupported file type. Supported: "
                    "jpg, jpeg, png, bmp, tiff, tif, webp"
                ),
            },
        )

    # Save uploaded file
    filename = safe_filename(request_id, file.filename or "image.jpg")
    filepath = os.path.join(settings.UPLOAD_DIR, filename)

    try:
        contents = await file.read()
        with open(filepath, "wb") as f:
            f.write(contents)
    except Exception as e:
        logger.error(f"Failed to save file: {request_id} | {e}")
        metrics_collector.record_failure()
        raise HTTPException(
            status_code=500,
            detail={"request_id": request_id, "error": f"File save failed: {str(e)}"},
        )

    # Run OCR via queue manager
    try:
        queue_wait = time.perf_counter() - queue_enter

        ocr_result = await queue_manager.submit(
            run_ocr,
            image_path=filepath,
            max_new_tokens=max_new_tokens,
            language=language,
            queue_wait_time=queue_wait,
        )

        # Save output text
        output_path = os.path.join(settings.OUTPUT_DIR, f"{request_id}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ocr_result.text)

        total_time = round(time.perf_counter() - request_start, 4)

        # Record metrics
        metrics_collector.record_success(
            inference_time=ocr_result.inference,
            total_time=total_time,
        )

        # Log request
        log_request(
            logger=logger,
            request_id=request_id,
            filename=file.filename or "unknown",
            device=ocr_result.device,
            input_tokens=ocr_result.input_tokens,
            output_tokens=ocr_result.output_tokens,
            queue_wait=ocr_result.queue_wait,
            inference_time=ocr_result.inference,
            total_time=total_time,
            status="success",
        )

        return OCRResponse(
            request_id=request_id,
            status="success",
            text=ocr_result.text,
            metrics=MetricsData(
                device=ocr_result.device,
                queue_wait=ocr_result.queue_wait,
                image_load=ocr_result.image_load,
                tokenization=ocr_result.tokenization,
                inference=ocr_result.inference,
                decode=ocr_result.decode,
                total=total_time,
                input_tokens=ocr_result.input_tokens,
                output_tokens=ocr_result.output_tokens,
                tokens_per_second=ocr_result.tokens_per_second,
                cpu=ocr_result.cpu,
                ram=ocr_result.ram,
                gpu=ocr_result.gpu,
                gpu_memory=ocr_result.gpu_memory,
            ),
        )

    except Exception as e:
        total_time = round(time.perf_counter() - request_start, 4)
        metrics_collector.record_failure()

        log_request(
            logger=logger,
            request_id=request_id,
            filename=file.filename or "unknown",
            device=model_manager.device,
            input_tokens=0,
            output_tokens=0,
            queue_wait=0.0,
            inference_time=0.0,
            total_time=total_time,
            status="failed",
            error=str(e),
        )

        logger.error(f"OCR failed: {request_id} | {str(e)}", exc_info=True)

        return OCRResponse(
            request_id=request_id,
            status="failed",
            text="",
            error=str(e),
        )


# ──────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """
    Health check endpoint.

    Returns model status, device info, queue size, and uptime.
    """
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "unhealthy",
        model_loaded=model_manager.is_loaded,
        device=model_manager.device,
        queue_size=queue_manager.queue_size,
        uptime=get_uptime(APP_START_TIME),
        model_name=settings.MODEL_NAME,
    )


# ──────────────────────────────────────────────
# GET /metrics
# ──────────────────────────────────────────────

@app.get("/metrics", response_model=SystemMetricsResponse)
async def metrics_endpoint() -> SystemMetricsResponse:
    """
    System and request metrics endpoint.

    Returns aggregate request statistics and current system resource usage.
    """
    stats = metrics_collector.get_stats()
    sys_metrics = get_system_metrics()

    return SystemMetricsResponse(
        total_requests=stats["total_requests"],
        success=stats["success"],
        failed=stats["failed"],
        avg_inference_time=stats["avg_inference_time"],
        avg_total_time=stats["avg_total_time"],
        cpu=sys_metrics["cpu"],
        ram=sys_metrics["ram"],
        gpu=sys_metrics["gpu"],
        gpu_memory=sys_metrics["gpu_memory"],
        queue_size=queue_manager.queue_size,
        device=model_manager.device,
    )
