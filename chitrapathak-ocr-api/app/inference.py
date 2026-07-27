"""
OCR inference pipeline for Chitrapathak-2 OCR API.

Performs a single-pass OCR on the original color image using
the Chitrapathak-2 VLM (Qwen2.5-VL architecture).

No preprocessing. No grayscale. No CLAHE. No sharpening. No multi-pass.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import torch
from PIL import Image

from app.config import settings
from app.logger import get_logger
from app.model import model_manager
from app.queue_manager import queue_manager
from app.metrics import get_system_metrics

logger = get_logger()

# The OCR prompt — used for every request
OCR_PROMPT = """Read this historical Sanskrit manuscript carefully.

Return ONLY the exact Devanagari text.

Rules:
- Preserve line breaks.
- Do not translate.
- Do not explain.
- Do not summarize.
- Do not correct spelling.
- Ignore borders and illustrations.
- Output only the manuscript text."""


@dataclass
class OCRResult:
    """Container for OCR inference results and timing metrics."""

    text: str = ""
    device: str = "cpu"
    queue_wait: float = 0.0
    image_load: float = 0.0
    tokenization: float = 0.0
    inference: float = 0.0
    decode: float = 0.0
    total: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    cpu: float = 0.0
    ram: float = 0.0
    gpu: Optional[float] = None
    gpu_memory: Optional[float] = None


def run_ocr(
    image_path: str,
    max_new_tokens: int = None,
    language: str = None,
    queue_wait_time: float = 0.0,
) -> OCRResult:
    """
    Execute single-pass OCR on a color image using Chitrapathak-2.

    Pipeline:
    1. Load original color image (no preprocessing)
    2. Build chat messages with OCR prompt
    3. processor.apply_chat_template() → text
    4. process_vision_info() → image tensors
    5. processor(text, images) → model inputs
    6. model.generate() → output token IDs
    7. Trim input tokens, decode generated tokens only
    8. Return OCR text + all timing metrics

    Args:
        image_path: Path to the uploaded image file.
        max_new_tokens: Maximum tokens to generate (default from config).
        language: Language hint (currently unused, reserved for future).
        queue_wait_time: Time the request spent waiting in queue.

    Returns:
        OCRResult with extracted text and performance metrics.
    """
    # Import qwen_vl_utils here to avoid import issues
    from qwen_vl_utils import process_vision_info

    total_start = time.perf_counter()

    max_tokens = max_new_tokens or settings.MAX_NEW_TOKENS
    device = model_manager.device
    model = model_manager.get_model()
    processor = model_manager.get_processor()

    result = OCRResult(device=device, queue_wait=queue_wait_time)

    # --- Step 1: Load Image (original color, no preprocessing) ---
    t0 = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    result.image_load = round(time.perf_counter() - t0, 4)

    # --- Step 2: Build chat messages ---
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": OCR_PROMPT},
            ],
        }
    ]

    # --- Step 3: Apply chat template ---
    t0 = time.perf_counter()
    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # --- Step 4: Process vision info ---
    image_inputs, video_inputs = process_vision_info(messages)

    # --- Step 5: Tokenize (processor call) ---
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move inputs to device
    inputs = inputs.to(model_manager.device)
    input_ids_length = inputs.input_ids.shape[1]
    result.input_tokens = input_ids_length
    result.tokenization = round(time.perf_counter() - t0, 4)

    # --- Step 6: Generate (with GPU lock if applicable) ---
    t0 = time.perf_counter()

    # Acquire GPU lock to prevent concurrent CUDA operations
    with queue_manager.gpu_lock:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
            )

    result.inference = round(time.perf_counter() - t0, 4)

    # --- Step 7: Decode only generated tokens ---
    t0 = time.perf_counter()

    # Trim input tokens from output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    result.decode = round(time.perf_counter() - t0, 4)

    # --- Step 8: Assemble result ---
    result.text = output_text[0] if output_text else ""
    result.output_tokens = len(generated_ids_trimmed[0]) if generated_ids_trimmed else 0

    # Token throughput
    if result.inference > 0 and result.output_tokens > 0:
        result.tokens_per_second = round(
            result.output_tokens / result.inference, 2
        )

    # Total time
    result.total = round(time.perf_counter() - total_start, 4)

    # System metrics snapshot
    sys_metrics = get_system_metrics()
    result.cpu = sys_metrics["cpu"]
    result.ram = sys_metrics["ram"]
    result.gpu = sys_metrics["gpu"]
    result.gpu_memory = sys_metrics["gpu_memory"]

    return result
