"""
Model loader for Chitrapathak-2 OCR API.

Implements singleton pattern to load the Chitrapathak-2 VLM
(Qwen2.5-VL architecture) once at application startup.
"""

import torch
from typing import Optional, Tuple

from app.config import settings
from app.logger import get_logger

logger = get_logger()


class ModelManager:
    """
    Singleton model manager for Chitrapathak-2 VLM.

    Loads the model and processor once, auto-detects CPU/GPU,
    and provides thread-safe access to both.
    """

    def __init__(self) -> None:
        """Initialize model manager with empty state."""
        self._model = None
        self._processor = None
        self._device: str = "cpu"
        self._loaded: bool = False

    @property
    def device(self) -> str:
        """Get the active compute device."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._loaded

    def _detect_device(self) -> str:
        """
        Auto-detect the best available compute device.

        Returns:
            'cuda' if NVIDIA GPU is available, otherwise 'cpu'.
        """
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            logger.info(
                f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)",
            )
        else:
            device = "cpu"
            logger.info("No GPU detected, using CPU")

        return device

    def load(self) -> None:
        """
        Load the Chitrapathak-2 model and processor.

        Uses float16 on GPU and float32 on CPU.
        The model is loaded once and reused for all requests.
        """
        if self._loaded:
            logger.warning("Model already loaded, skipping reload")
            return

        # Import here to avoid import errors if transformers not installed
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self._device = self._detect_device()

        logger.info(f"Loading model: {settings.MODEL_NAME}")
        logger.info(f"Target device: {self._device}")

        # Determine dtype based on device
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        try:
            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                settings.MODEL_NAME,
                trust_remote_code=True,
                token=settings.HF_TOKEN,
            )
            logger.info("Processor loaded successfully")

            # Load model
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                settings.MODEL_NAME,
                torch_dtype=dtype,
                device_map="auto" if self._device == "cuda" else None,
                trust_remote_code=True,
                token=settings.HF_TOKEN,
            )

            # Move to device if not using device_map="auto"
            if self._device == "cpu":
                self._model = self._model.to(self._device)

            self._model.eval()
            self._loaded = True

            logger.info(
                f"Model loaded successfully on {self._device} "
                f"with dtype={dtype}"
            )

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}") from e

    def get_model(self):
        """
        Get the loaded model instance.

        Returns:
            The loaded Qwen2.5-VL model.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    def get_processor(self):
        """
        Get the loaded processor instance.

        Returns:
            The loaded AutoProcessor.

        Raises:
            RuntimeError: If the processor has not been loaded.
        """
        if not self._loaded or self._processor is None:
            raise RuntimeError("Processor not loaded. Call load() first.")
        return self._processor

    def unload(self) -> None:
        """
        Unload the model and free memory.

        Clears CUDA cache if GPU was used.
        """
        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded and memory freed")


# Singleton model manager instance
model_manager = ModelManager()
