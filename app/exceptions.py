"""
Custom exception hierarchy for Servixa AI.

Usage:
    from app.exceptions import ModelLoadingException
    raise ModelLoadingException(
        "Failed to load pricing model",
        details={"path": str(model_path)},
    )
"""

from typing import Any, Dict, Optional

class ServixaException(Exception):
    """
    Base exception for all Servixa AI application errors.
    """
    status_code: int = 500

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | details={self.details}"
        return self.message


class ConfigurationException(ServixaException):
    """Raised when required configuration is missing or invalid."""
    status_code = 500

class ModelLoadingException(ServixaException):
    """Raised when an ML/DL model (pricing, YOLO, embeddings) fails to load."""
    status_code = 500

class PredictionException(ServixaException):
    """Raised when a model inference/prediction step fails."""
    status_code = 500

class InvalidImageException(ServixaException):
    """Raised when an input image is missing, corrupted, or unsupported."""
    status_code = 400

class OCRException(ServixaException):
    """Raised when OCR text extraction fails or produces unusable output."""
    status_code = 422

class ValidationException(ServixaException):
    """Raised when input or parsed data fails schema/business validation."""
    status_code = 422

class LLMException(ServixaException):
    """Raised when a call to an LLM provider (e.g. Gemini) fails."""
    status_code = 502

class VectorDatabaseException(ServixaException):
    """Raised when vector store operations (index, query, persist) fail."""
    status_code = 500