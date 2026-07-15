"""
Shared application resources for Servixa AI.
"""

from dataclasses import dataclass
from typing import Any, Optional
from fastapi import FastAPI, Request

from app.exceptions import ConfigurationException
from app.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ResourceRegistry:
    """
    Container for every shared, process-wide resource used by the app.
    """
    # Pricing module
    pricing_model: Optional[Any] = None
    pricing_pipeline: Optional[Any] = None

    # ID verification module
    yolo_detector: Optional[Any] = None

    # LLM module (Google Gemini)
    gemini_client: Optional[Any] = None

    # RAG / chatbot module
    embedding_model: Optional[Any] = None
    vector_db: Optional[Any] = None


async def init_resources(app: FastAPI) -> None:
    """
    Initialize all shared resources at application startup.
    """
    registry = ResourceRegistry()

    # --- Pricing model -------------------------------------------------
    # TODO: load `settings.pricing.model_path` / `pipeline_path`.
    logger.info("Pricing model: not yet implemented (placeholder).")

    # --- ID verification (YOLO) -----------------------------------------
    # TODO: load `settings.id_verification.yolo_model_path`.
    logger.info("YOLO ID detector: not yet implemented (placeholder).")

    # --- LLM client (Gemini) ---------------------------------------------
    # TODO: construct client using `settings.llm.gemini_api_key`.
    logger.info("Gemini client: not yet implemented (placeholder).")

    # --- RAG (embeddings + vector DB) ------------------------------------
    # TODO: load `settings.rag.embedding_model` and connect to
    # `settings.rag.vector_db_path`.
    logger.info("Embedding model / vector DB: not yet implemented (placeholder).")

    app.state.resources = registry
    logger.info("Resource registry initialized.")


async def shutdown_resources(app: FastAPI) -> None:
    """
    Release/close all shared resources at application shutdown.
    """
    registry: Optional[ResourceRegistry] = getattr(app.state, "resources", None)
    if registry is None:
        logger.info("No resource registry to shut down.")
        return

    # TODO: add explicit teardown per resource as they are implemented,
    # e.g. `registry.vector_db.close()`.
    logger.info("Resource registry shut down.")


def _get_registry(request: Request) -> ResourceRegistry:
    """Fetch the process-wide resource registry from application state."""
    registry: Optional[ResourceRegistry] = getattr(
        request.app.state, "resources", None
    )
    if registry is None:
        raise ConfigurationException(
            "Resource registry has not been initialized.",
            details={"hint": "Is the application lifespan running?"},
        )
    return registry


# --- FastAPI dependency accessors -------------------------------------------
#
# Future routers should depend on these functions (e.g.
# `model=Depends(get_pricing_model)`) rather than reading
# `request.app.state` directly. Each raises a clear, typed exception
# until the corresponding resource is actually loaded in
# `init_resources`.


def get_pricing_model(request: Request) -> Any:
    """Return the loaded pricing model, once implemented."""
    registry = _get_registry(request)
    if registry.pricing_model is None:
        raise ConfigurationException("Pricing model is not loaded yet.")
    return registry.pricing_model


def get_pricing_pipeline(request: Request) -> Any:
    """Return the loaded pricing preprocessing pipeline, once implemented."""
    registry = _get_registry(request)
    if registry.pricing_pipeline is None:
        raise ConfigurationException("Pricing pipeline is not loaded yet.")
    return registry.pricing_pipeline


def get_yolo_detector(request: Request) -> Any:
    """Return the loaded YOLO ID-card detector, once implemented."""
    registry = _get_registry(request)
    if registry.yolo_detector is None:
        raise ConfigurationException("YOLO detector is not loaded yet.")
    return registry.yolo_detector


def get_gemini_client(request: Request) -> Any:
    """Return the configured Gemini client, once implemented."""
    registry = _get_registry(request)
    if registry.gemini_client is None:
        raise ConfigurationException("Gemini client is not configured yet.")
    return registry.gemini_client


def get_embedding_model(request: Request) -> Any:
    """Return the loaded embedding model, once implemented."""
    registry = _get_registry(request)
    if registry.embedding_model is None:
        raise ConfigurationException("Embedding model is not loaded yet.")
    return registry.embedding_model


def get_vector_db(request: Request) -> Any:
    """Return the connected vector database handle, once implemented."""
    registry = _get_registry(request)
    if registry.vector_db is None:
        raise ConfigurationException("Vector database is not connected yet.")
    return registry.vector_db