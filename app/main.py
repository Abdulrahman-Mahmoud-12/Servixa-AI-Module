"""
Servixa AI - FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI

from app.api.router import api_router
from app.config import settings
from app.dependencies import init_resources, shutdown_resources
from app.logger import configure_logging, get_logger
from app.utils.constants import API_PREFIX, API_VERSION
from app.utils.response import success_response

# Configure logging as early as possible, before anything else initializes.
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application startup and shutdown events.\
    """
    # --- Startup ---------------------------------------------------
    logger.info(
        "Starting %s v%s (debug=%s)",
        settings.app.name,
        settings.app.version,
        settings.app.debug,
    )

    await init_resources(app)

    logger.info("Servixa AI startup complete.")

    yield
    # --- Shutdown --------------------------------------------------
    await shutdown_resources(app)
    logger.info("Shutting down %s.", settings.app.name)


app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    description=(
        "Servixa AI is an AI microservice providing pricing prediction, "
        "ID verification, review analysis, and a retrieval-augmented "
        "chatbot, designed to be consumed by a separate backend project."
    ),
    debug=settings.app.debug,
    lifespan=lifespan,
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
    openapi_url=f"{API_PREFIX}/openapi.json",
)

# Register the central API router (health check now; pricing, ID
# verification, review analysis, and chatbot routers will be added
# inside app/api/router.py as those modules are implemented).
app.include_router(api_router, prefix=API_PREFIX)


@app.get("/", tags=["Root"], summary="Basic API information")
async def root():
    """
    Root endpoint returning basic, non-sensitive service information.
    """
    return success_response(
        data={
            "service": settings.app.name,
            "version": settings.app.version,
            "api_version": API_VERSION,
            "docs": f"{API_PREFIX}/docs",
        },
        message=f"Welcome to {settings.app.name}.",
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=False,
        reload_dirs=["app"],
    )