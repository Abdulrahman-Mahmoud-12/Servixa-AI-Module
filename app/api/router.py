"""
Central API router for Servixa AI.
"""
from fastapi import APIRouter

from app.config import settings
from app.utils.constants import HEALTH_TAG
from app.utils.response import success_response
from app.api import id_verification, review_analysis

api_router = APIRouter(prefix="/router")

api_router.include_router(id_verification.router)
api_router.include_router(review_analysis.router)

@api_router.get("/health", tags=[HEALTH_TAG], summary="Service health check")
async def health_check():
    """
    Lightweight liveness/readiness check for the API layer.
    """
    return success_response(
        data={
            "status": "healthy",
            "service": settings.app.name,
            "version": settings.app.version,
        },
        message="Service is healthy.",
    )


# ---------------------------------------------------------------------------
# Future feature routers.
#
# Each module below will expose its own `router: APIRouter` (typically
# in `app/api/<module>/router.py` or similar) and be wired in here as
# it is implemented. Left commented out intentionally -- do not
# implement these routers yet.
#
#     from app.api.pricing.router import router as pricing_router
#     api_router.include_router(pricing_router, prefix="/pricing", tags=[PRICING_TAG])
#
#     from app.api.id_verification.router import router as id_verification_router
#     api_router.include_router(
#         id_verification_router, prefix="/id-verification", tags=[ID_VERIFICATION_TAG]
#     )
#
#     from app.api.review_analysis.router import router as review_analysis_router
#     api_router.include_router(
#         review_analysis_router, prefix="/reviews", tags=[REVIEW_ANALYSIS_TAG]
#     )
#
#     from app.api.chatbot.router import router as chatbot_router
#     api_router.include_router(chatbot_router, prefix="/chatbot", tags=[CHATBOT_TAG])
# ---------------------------------------------------------------------------