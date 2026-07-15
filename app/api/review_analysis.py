"""
API routes for Review Analysis. Accepts a customer review and returns
a structured authenticity + sentiment analysis.
"""
from fastapi import APIRouter, HTTPException

from app.exceptions import ServixaException
from app.logger import get_logger
from app.schemas.review_analysis import ReviewRequest, ReviewResponse
from app.services.review_service import ReviewService
from app.utils.constants import REVIEW_ANALYSIS_TAG

logger = get_logger(__name__)
router = APIRouter(prefix="/review-analysis", tags=[REVIEW_ANALYSIS_TAG])

# Single service instance reused across requests.
_service = ReviewService()

@router.post("/analyze", response_model=ReviewResponse)
async def analyze_review(request: ReviewRequest) -> ReviewResponse:
    """
    Analyze a single customer review for authenticity and sentiment.
    """
    try:
        return _service.analyze(request)
    except ServixaException as exc:
        logger.warning("Review analysis failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc