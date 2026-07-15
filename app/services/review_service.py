"""
Service layer for Review Analysis: bridges the API layer and the
underlying inference pipeline, converting pipeline output into the
response schema.
"""
import time

from app.logger import get_logger
from app.modules.review_analysis.inference import run_review_analysis
from app.schemas.review_analysis import ReviewRequest, ReviewResponse

logger = get_logger(__name__)


class ReviewService:
    """Orchestrates review analysis requests against the inference pipeline."""
    def analyze(self, request: ReviewRequest) -> ReviewResponse:
        """
        Analyze a single customer review and return the wrapped response.
        """
        logger.info("Review analysis requested (length=%d chars)", len(request.review))

        start_time = time.perf_counter()
        result = run_review_analysis(request.review)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info("Review analysis service call completed in %.1f ms", elapsed_ms)

        return ReviewResponse(result=result)