"""
Post-structured-output parsing for the Review Analysis module.
"""
from app.logger import get_logger
from app.schemas.review_analysis import ReviewAnalysisResult

logger = get_logger(__name__)

def normalize_result(result: ReviewAnalysisResult) -> ReviewAnalysisResult:
    """
    Apply light, non-destructive normalization to a structured LLM result.
    """
    normalized_sentiment = result.sentiment.strip().lower()
    if normalized_sentiment != result.sentiment:
        logger.debug(
            "Normalized sentiment casing: %r -> %r",
            result.sentiment,
            normalized_sentiment,
        )
        result = result.model_copy(update={"sentiment": normalized_sentiment})

    normalized_reason = result.reason.strip()
    if normalized_reason != result.reason:
        result = result.model_copy(update={"reason": normalized_reason})

    return result