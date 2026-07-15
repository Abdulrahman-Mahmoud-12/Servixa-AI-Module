"""
Validation logic for the Review Analysis module.
"""
from app.exceptions import ValidationException
from app.logger import get_logger
from app.schemas.review_analysis import ReviewAnalysisResult

logger = get_logger(__name__)

MIN_SENTIMENT_STRENGTH = 0.0
MAX_SENTIMENT_STRENGTH = 5.0
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
_VALID_SENTIMENTS = {"positive", "negative"}


def validate_review_text(review: str) -> str:
    """
    Validate raw review text before it reaches the LLM.
    """
    if review is None:
        raise ValidationException("Review text is missing.")

    stripped = review.strip()
    if not stripped:
        raise ValidationException("Review text must not be empty.")

    return stripped


def validate_analysis_result(result: ReviewAnalysisResult) -> ReviewAnalysisResult:
    """
    Defense-in-depth validation of the LLM's structured output.
    """
    if result.sentiment not in _VALID_SENTIMENTS:
        raise ValidationException(
            "Invalid sentiment value returned by the model.",
            details={"sentiment": result.sentiment},
        )

    if not (MIN_SENTIMENT_STRENGTH <= result.sentiment_strength <= MAX_SENTIMENT_STRENGTH):
        raise ValidationException(
            "sentiment_strength out of allowed range [0.0, 5.0].",
            details={"sentiment_strength": result.sentiment_strength},
        )

    if not (MIN_CONFIDENCE <= result.fake_confidence <= MAX_CONFIDENCE):
        raise ValidationException(
            "fake_confidence out of allowed range [0.0, 1.0].",
            details={"fake_confidence": result.fake_confidence},
        )

    if not result.reason or not result.reason.strip():
        raise ValidationException("Model returned an empty reason.")

    return result