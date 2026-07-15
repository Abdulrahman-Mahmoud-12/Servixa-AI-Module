"""
End-to-end Review Analysis pipeline: validate input, run the LangChain
Groq pipeline, and validate the structured output before returning it.
"""
import time

from app.exceptions import LLMException
from app.logger import get_logger
from app.modules.review_analysis.llm_chain import get_review_analysis_chain
from app.modules.review_analysis.parser import normalize_result
from app.modules.review_analysis.validator import (
    validate_analysis_result,
    validate_review_text,
)
from app.schemas.review_analysis import ReviewAnalysisResult

logger = get_logger(__name__)


def run_review_analysis(review: str) -> ReviewAnalysisResult:
    """
    Run the complete review analysis pipeline on a single review.

    Flow:
        Receive review -> validate input -> prompt -> LangChain
        -> structured output -> validate output -> return result.
    """
    validated_review = validate_review_text(review)

    chain = get_review_analysis_chain()

    start_time = time.perf_counter()
    try:
        result = chain.invoke({"review": validated_review})
    except Exception as exc:
        logger.error("Groq LLM call failed during review analysis: %s", exc)
        raise LLMException(
            "Failed to analyze review with the LLM provider.",
            details={"error": str(exc)},
        ) from exc
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    if not isinstance(result, ReviewAnalysisResult):
        raise LLMException(
            "LLM did not return the expected structured output type.",
            details={"returned_type": type(result).__name__},
        )

    result = normalize_result(result)
    result = validate_analysis_result(result)

    logger.info(
        "Review analysis complete in %.1f ms (is_fake=%s, sentiment=%s, strength=%.2f)",
        elapsed_ms,
        result.is_fake,
        result.sentiment,
        result.sentiment_strength,
    )
    return result