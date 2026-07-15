"""
Pydantic schemas for the Review Analysis module: request/response
contracts for the `/review-analysis` endpoints, and the structured
output contract returned directly by the LLM via LangChain.
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator


class ReviewRequest(BaseModel):
    """Incoming request payload: a single customer review to analyze."""
    review: str = Field(
        min_length=1,
        description="Raw customer review text to analyze.")

    @field_validator("review")
    @classmethod
    def review_must_not_be_blank(cls, value: str) -> str:
        """Reject reviews that are empty or whitespace-only after stripping."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Review text must not be empty or whitespace-only.")
        return stripped


class ReviewAnalysisResult(BaseModel):
    """
    Structured output returned directly by the LLM via LangChain's
    structured output feature. Every field is filled in by the model;
    no manual JSON parsing is performed.
    """
    is_fake: bool = Field(
        description="Determine if the review is Real 'False' (written by a genuine customer) or Fake 'True' (fabricated, spam, or bot-generated).",)
    
    fake_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Model confidence in the fake/real verdict, from 0.0 "
            "(no confidence) to 1.0 (certain)."))
    
    sentiment: Literal["positive", "negative"] = Field(
        description="Overall sentiment expressed in the review.")
    
    sentiment_strength: float = Field(
        ge=0.0,
        le=5.0,
        description=(
            "Strength of the sentiment, from 0.0 (extremely weak) "
            "to 5.0 (extremely strong)."))
    
    reason: str = Field(
        min_length=1,
        description=(
            "Short explanation grounding the is_fake and sentiment "
            "decisions in the review text."))


class ReviewResponse(BaseModel):
    """Response payload wrapping the complete review analysis result."""
    result: ReviewAnalysisResult
    message: str = "Review analysis completed"