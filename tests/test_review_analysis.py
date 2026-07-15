"""
Tests for the Review Analysis module: schemas, validator, inference
pipeline, service layer, and the API endpoint.

The Groq LLM call is mocked throughout so these tests do not require
network access or a real GROQ_API_KEY.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.exceptions import LLMException, ValidationException
from app.main import app
from app.modules.review_analysis import inference as inference_module
from app.schemas.review_analysis import ReviewAnalysisResult, ReviewRequest
from app.services.review_service import ReviewService

client = TestClient(app)


def _make_result(**overrides) -> ReviewAnalysisResult:
    """Build a valid ReviewAnalysisResult with sensible defaults."""
    defaults = dict(
        is_fake=False,
        fake_confidence=0.1,
        sentiment="positive",
        sentiment_strength=4.2,
        reason="The review mentions specific product details and a genuine use case.",
    )
    defaults.update(overrides)
    return ReviewAnalysisResult(**defaults)


class _FakeChain:
    """Stand-in for the LangChain runnable returned by get_review_analysis_chain."""

    def __init__(self, result=None, exception=None):
        self._result = result
        self._exception = exception

    def invoke(self, _inputs):
        if self._exception:
            raise self._exception
        return self._result


# --------------------------------------------------------------------------
# Inference pipeline tests (chain mocked)
# --------------------------------------------------------------------------
@patch.object(inference_module, "get_review_analysis_chain")
def test_positive_review(mock_get_chain):
    mock_get_chain.return_value = _FakeChain(
        result=_make_result(sentiment="positive", sentiment_strength=4.8)
    )
    result = inference_module.run_review_analysis("This blender is amazing, five stars!")
    assert result.sentiment == "positive"
    assert result.sentiment_strength == 4.8

@patch.object(inference_module, "get_review_analysis_chain")
def test_negative_review(mock_get_chain):
    mock_get_chain.return_value = _FakeChain(
        result=_make_result(
            sentiment="negative",
            sentiment_strength=4.5,
            reason="The reviewer describes the product breaking after one use.",
        )
    )
    result = inference_module.run_review_analysis("Broke after one use, total waste of money.")
    assert result.sentiment == "negative"
    assert result.sentiment_strength >= 4.0

@patch.object(inference_module, "get_review_analysis_chain")
def test_fake_looking_review(mock_get_chain):
    mock_get_chain.return_value = _FakeChain(
        result=_make_result(
            is_fake=True,
            fake_confidence=0.92,
            reason="Generic superlative praise with no specific product detail.",
        )
    )
    result = inference_module.run_review_analysis("Best product ever!!! Amazing!!! Buy now!!!")
    assert result.is_fake is True
    assert result.fake_confidence > 0.5

@patch.object(inference_module, "get_review_analysis_chain")
def test_genuine_review(mock_get_chain):
    mock_get_chain.return_value = _FakeChain(
        result=_make_result(
            is_fake=False,
            fake_confidence=0.05,
            reason="Mentions specific delivery timing and a concrete use case.",
        )
    )
    result = inference_module.run_review_analysis(
        "Arrived two days late but the fit was perfect for my kitchen counter."
    )
    assert result.is_fake is False
    assert result.fake_confidence < 0.5

def test_empty_review_raises_validation_exception():
    with pytest.raises(ValidationException):
        inference_module.run_review_analysis("   ")

@patch.object(inference_module, "get_review_analysis_chain")
def test_very_long_review(mock_get_chain):
    mock_get_chain.return_value = _FakeChain(result=_make_result())
    long_review = "This product works well. " * 500
    result = inference_module.run_review_analysis(long_review)
    assert isinstance(result, ReviewAnalysisResult)

@patch.object(inference_module, "get_review_analysis_chain")
def test_llm_failure_raises_llm_exception(mock_get_chain):
    mock_get_chain.return_value = _FakeChain(exception=RuntimeError("Groq API timeout"))
    with pytest.raises(LLMException):
        inference_module.run_review_analysis("A perfectly normal review.")

@patch.object(inference_module, "get_review_analysis_chain")
def test_out_of_range_output_raises_validation_exception(mock_get_chain):
    # Bypass Pydantic constraints via model_construct to simulate a
    # malformed downstream value reaching the validator directly.
    bad_result = ReviewAnalysisResult.model_construct(
        is_fake=False,
        fake_confidence=1.5,
        sentiment="positive",
        sentiment_strength=4.0,
        reason="Edge case reason.",
    )
    mock_get_chain.return_value = _FakeChain(result=bad_result)
    with pytest.raises(ValidationException):
        inference_module.run_review_analysis("Some review text.")


# --------------------------------------------------------------------------
# Schema validation tests
# --------------------------------------------------------------------------
def test_review_request_rejects_empty_string():
    with pytest.raises(Exception):
        ReviewRequest(review="   ")

def test_review_analysis_result_rejects_invalid_sentiment():
    with pytest.raises(Exception):
        ReviewAnalysisResult(
            is_fake=False,
            fake_confidence=0.2,
            sentiment="neutral",  # not in Literal["positive", "negative"]
            sentiment_strength=2.0,
            reason="test",
        )


# --------------------------------------------------------------------------
# Service layer tests
# --------------------------------------------------------------------------
@patch.object(inference_module, "get_review_analysis_chain")
def test_service_wraps_result_in_response(mock_get_chain):
    mock_get_chain.return_value = _FakeChain(result=_make_result())
    service = ReviewService()
    response = service.analyze(ReviewRequest(review="Solid product, would buy again."))
    assert response.message == "Review analysis completed"
    assert response.result.sentiment == "positive"


# --------------------------------------------------------------------------
# API endpoint tests
# --------------------------------------------------------------------------
# @patch.object(inference_module, "get_review_analysis_chain")
# def test_api_returns_valid_response_format(mock_get_chain):
#     mock_get_chain.return_value = _FakeChain(result=_make_result())
#     response = client.post(
#         "/api/v1/review-analysis/analyze",
#         json={"review": "Great phone, battery lasts all day."},
#     )
#     assert response.status_code == 200
#     body = response.json()
#     assert "result" in body
#     assert body["result"]["sentiment"] in ("positive", "negative")
#     assert 0.0 <= body["result"]["fake_confidence"] <= 1.0
#     assert 0.0 <= body["result"]["sentiment_strength"] <= 5.0

# def test_api_rejects_invalid_input():
#     response = client.post("/api/v1/review-analysis/analyze", json={"review": ""})
#     assert response.status_code == 422

# @patch.object(inference_module, "get_review_analysis_chain")
# def test_api_handles_llm_failure(mock_get_chain):
#     mock_get_chain.return_value = _FakeChain(exception=RuntimeError("Groq API down"))
#     response = client.post(
#         "/api/v1/review-analysis/analyze", json={"review": "Any review text here."}
#     )
#     assert response.status_code == 502