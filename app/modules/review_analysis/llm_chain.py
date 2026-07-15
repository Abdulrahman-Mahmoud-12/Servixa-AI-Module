"""
LangChain pipeline for the Review Analysis module.

Wires a Groq-hosted LLM (llama-3.1-8b-instant) into a LangChain
runnable that returns a `ReviewAnalysisResult` object directly via
structured output -- no manual JSON parsing required.
"""

from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

from app.config import settings
from app.exceptions import ConfigurationException
from app.logger import get_logger
from app.modules.review_analysis.prompts import REVIEW_ANALYSIS_SYSTEM_PROMPT
from app.schemas.review_analysis import ReviewAnalysisResult

logger = get_logger(__name__)

def _build_llm() -> ChatGroq:
    """Construct the Groq chat model configured for this module."""
    api_key = settings.review_analysis.groq_api_key
    if not api_key:
        raise ConfigurationException(
            "GROQ_API_KEY is not configured.",
            details={"hint": "Set GROQ_API_KEY in your .env file."},
        )

    return ChatGroq(
        api_key=api_key,
        model=settings.review_analysis.review_model,
        temperature=settings.review_analysis.review_temperature,
    )


def _build_chain() -> Runnable:
    """Build the prompt -> LLM (structured output) runnable chain."""
    llm = _build_llm()
    structured_llm = llm.with_structured_output(ReviewAnalysisResult)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REVIEW_ANALYSIS_SYSTEM_PROMPT),
            ("human", "{review}"),
        ]
    )

    return prompt | structured_llm


@lru_cache(maxsize=1)
def get_review_analysis_chain() -> Runnable:
    """
    Return the process-wide, lazily-built review analysis chain.
    """
    logger.info(
        "Building review analysis LLM chain (model=%s, temperature=%s)",
        settings.review_analysis.review_model,
        settings.review_analysis.review_temperature,
    )
    return _build_chain()