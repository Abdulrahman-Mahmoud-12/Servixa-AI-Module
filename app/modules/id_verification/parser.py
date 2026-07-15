"""
Parse raw Groq OCR Vision responses (JSON-ish text) into normalized
Python dictionaries, tolerating minor formatting noise such as
markdown code fences that models sometimes add despite instructions
not to.
"""
import json
import re
from typing import Any, Dict

from app.exceptions import ValidationException
from app.logger import get_logger

logger = get_logger(__name__)
_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?|```$", re.MULTILINE)
_EMPTY_VALUE_MARKERS = {"", "null", "n/a", "none"}


def _strip_code_fences(raw_text: str) -> str:
    """Remove Markdown code fences from a raw model response."""
    return _CODE_FENCE_PATTERN.sub("", raw_text).strip()

def parse_extraction_response(raw_text: str, side: str) -> Dict[str, Any]:
    """
    Parse a raw Groq OCR Vision response into a plain dictionary.
    """
    cleaned = _strip_code_fences(raw_text)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValidationException(
            "Failed to parse Groq OCR Vision response as JSON",
            details={"side": side, "raw_response": raw_text},
        ) from exc

    if not isinstance(parsed, dict):
        raise ValidationException(
            "Groq OCR Vision response was valid JSON but not an object",
            details={"side": side, "parsed_type": type(parsed).__name__},
        )

    normalized = {
        key: (None if isinstance(value, str) and value.strip().lower() in _EMPTY_VALUE_MARKERS else value)
        for key, value in parsed.items()
    }

    logger.info("Parsed %s side fields: %s", side, list(normalized.keys()))
    return normalized