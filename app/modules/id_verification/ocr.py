"""
Text/field extraction from enhanced ID card crops using Groq's Llama 3.2 Vision,
completely free of charge using your local GROQ_API_KEY.
"""

import base64
import os
from pathlib import Path
from typing import Literal
from groq import Groq

from app.config import settings
from app.exceptions import OCRException
from app.logger import get_logger
from app.modules.id_verification.prompts import BACK_ID_PROMPT, FRONT_ID_PROMPT

logger = get_logger(__name__)


def encode_image_to_base64(image_path: Path) -> str:
    """Helper to convert the image file into a base64 string for the Vision API."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as exc:
        raise OCRException(
            "Failed to encode image to base64",
            details={"path": str(image_path), "error": str(exc)},
        ) from exc


def extract_fields(image_path: Path, side: Literal["front", "back"]) -> str:
    """
    Send an enhanced ID card crop to Groq's Llama-3.2-11b-vision-preview model
    and return its raw text response (JSON formatted based on prompt instructions).
    """
    # Fetch your GROQ_API_KEY from settings or environment
    api_key = getattr(settings.llm, "groq_api_key", None) or os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise OCRException(
            "GROQ_API_KEY is not configured",
            details={"hint": "Set GROQ_API_KEY in your .env file"},
        )

    prompt = FRONT_ID_PROMPT if side == "front" else BACK_ID_PROMPT
    base64_image = encode_image_to_base64(image_path)

    try:
        # Initialize Groq Client
        client = Groq(api_key=api_key)

        # Execute Vision Completion using Llama 3.2
        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            # Enforces Groq to return a valid JSON object matching the prompt schema
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        response_text = response.choices[0].message.content
        if not response_text:
            raise OCRException("Groq Vision returned an empty response")

        logger.info("Groq Llama 3.2 Vision extraction succeeded for %s side", side)
        return response_text

    except Exception as exc:
        raise OCRException(
            "Groq OCR Vision extraction failed permanently after retries",
            details={"side": side, "error": str(exc)},
        ) from exc