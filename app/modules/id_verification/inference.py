"""
End-to-end ID verification pipeline: detect card regions, crop,
enhance, extract fields via Gemini Vision, parse, and validate.
Only processes the front side to remain within Free Tier API limits.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from app.exceptions import InvalidImageException
from app.logger import get_logger
from app.modules.id_verification.cropper import crop_detections
from app.modules.id_verification.detector import IDCardDetector
from app.modules.id_verification.enhancer import enhance_crop
from app.modules.id_verification.ocr import extract_fields
from app.modules.id_verification.parser import parse_extraction_response
from app.modules.id_verification.validator import validate_national_id

logger = get_logger(__name__)
_detector = IDCardDetector()

def run_id_verification(image_path: Path) -> Dict[str, Any]:
    """
    Run the ID verification pipeline on an uploaded image,
    specifically isolating and extracting data only from the front ID.
    """
    if not image_path.exists():
        raise InvalidImageException(
            "Uploaded image not found", details={"path": str(image_path)}
        )

    # 1. Detect bounding boxes
    detections = _detector.detect(str(image_path))
    
    # 2. Crop the detected regions
    crop_paths = crop_detections(image_path, detections)

    # Initialize result with only front data
    result: Dict[str, Any] = {"front": None, "back": None, "is_valid": False}

    # Only look for "front_id" in the crops
    front_crop_path = crop_paths.get("front_id")
    if front_crop_path is not None:
        logger.info("Processing only the front ID side to preserve Free Tier quota.")
        # 3. Enhance the front crop
        enhanced_path = enhance_crop(front_crop_path)
        
        # 4. Extract data using Gemini Vision
        raw_response = extract_fields(enhanced_path, side="front")
        
        # 5. Parse JSON fields
        parsed_fields = parse_extraction_response(raw_response, side="front")
        result["front"] = parsed_fields
    else:
        logger.warning("No front_id region detected in the uploaded image.")

    # 6. Validate the National ID from the front side
    front_fields = result.get("front") or {}
    national_id = front_fields.get("national_id")

    validation: Optional[Dict[str, Any]] = None
    if national_id:
        try:
            validation = validate_national_id(national_id)
            result["is_valid"] = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("National ID validation failed: %s", exc)
            validation = {"error": str(exc)}

    result["validation"] = validation
    return result