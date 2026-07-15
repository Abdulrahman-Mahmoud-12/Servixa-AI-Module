"""
Crop detected ID card regions (front/back) from the source image,
applying padding around each bounding box and persisting the crop
to the configured storage location.
"""

import uuid
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

from app.config import settings
from app.exceptions import InvalidImageException
from app.logger import get_logger
from app.modules.id_verification.detector import Detection

logger = get_logger(__name__)
_PADDING_RATIO = 0.05

def _pad_box(
    bbox: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
    ratio: float = _PADDING_RATIO,
) -> Tuple[float, float, float, float]:
    """Expand a bounding box by `ratio` on each side, clamped to image bounds."""
    x1, y1, x2, y2 = bbox
    width, height = image_size

    pad_x = (x2 - x1) * ratio
    pad_y = (y2 - y1) * ratio

    return (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(width), x2 + pad_x),
        min(float(height), y2 + pad_y),
    )


def crop_detections(image_path: Path, detections: List[Detection]) -> Dict[str, Path]:
    """
    Crop each detected region from the source image and save it to
    `settings.storage.cropped_path`.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise InvalidImageException(
            "Unable to open image for cropping",
            details={"path": str(image_path), "error": str(exc)},
        ) from exc

    settings.storage.cropped_path.mkdir(parents=True, exist_ok=True)

    best_by_class: Dict[str, Detection] = {}
    for detection in detections:
        current_best = best_by_class.get(detection["class_name"])
        if current_best is None or detection["confidence"] > current_best["confidence"]:
            best_by_class[detection["class_name"]] = detection

    if not best_by_class:
        raise InvalidImageException(
            "No ID card regions detected in the provided image",
            details={"path": str(image_path)},
        )

    crop_paths: Dict[str, Path] = {}
    for class_name, detection in best_by_class.items():
        padded_bbox = _pad_box(detection["bbox"], image.size)
        crop = image.crop(padded_bbox)

        crop_filename = f"{class_name}_{uuid.uuid4().hex}.jpg"
        crop_path = settings.storage.cropped_path / crop_filename
        crop.save(crop_path, format="JPEG", quality=95)

        crop_paths[class_name] = crop_path
        logger.info("Saved %s crop to %s", class_name, crop_path)

    return crop_paths