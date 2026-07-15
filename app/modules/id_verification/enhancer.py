"""
Image enhancement steps applied to cropped ID card regions before
sending them to Gemini Vision for field extraction. Enhancement
improves legibility of small printed text under poor lighting, low
resolution, or slight rotation.
"""
from pathlib import Path
import cv2
import numpy as np

from app.config import settings
from app.exceptions import InvalidImageException
from app.logger import get_logger

logger = get_logger(__name__)
_TARGET_MIN_WIDTH = 1000  

def _upscale_if_needed(image: np.ndarray, target_min_width: int = _TARGET_MIN_WIDTH) -> np.ndarray:
    """Upscale the image if it's smaller than the target working resolution."""
    height, width = image.shape[:2]
    if width >= target_min_width:
        return image

    scale = target_min_width / width
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)


def _deskew(image: np.ndarray) -> np.ndarray:
    """Correct minor rotation using the minimum-area bounding rectangle of content."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)

    if coords is None:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Skip correction for negligible angles to avoid introducing artifacts.
    if abs(angle) < 0.5:
        return image

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement on the luminance channel only."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _unsharp_mask(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Sharpen the image using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def enhance_crop(image_path: Path) -> Path:
    """
    Run the full enhancement pipeline (upscale -> deskew -> CLAHE ->
    unsharp mask) on a cropped ID region and save the result to the
    configured enhanced-images directory.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise InvalidImageException(
            "Unable to read image for enhancement", details={"path": str(image_path)}
        )

    image = _upscale_if_needed(image)
    image = _deskew(image)
    image = _apply_clahe(image)
    image = _unsharp_mask(image)

    settings.storage.enhanced_path.mkdir(parents=True, exist_ok=True)
    enhanced_path = settings.storage.enhanced_path / image_path.name
    cv2.imwrite(str(enhanced_path), image)

    logger.info("Enhanced image saved to %s", enhanced_path)
    return enhanced_path