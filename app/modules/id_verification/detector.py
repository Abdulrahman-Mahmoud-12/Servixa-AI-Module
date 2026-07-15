"""
YOLO-based detector for locating Egyptian National ID card faces
(front and back) within an uploaded image.
"""

from pathlib import Path
from typing import List, Optional, TypedDict, Union
import numpy as np
from ultralytics import YOLO

from app.config import settings
from app.exceptions import ModelLoadingException, PredictionException
from app.logger import get_logger

logger = get_logger(__name__)

class Detection(TypedDict):
    """A single YOLO detection result."""
    class_name: str
    confidence: float
    bbox: tuple

class IDCardDetector:
    """
    Wraps an Ultralytics YOLO model trained to detect `front_id` and
    `back_id` regions on an Egyptian National ID card photo.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        self._model_path = model_path or settings.id_verification.yolo_model_path
        self._confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.id_verification.yolo_confidence_threshold
        )
        self._model: Optional[YOLO] = None

    def load(self) -> None:
        """Load the YOLO model weights into memory. Idempotent."""
        if self._model is not None:
            return

        if not self._model_path.exists():
            raise ModelLoadingException(
                "YOLO ID detector weights not found",
                details={"path": str(self._model_path)},
            )

        try:
            logger.info("Loading YOLO ID detector from %s", self._model_path)
            self._model = YOLO(str(self._model_path))
        except Exception as exc:  # noqa: BLE001 - re-raised as domain exception
            raise ModelLoadingException(
                "Failed to load YOLO ID detector",
                details={"path": str(self._model_path), "error": str(exc)},
            ) from exc

    def detect(self, image: Union[str, Path, np.ndarray]) -> List[Detection]:
        """
        Run detection on an image and return bounding boxes for
        `front_id` / `back_id` classes above the configured
        confidence threshold.
        """
        self.load()
        assert self._model is not None  

        try:
            results = self._model.predict(
                source=image,
                conf=self._confidence_threshold,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise PredictionException(
                "YOLO detection failed", details={"error": str(exc)}
            ) from exc

        detections: List[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                detections.append(
                    Detection(
                        class_name=names[class_id],
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        logger.info("Detected %d ID region(s)", len(detections))
        return detections