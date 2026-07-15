"""
Service layer for ID verification: bridges the API layer and the
underlying inference pipeline, handling upload persistence and
converting raw pipeline output into the response schema.
"""
import uuid
from pathlib import Path
from fastapi import UploadFile

from app.config import settings
from app.exceptions import InvalidImageException
from app.logger import get_logger
from app.modules.id_verification.inference import run_id_verification
from app.schemas.id_verification import (
    BackIDFields,
    FrontIDFields,
    IDVerificationResponse,
    NationalIDValidation,
)

logger = get_logger(__name__)
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


class IDVerificationService:
    """Orchestrates saving an upload and running it through the ID verification pipeline."""
    async def verify(self, upload_file: UploadFile) -> IDVerificationResponse:
        """
        Save the uploaded image and run the full ID verification
        pipeline against it.
        """
        if upload_file.content_type not in _ALLOWED_CONTENT_TYPES:
            raise InvalidImageException(
                "Unsupported image type",
                details={"content_type": upload_file.content_type},
            )

        settings.storage.uploads_path.mkdir(parents=True, exist_ok=True)
        suffix = Path(upload_file.filename or "upload.jpg").suffix or ".jpg"
        saved_path = settings.storage.uploads_path / f"{uuid.uuid4().hex}{suffix}"

        contents = await upload_file.read()
        saved_path.write_bytes(contents)
        logger.info("Saved uploaded ID image to %s", saved_path)

        result = run_id_verification(saved_path)

        return IDVerificationResponse(
            is_valid=result["is_valid"],
            front=FrontIDFields(**result["front"]) if result.get("front") else None,
            back=BackIDFields(**result["back"]) if result.get("back") else None,
            validation=(
                NationalIDValidation(**result["validation"])
                if result.get("validation")
                else None
            ),
        )