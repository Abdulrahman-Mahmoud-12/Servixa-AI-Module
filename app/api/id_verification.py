"""
API routes for ID verification. Accepts an uploaded Egyptian National
ID card image and returns extracted, validated fields.
"""
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.exceptions import ServixaException
from app.logger import get_logger
from app.schemas.id_verification import IDVerificationResponse
from app.services.id_service import IDVerificationService

logger = get_logger(__name__)
router = APIRouter(prefix="/id-verification", tags=["ID Verification"])

# Single service instance reused across requests.
_service = IDVerificationService()


@router.post("/verify", response_model=IDVerificationResponse)
async def verify_id(file: UploadFile = File(...)) -> IDVerificationResponse:
    """
    Verify an Egyptian National ID card image.
    """
    try:
        return await _service.verify(file)
    except ServixaException as exc:
        logger.warning("ID verification failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc