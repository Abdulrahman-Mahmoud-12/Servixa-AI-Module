"""
Tests for the ID Verification module: detector, cropper, enhancer,
Gemini Vision extraction, parser, validator, the full inference
pipeline, and the API endpoint.

External services (YOLO model weights, the Gemini API) are mocked so
this suite runs deterministically, offline, and without needing real
model files or an API key.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.exceptions import (
    InvalidImageException,
    ModelLoadingException,
    OCRException,
    ValidationException,
)
from app.modules.id_verification.parser import parse_extraction_response
from app.modules.id_verification.validator import validate_national_id

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a small synthetic RGB image on disk for crop/enhance tests."""
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (400, 250), color=(200, 200, 200)).save(image_path)
    return image_path


@pytest.fixture(autouse=True)
def isolate_storage(tmp_path, monkeypatch):
    """Redirect all storage paths to a temp dir so tests never touch real app/storage."""
    from app.config import settings

    monkeypatch.setattr(settings.storage, "uploads_path", tmp_path / "uploads")
    monkeypatch.setattr(settings.storage, "cropped_path", tmp_path / "cropped")
    monkeypatch.setattr(settings.storage, "enhanced_path", tmp_path / "enhanced")


# ---------------------------------------------------------------------------
# Validator - Egyptian National ID structural validation
# ---------------------------------------------------------------------------
class TestValidateNationalID:
    def test_valid_id_returns_expected_fields(self):
        result = validate_national_id("29001011712341")
        assert result["governorate"] == "Monufia"
        assert result["gender"] == "female"
        assert result["birth_date"] == "1990-01-01"

    def test_female_gender_from_even_check_digit(self):
        result = validate_national_id("29001011712342")
        assert result["gender"] == "female"

    def test_missing_id_raises(self):
        with pytest.raises(ValidationException):
            validate_national_id(None)

    def test_wrong_length_raises(self):
        with pytest.raises(ValidationException):
            validate_national_id("12345")

    def test_invalid_century_digit_raises(self):
        with pytest.raises(ValidationException):
            validate_national_id("19001011712341")

    def test_invalid_month_raises(self):
        with pytest.raises(ValidationException):
            validate_national_id("29013011712341")

    def test_unknown_governorate_raises(self):
        with pytest.raises(ValidationException):
            validate_national_id("29001019912341")


# ---------------------------------------------------------------------------
# Parser - normalizing Gemini Vision's JSON response
# ---------------------------------------------------------------------------
class TestParseExtractionResponse:
    def test_parses_plain_json(self):
        raw = '{"full_name": "Ahmed", "national_id": "29001011712341"}'
        result = parse_extraction_response(raw, side="front")
        assert result["full_name"] == "Ahmed"

    def test_strips_markdown_code_fences(self):
        raw = '```json\n{"full_name": "Ahmed"}\n```'
        result = parse_extraction_response(raw, side="front")
        assert result["full_name"] == "Ahmed"

    def test_normalizes_empty_markers_to_none(self):
        raw = '{"address": "null", "governorate": "N/A", "husband_name": ""}'
        result = parse_extraction_response(raw, side="back")
        assert result["address"] is None
        assert result["governorate"] is None
        assert result["husband_name"] is None

    def test_invalid_json_raises(self):
        with pytest.raises(ValidationException):
            parse_extraction_response("not json at all", side="front")

    def test_non_object_json_raises(self):
        with pytest.raises(ValidationException):
            parse_extraction_response("[1, 2, 3]", side="front")


# ---------------------------------------------------------------------------
# Cropper - padding, clamping, saving detected regions
# ---------------------------------------------------------------------------
class TestCropDetections:
    def test_crops_and_saves_highest_confidence_per_class(self, sample_image):
        from app.modules.id_verification.cropper import crop_detections

        detections = [
            {"class_name": "front_id", "confidence": 0.7, "bbox": (10, 10, 100, 80)},
            {"class_name": "front_id", "confidence": 0.95, "bbox": (20, 20, 150, 100)},
            {"class_name": "back_id", "confidence": 0.88, "bbox": (5, 5, 90, 60)},
        ]
        crop_paths = crop_detections(sample_image, detections)

        assert set(crop_paths.keys()) == {"front_id", "back_id"}
        assert all(path.exists() for path in crop_paths.values())

    def test_no_detections_raises(self, sample_image):
        from app.modules.id_verification.cropper import crop_detections

        with pytest.raises(InvalidImageException):
            crop_detections(sample_image, [])

    def test_missing_image_raises(self, tmp_path):
        from app.modules.id_verification.cropper import crop_detections

        with pytest.raises(InvalidImageException):
            crop_detections(
                tmp_path / "missing.jpg",
                [{"class_name": "front_id", "confidence": 0.9, "bbox": (0, 0, 10, 10)}],
            )


# ---------------------------------------------------------------------------
# Enhancer - upscale/deskew/CLAHE/sharpen
# ---------------------------------------------------------------------------
class TestEnhanceCrop:
    def test_enhances_and_saves_image(self, sample_image):
        from app.modules.id_verification.enhancer import enhance_crop

        enhanced_path = enhance_crop(sample_image)
        assert enhanced_path.exists()

        with Image.open(enhanced_path) as img:
            # The 400px-wide input should be upscaled to the target min width.
            assert img.width >= 1000

    def test_unreadable_image_raises(self, tmp_path):
        from app.modules.id_verification.enhancer import enhance_crop

        bad_path = tmp_path / "not_an_image.jpg"
        bad_path.write_text("this is not an image")

        with pytest.raises(InvalidImageException):
            enhance_crop(bad_path)


# ---------------------------------------------------------------------------
# Detector - YOLO wrapper (model itself is mocked, no real weights needed)
# ---------------------------------------------------------------------------
class TestIDCardDetector:
    def test_missing_weights_raises_model_loading_exception(self, tmp_path):
        from app.modules.id_verification.detector import IDCardDetector

        detector = IDCardDetector(model_path=tmp_path / "missing.pt")
        with pytest.raises(ModelLoadingException):
            detector.detect("irrelevant.jpg")

    @patch("app.modules.id_verification.detector.YOLO")
    def test_detect_parses_yolo_results(self, mock_yolo_cls, tmp_path):
        from app.modules.id_verification.detector import IDCardDetector

        weights_path = tmp_path / "yolo.pt"
        weights_path.touch()

        mock_box = MagicMock()
        mock_box.cls = [np.array(0)]
        mock_box.conf = [np.array(0.93)]
        mock_box.xyxy = [np.array([10.0, 10.0, 200.0, 150.0])]

        mock_result = MagicMock()
        mock_result.names = {0: "front_id", 1: "back_id"}
        mock_result.boxes = [mock_box]

        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = [mock_result]
        mock_yolo_cls.return_value = mock_model_instance

        detector = IDCardDetector(model_path=weights_path, confidence_threshold=0.5)
        detections = detector.detect("some_image.jpg")

        assert len(detections) == 1
        assert detections[0]["class_name"] == "front_id"
        assert detections[0]["confidence"] == pytest.approx(0.93)


# ---------------------------------------------------------------------------
# OCR / Gemini Vision extraction (client is mocked - no network/API key needed)
# ---------------------------------------------------------------------------
class TestExtractFields:
    def test_missing_api_key_raises(self, sample_image, monkeypatch):
        from app.config import settings
        from app.modules.id_verification import ocr

        monkeypatch.setattr(settings.llm, "gemini_api_key", "")
        ocr._client = None  # reset the module-level lazy client between tests

        with pytest.raises(OCRException):
            ocr.extract_fields(sample_image, side="front")

    @patch("app.modules.id_verification.ocr.genai")
    def test_successful_extraction_returns_raw_text(
        self, mock_genai, sample_image, monkeypatch
    ):
        from app.config import settings
        from app.modules.id_verification import ocr

        monkeypatch.setattr(settings.llm, "gemini_api_key", "fake-key")
        ocr._client = None

        mock_response = MagicMock()
        mock_response.text = '{"full_name": "Ahmed"}'
        mock_genai.Client.return_value.models.generate_content.return_value = mock_response

        result = ocr.extract_fields(sample_image, side="front")
        assert result == '{"full_name": "Ahmed"}'

    @patch("app.modules.id_verification.ocr.genai")
    def test_empty_gemini_response_raises(self, mock_genai, sample_image, monkeypatch):
        from app.config import settings
        from app.modules.id_verification import ocr

        monkeypatch.setattr(settings.llm, "gemini_api_key", "fake-key")
        ocr._client = None

        mock_response = MagicMock()
        mock_response.text = ""
        mock_genai.Client.return_value.models.generate_content.return_value = mock_response

        with pytest.raises(OCRException):
            ocr.extract_fields(sample_image, side="front")


# ---------------------------------------------------------------------------
# Full pipeline (inference.py) - all sub-steps mocked to test orchestration
# ---------------------------------------------------------------------------
class TestRunIDVerification:
    @patch("app.modules.id_verification.inference.validate_national_id")
    @patch("app.modules.id_verification.inference.parse_extraction_response")
    @patch("app.modules.id_verification.inference.extract_fields")
    @patch("app.modules.id_verification.inference.enhance_crop")
    @patch("app.modules.id_verification.inference.crop_detections")
    @patch("app.modules.id_verification.inference._detector")
    def test_full_pipeline_assembles_front_and_back(
        self,
        mock_detector,
        mock_crop_detections,
        mock_enhance_crop,
        mock_extract_fields,
        mock_parse_response,
        mock_validate,
        sample_image,
    ):
        from app.modules.id_verification.inference import run_id_verification

        mock_detector.detect.return_value = [
            {"class_name": "front_id", "confidence": 0.9, "bbox": (0, 0, 10, 10)},
            {"class_name": "back_id", "confidence": 0.9, "bbox": (0, 0, 10, 10)},
        ]
        mock_crop_detections.return_value = {
            "front_id": sample_image,
            "back_id": sample_image,
        }
        mock_enhance_crop.return_value = sample_image
        mock_extract_fields.side_effect = ['{"national_id": "29001011712341"}', "{}"]
        mock_parse_response.side_effect = [
            {"national_id": "29001011712341", "full_name": "Ahmed"},
            {"occupation": "Engineer"},
        ]
        mock_validate.return_value = {
            "national_id": "29001011712341",
            "birth_date": "1990-01-01",
            "gender": "female",
            "governorate": "Monufia",
        }

        result = run_id_verification(sample_image)

        assert result["is_valid"] is True
        assert result["front"]["full_name"] == "Ahmed"
        assert result["back"]["occupation"] == "Engineer"
        assert result["validation"]["governorate"] == "Monufia"

    @patch("app.modules.id_verification.inference._detector")
    def test_missing_image_raises(self, mock_detector, tmp_path):
        from app.modules.id_verification.inference import run_id_verification

        with pytest.raises(InvalidImageException):
            run_id_verification(tmp_path / "does_not_exist.jpg")


# ---------------------------------------------------------------------------
# API endpoint - service layer mocked to isolate HTTP wiring/error mapping
# ---------------------------------------------------------------------------
class TestVerifyIDEndpoint:
    @patch("app.api.id_verification._service")
    def test_successful_verification_returns_200(self, mock_service, sample_image):
        from app.main import app
        from app.schemas.id_verification import FrontIDFields, IDVerificationResponse

        mock_service.verify = AsyncMock(
            return_value=IDVerificationResponse(
                is_valid=True,
                front=FrontIDFields(full_name="Ahmed", national_id="29001011712341"),
            )
        )

        client = TestClient(app)
        with open(sample_image, "rb") as f:
            response = client.post(
                "/api/v1/router/id-verification/verify",
                files={"file": ("id.jpg", f, "image/jpeg")},
            )

        assert response.status_code == 200
        assert response.json()["is_valid"] is True

    @patch("app.api.id_verification._service")
    def test_service_exception_returns_mapped_status_code(
        self, mock_service, sample_image
    ):
        from app.main import app

        mock_service.verify = AsyncMock(
            side_effect=InvalidImageException("Unsupported image type")
        )

        client = TestClient(app)
        with open(sample_image, "rb") as f:
            response = client.post(
                "/api/v1/router/id-verification/verify",
                files={"file": ("id.jpg", f, "image/jpeg")},
            )

        assert response.status_code == 400