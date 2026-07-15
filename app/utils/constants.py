"""
Application-wide constants for Servixa AI.
"""
from typing import Final

# --- API versioning / routing --------------------------------------------
API_VERSION: Final[str] = "v1"
API_PREFIX: Final[str] = f"/api/{API_VERSION}"

# --- Uploads / media --------------------------------------------------------
SUPPORTED_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
)
SUPPORTED_IMAGE_CONTENT_TYPES: Final[frozenset[str]] = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/bmp"}
)
MAX_UPLOAD_SIZE_MB: Final[int] = 10
MAX_UPLOAD_SIZE_BYTES: Final[int] = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# --- Timeouts ----------------------------------------------------------------
DEFAULT_REQUEST_TIMEOUT_SECONDS: Final[int] = 30
DEFAULT_LLM_TIMEOUT_SECONDS: Final[int] = 60

# --- Pagination (useful once list endpoints exist, e.g. chat history) ------
DEFAULT_PAGE_SIZE: Final[int] = 20
MAX_PAGE_SIZE: Final[int] = 100

# --- Misc application-wide labels -------------------------------------------
HEALTH_TAG: Final[str] = "Health"
PRICING_TAG: Final[str] = "Pricing"
ID_VERIFICATION_TAG: Final[str] = "ID Verification"
REVIEW_ANALYSIS_TAG: Final[str] = "Review Analysis"
CHATBOT_TAG: Final[str] = "Chatbot"