from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Shared config so every nested settings group -- not just the top-level
# `Settings` class -- knows to read values from the same `.env` file.
_ENV_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    populate_by_name=True,
)


class AppSettings(BaseSettings):
    """Core application identity and server runtime settings."""

    model_config = _ENV_CONFIG

    name: str = Field(default="Servixa AI", alias="APP_NAME")

    version: str = Field(default="0.1.0", alias="APP_VERSION")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    debug: bool = Field(default=False, alias="DEBUG")


class LoggingSettings(BaseSettings):
    """Logging verbosity and output destination."""

    model_config = _ENV_CONFIG

    level: str = Field(default="INFO", alias="LOG_LEVEL")
    file_path: Path = Field(
        default=Path("app/storage/logs/servixa.log"), alias="LOG_FILE_PATH"
    )


class StorageSettings(BaseSettings):
    """Filesystem locations used for uploads, intermediate, and cached artifacts."""

    model_config = _ENV_CONFIG

    storage_path: Path = Field(default=Path("app/storage"), alias="STORAGE_PATH")
    uploads_path: Path = Field(
        default=Path("app/storage/uploads"), alias="UPLOADS_PATH"
    )
    cropped_path: Path = Field(
        default=Path("app/storage/cropped"), alias="CROPPED_PATH"
    )
    enhanced_path: Path = Field(
        default=Path("app/storage/enhanced"), alias="ENHANCED_PATH"
    )
    cache_path: Path = Field(default=Path("app/storage/cache"), alias="CACHE_PATH")
    outputs_path: Path = Field(
        default=Path("app/storage/outputs"), alias="OUTPUTS_PATH"
    )


class PricingModelSettings(BaseSettings):
    """Paths to the trained pricing model and its preprocessing pipeline."""

    model_config = _ENV_CONFIG

    model_path: Path = Field(
        default=Path("app/modules/pricing/models/model.pkl"),
        alias="PRICING_MODEL_PATH",
    )
    pipeline_path: Path = Field(
        default=Path("app/modules/pricing/models/pipeline.pkl"),
        alias="PRICING_PIPELINE_PATH",
    )


class IDVerificationSettings(BaseSettings):
    """Configuration for the YOLO-based ID card detection module."""

    model_config = _ENV_CONFIG

    yolo_model_path: Path = Field(
        default=Path("app/modules/id_verification/models/yolo_id_detector.pt"),
        alias="YOLO_MODEL_PATH",
    )
    yolo_confidence_threshold: float = Field(
        default=0.5, alias="YOLO_CONFIDENCE_THRESHOLD"
    )


class ReviewAnalysisSettings(BaseSettings):
    """Configuration for the Groq-based Review Analysis module."""

    model_config = _ENV_CONFIG

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    review_model: str = Field(default="llama-3.1-8b-instant", alias="REVIEW_MODEL")
    review_temperature: float = Field(default=0.2, alias="REVIEW_TEMPERATURE")


class LLMSettings(BaseSettings):
    """Configuration for the Groq Llama (OCR) LLM client."""

    model_config = _ENV_CONFIG

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    ocr_model: str = Field(default="llama-3.2-11b-vision-preview", alias="OCR_MODEL")
    ocr_temperature: float = Field(default=0.2, alias="OCR_TEMPERATURE")


class RAGSettings(BaseSettings):
    """Configuration for the retrieval-augmented generation chatbot module."""

    model_config = _ENV_CONFIG

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    vector_db_path: Path = Field(
        default=Path("app/modules/rag_chatbot/vectordb"), alias="VECTOR_DB_PATH"
    )


class Settings(BaseSettings):
    """
    Top-level settings object composing all configuration sections.

    Each attribute is a nested settings group so configuration stays
    organized by concern while still being loaded from a single flat
    `.env` file.
    """

    model_config = _ENV_CONFIG

    app: AppSettings = Field(default_factory=AppSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    pricing: PricingModelSettings = Field(default_factory=PricingModelSettings)
    id_verification: IDVerificationSettings = Field(
        default_factory=IDVerificationSettings
    )
    review_analysis: ReviewAnalysisSettings = Field(
        default_factory=ReviewAnalysisSettings
    )
    llm: LLMSettings = Field(default_factory=LLMSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)


# Single global settings instance used throughout the application.
settings = Settings()