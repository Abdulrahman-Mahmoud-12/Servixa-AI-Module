"""
Centralized logging configuration for Servixa AI.

Usage:
    from app.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Service started")
"""
import logging
import sys
from logging.handlers import RotatingFileHandler

from app.config import settings

# Log record format shared by both console and file handlers.
_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Rotating file handler settings.
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per log file
_BACKUP_COUNT = 5  # Keep up to 5 rotated log files
_configured = False

def configure_logging() -> None:
    """
    Configure the root logger with a console handler and a rotating
    file handler. Safe to call multiple times; configuration is only
    applied once per process.
    """
    global _configured
    if _configured:
        return

    log_level = getattr(logging, settings.logging.level.upper(), logging.INFO)

    # Ensure the directory for the log file exists before attaching
    log_file_path = settings.logging.file_path
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    file_handler = RotatingFileHandler(
        filename=str(log_file_path),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger, ensuring global logging
    configuration has been applied first.
    """
    configure_logging()
    return logging.getLogger(name)