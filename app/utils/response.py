"""
Reusable, consistent JSON response helpers for Servixa AI.
"""

from typing import Any, Dict, Optional, Union
from fastapi.responses import JSONResponse

# Shape accepted for the `errors` field: a single message, a list of
# field-level issues, or a free-form dict (e.g. pydantic error output).
ErrorDetail = Union[str, list, Dict[str, Any]]


def success_response(
    data: Optional[Any] = None,
    message: str = "Request completed successfully.",
    status_code: int = 200,
) -> JSONResponse:
    """
    Build a standardized success response.
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "success": True,
            "message": message,
            "data": data,
            "errors": None,
        },
    )


def error_response(
    message: str = "An unexpected error occurred.",
    errors: Optional[ErrorDetail] = None,
    status_code: int = 500,
) -> JSONResponse:
    """
    Build a standardized error response.
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "message": message,
            "data": None,
            "errors": errors,
        },
    )