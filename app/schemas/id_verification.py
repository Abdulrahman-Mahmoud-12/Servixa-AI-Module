"""
Pydantic schemas for the ID verification API: response contracts for
the `/id-verification` endpoints.
"""

from datetime import date
from typing import Literal, Optional
from pydantic import BaseModel, Field

class FrontIDFields(BaseModel):
    """Fields extracted from the front side of the ID card."""
    full_name: Optional[str] = None
    national_id: Optional[str] = None
    address: Optional[str] = None
    governorate: Optional[str] = None

class BackIDFields(BaseModel):
    """Fields extracted from the back side of the ID card."""
    occupation: Optional[str] = None
    address: Optional[str] = None
    husband_name: Optional[str] = None
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None

class NationalIDValidation(BaseModel):
    """Attributes derived and validated from the national ID number itself."""
    national_id: str
    birth_date: date
    gender: Literal["male", "female"]
    governorate: str

class IDVerificationResponse(BaseModel):
    """Response payload returned to the client after verification."""
    is_valid: bool = Field(
        description="Whether a valid national ID number was extracted and validated"
    )
    front: Optional[FrontIDFields] = None
    back: Optional[BackIDFields] = None
    validation: Optional[NationalIDValidation] = None
    message: str = "ID verification completed"