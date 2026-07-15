"""
Validation logic for the Egyptian National ID number (14 digits) and
attributes derived from it: birth date, gender, and governorate.
"""
import re
from datetime import date
from typing import Dict, Optional

from app.exceptions import ValidationException
from app.logger import get_logger

logger = get_logger(__name__)
_ID_PATTERN = re.compile(r"^\d{14}$")

# Egyptian governorate codes as printed in digits 8-9 of the National ID.
_GOVERNORATE_CODES: Dict[str, str] = {
    "01": "Cairo",
    "02": "Alexandria",
    "03": "Port Said",
    "04": "Suez",
    "11": "Damietta",
    "12": "Dakahlia",
    "13": "Sharqia",
    "14": "Qalyubia",
    "15": "Kafr El Sheikh",
    "16": "Gharbia",
    "17": "Monufia",
    "18": "Beheira",
    "19": "Ismailia",
    "21": "Giza", "22":
    "Beni Suef",
    "23": "Fayoum",
    "24": "Minya",
    "25": "Assiut",
    "26": "Sohag",
    "27": "Qena",
    "28": "Aswan",
    "29": "Luxor",
    "31": "Red Sea",
    "32": "New Valley",
    "33": "Matrouh",
    "34": "North Sinai",
    "35": "South Sinai",
    "88": "Foreign Born"
}


def validate_national_id(national_id: Optional[str]) -> Dict[str, object]:
    """
    Validate an Egyptian National ID number and derive birth date,
    gender, and governorate from it.

    National ID structure (14 digits):
        [century][yy][mm][dd][governorate: 2 digits][sequence: 4 digits][check: 1 digit]
    """
    if not national_id:
        raise ValidationException("National ID number was not extracted")

    digits = national_id.strip().replace(" ", "")
    if not _ID_PATTERN.match(digits):
        raise ValidationException(
            "National ID must be exactly 14 digits",
            details={"national_id": national_id},
        )

    century_digit = digits[0]
    if century_digit not in ("2", "3"):
        raise ValidationException(
            "National ID century digit must be 2 (1900s) or 3 (2000s)",
            details={"national_id": national_id},
        )
    century_base = 1900 if century_digit == "2" else 2000

    year = century_base + int(digits[1:3])
    month = int(digits[3:5])
    day = int(digits[5:7])

    try:
        birth_date = date(year, month, day)
    except ValueError as exc:
        raise ValidationException(
            "National ID encodes an invalid birth date",
            details={"national_id": national_id, "year": year, "month": month, "day": day},
        ) from exc

    governorate_code = digits[7:9]
    governorate = _GOVERNORATE_CODES.get(governorate_code)
    if governorate is None:
        raise ValidationException(
            "National ID contains an unrecognized governorate code",
            details={"national_id": national_id, "governorate_code": governorate_code},
        )

    # 14th digit (index 12): odd = male, even = female.
    gender = "male" if int(digits[12]) % 2 == 1 else "female"
    logger.info("Validated national ID for governorate=%s gender=%s", governorate, gender)

    return {
        "national_id": digits,
        "birth_date": birth_date.isoformat(),
        "gender": gender,
        "governorate": governorate,
    }