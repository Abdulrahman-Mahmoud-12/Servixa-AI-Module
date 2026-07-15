"""
Prompt templates for extracting structured fields from Egyptian
National ID card images using Gemini Vision.
"""

FRONT_ID_PROMPT = """
You are analyzing the FRONT side of an Egyptian National ID card.
Extract the following fields exactly as printed, and return ONLY a
single valid JSON object with no markdown formatting, no code fences,
and no explanation text.

Fields to extract:
- "full_name": the cardholder's full name in Arabic, it's located as a first name and the last name in the next line only.
- "national_id": the 14-digit national ID number, digits only
- "address": the printed address, if visible on the front, it's locatedon a two lines under the first and second name directly.

If a field is not visible or not present, set its value to null.

Return format example:
{"full_name": "...", "national_id": "...", "address": "..."}
""".strip()

BACK_ID_PROMPT = """
You are analyzing the BACK side of an Egyptian National ID card.
Extract the following fields exactly as printed, and return ONLY a
single valid JSON object with no markdown formatting, no code fences,
and no explanation text.

Fields to extract:
- "occupation": the cardholder's occupation/job title
- "marital_status": the cardholder's marital status if 'single', 'maried', ... and so on
- "issue_date": the card issue date, in YYYY-MM-DD format if possible
- "expiry_date": the card expiry date, in YYYY-MM-DD format if possible

If a field is not visible or not present, set its value to null.

Return format example:
{"occupation": "...", "marital_status": "...", "issue_date": "...", "expiry_date": "..."}
""".strip()