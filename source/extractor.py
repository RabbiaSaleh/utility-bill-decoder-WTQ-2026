"""
Level-1 Extraction Engine.

Pipeline:
1. OCR text extraction
2. Multimodal LLM structured extraction (image + OCR text)
3. Schema cleaning & verification (sum consistency)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from llm_client import LLMClient
from ocr_engine import extract_text
from schema import EMPTY_TEMPLATE, to_json_string, validate_and_clean

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are an expert at reading Pakistani electricity bills (K-Electric / KE, LESCO, IESCO).

Extract ONLY the non-identifying billing details into a single JSON object.
Do NOT invent values. If a field is missing, blank, covered by a grey box, or unreadable, use null.
Use numbers (not strings) for all numeric fields. Credits / subsidies are negative numbers.
Dates must be YYYY-MM-DD. Bill month must be YYYY-MM.

Return EXACTLY this JSON structure (no extra wrapper keys):

{
  "provider": "KE" | "LESCO" | "IESCO" | null,
  "tariff": "exact tariff code as printed, e.g. A1-R" | null,
  "sanctioned_load_kw": number | null,
  "bill_month": "YYYY-MM" | null,
  "reading_date": "YYYY-MM-DD" | null,
  "issue_date": "YYYY-MM-DD" | null,
  "due_date": "YYYY-MM-DD" | null,
  "previous_reading": number | null,
  "current_reading": number | null,
  "units_consumed": number | null,
  "charges": [
    {"type": "energy|fixed|fpa|quarterly_adjustment|surcharge|meter_rent|subsidy|other", "amount": number}
  ],
  "total_charges": number | null,
  "taxes": [
    {"type": "gst|electricity_duty|income_tax|municipal_tax|other_tax", "amount": number}
  ],
  "total_taxes": number | null,
  "current_bill": number | null,
  "arrears": number | null,
  "payable_within_due_date": number | null,
  "payable_after_due_date": number | null,
  "usage_history": [
    {"period": "July 2025", "units": 239}
  ],
  "payment_history": []
}

Charge type mapping:
- energy: Variable / Cost of Electricity / Energy Charges / Units
- fixed: Fixed charges
- fpa: Fuel Price Adjustment / FPA / FCA
- quarterly_adjustment: QTA / Uniform Quarterly Adjustment
- surcharge: Additional Surcharge / F.C Surcharge / PHL
- meter_rent: Meter rent / Service rent
- subsidy: Government subsidy or relief (make amount negative)
- other: anything else in charges section

Tax type mapping:
- gst: Sales tax / GST (including GST on FPA)
- electricity_duty: Electricity Duty / ED
- income_tax: Income tax
- municipal_tax: MUCT / KMC / local municipal
- other_tax: Extra tax, further tax, TV fee, etc.

Rules:
- Skip blank lines and lines printed as 0.
- Keep separate entries when the same type appears more than once.
- Classify by SECTION (charges vs taxes), not just the name.
- If only a combined tax amount is printed, put [] in taxes and the amount in total_taxes.
- payable_after_due_date = the HIGHEST late-payment amount shown.
- Exclude names, addresses, CNIC, account numbers, reference numbers, meter numbers.
- usage_history and payment_history are optional internal data (for Level-2). If not readable, return [].
- Return ONLY valid JSON. No markdown, no commentary.

OCR text from the bill (may contain errors):
---
__OCR_TEXT__
---
"""


def _parse_json_from_llm(text: str) -> dict:
    """Extract the first JSON object from LLM response."""
    text = text.strip()
    # Remove markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    # Find outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        logger.warning("No JSON object found in LLM response")
        return dict(EMPTY_TEMPLATE)
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error: %s", e)
        return dict(EMPTY_TEMPLATE)


def verify_consistency(data: dict) -> dict:
    """
    Light verification: if printed totals exist, keep them.
    Optionally flag large mismatches (logged only).
    Does NOT overwrite printed values (competition rule).
    """
    charges_sum = sum(c.get("amount", 0) for c in (data.get("charges") or []) if isinstance(c, dict))
    taxes_sum = sum(t.get("amount", 0) for t in (data.get("taxes") or []) if isinstance(t, dict))

    tc = data.get("total_charges")
    tt = data.get("total_taxes")

    if tc is not None and charges_sum and abs(float(tc) - charges_sum) > 2:
        logger.info(
            "total_charges mismatch: printed=%.2f vs sum=%.2f", float(tc), charges_sum
        )
    if tt is not None and taxes_sum and abs(float(tt) - taxes_sum) > 2:
        logger.info(
            "total_taxes mismatch: printed=%.2f vs sum=%.2f", float(tt), taxes_sum
        )

    return data


def extract_bill(
    image_path: str | Path,
    llm: LLMClient | None = None,
    use_vision: bool = True,
    return_context: bool = False,
) -> dict:
    """
    Full Level-1 pipeline for one bill.

    By default returns the cleaned Level-1 dict (ready for CSV).
    If return_context=True, returns:
        {
          "level1": <cleaned Level-1 dict>,
          "usage_history": [...],
          "payment_history": [...],
          "ocr_text": "..."
        }
    """
    image_path = Path(image_path)
    logger.info("Extracting: %s", image_path.name)

    # Step 1 – OCR
    ocr = extract_text(image_path)
    ocr_text = (ocr.get("raw_text") or "")[:6000]

    # Step 2 – LLM structured extraction
    if llm is None:
        llm = LLMClient()

    prompt = EXTRACTION_PROMPT.replace(
        "__OCR_TEXT__",
        ocr_text or "(OCR produced no text)",
    )

    try:
        if use_vision:
            raw = llm.generate(prompt, image_path=image_path, temperature=0.05)
        else:
            raw = llm.generate(prompt, temperature=0.05)
    except Exception as e:
        logger.error("LLM extraction failed for %s: %s", image_path.name, e)
        empty = dict(EMPTY_TEMPLATE)
        if return_context:
            return {
                "level1": empty,
                "usage_history": [],
                "payment_history": [],
                "ocr_text": ocr_text,
            }
        return empty

    parsed = _parse_json_from_llm(raw)

    # Support both flat response and {"level1": {...}} wrapper
    if isinstance(parsed.get("level1"), dict):
        level1_payload = parsed["level1"]
        usage_history = parsed.get("usage_history") or []
        payment_history = parsed.get("payment_history") or []
    else:
        level1_payload = parsed
        usage_history = []
        payment_history = []
        if isinstance(parsed, dict):
            usage_history = parsed.pop("usage_history", []) or []
            payment_history = parsed.pop("payment_history", []) or []

    cleaned = validate_and_clean(level1_payload)
    cleaned = verify_consistency(cleaned)

    # Heuristic provider from filename if still missing
    if not cleaned.get("provider"):
        name = image_path.stem.upper()
        if name.startswith("KESC") or name.startswith("KE"):
            cleaned["provider"] = "KE"
        elif name.startswith("LESCO"):
            cleaned["provider"] = "LESCO"
        elif name.startswith("IESCO"):
            cleaned["provider"] = "IESCO"

    if return_context:
        return {
            "level1": cleaned,
            "usage_history": usage_history if isinstance(usage_history, list) else [],
            "payment_history": payment_history if isinstance(payment_history, list) else [],
            "ocr_text": ocr_text,
        }

    return cleaned


def extract_to_json_string(image_path: str | Path, llm: LLMClient | None = None) -> str:
    data = extract_bill(image_path, llm=llm)
    return to_json_string(data)