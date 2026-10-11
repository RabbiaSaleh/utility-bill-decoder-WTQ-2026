"""
Level-1 JSON schema and validation helpers.
Matches the exact structure required by the competition guide.
"""

from __future__ import annotations

import json
from typing import Any

# Allowed charge types (section 5.4.1)
CHARGE_TYPES = {
    "energy",
    "fixed",
    "fpa",
    "quarterly_adjustment",
    "surcharge",
    "meter_rent",
    "subsidy",
    "other",
}

# Allowed tax types (section 5.4.2)
TAX_TYPES = {
    "gst",
    "electricity_duty",
    "income_tax",
    "municipal_tax",
    "other_tax",
}

PROVIDERS = {"KE", "LESCO", "IESCO"}

EMPTY_TEMPLATE: dict[str, Any] = {
    "provider": None,
    "tariff": None,
    "sanctioned_load_kw": None,
    "bill_month": None,
    "reading_date": None,
    "issue_date": None,
    "due_date": None,
    "previous_reading": None,
    "current_reading": None,
    "units_consumed": None,
    "charges": [],
    "total_charges": None,
    "taxes": [],
    "total_taxes": None,
    "current_bill": None,
    "arrears": None,
    "payable_within_due_date": None,
    "payable_after_due_date": None,
}


def normalize_provider(name: str | None) -> str | None:
    if not name:
        return None
    n = name.strip().upper()
    if n in ("K-ELECTRIC", "KESC", "KE"):
        return "KE"
    if "LESCO" in n:
        return "LESCO"
    if "IESCO" in n:
        return "IESCO"
    if n in PROVIDERS:
        return n
    return None


def safe_number(val: Any) -> float | int | None:
    """Convert printed amount to number. Handles commas, CR/DR, Rs."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    s = str(val).strip()
    if not s or s.lower() in ("null", "none", "-", "n/a"):
        return None
    s = s.replace(",", "").replace("Rs.", "").replace("Rs", "").replace("PKR", "")
    s = s.replace(" ", "")
    negative = False
    if s.upper().endswith("CR") or s.startswith("-") or s.startswith("("):
        negative = True
        s = s.upper().replace("CR", "").replace("(", "").replace(")", "").lstrip("-")
    if s.upper().endswith("DR"):
        s = s.upper().replace("DR", "")
    try:
        num = float(s)
        if num.is_integer():
            num = int(num)
        return -num if negative else num
    except ValueError:
        return None


def validate_and_clean(data: dict) -> dict:
    """Ensure every required key exists and types are correct."""
    out = dict(EMPTY_TEMPLATE)

    # Simple scalar fields
    for key in (
        "provider",
        "tariff",
        "bill_month",
        "reading_date",
        "issue_date",
        "due_date",
    ):
        val = data.get(key)
        if val is not None and str(val).strip().lower() not in ("null", "none", ""):
            out[key] = str(val).strip()

    out["provider"] = normalize_provider(out.get("provider") or data.get("provider"))

    for key in (
        "sanctioned_load_kw",
        "previous_reading",
        "current_reading",
        "units_consumed",
        "total_charges",
        "total_taxes",
        "current_bill",
        "arrears",
        "payable_within_due_date",
        "payable_after_due_date",
    ):
        out[key] = safe_number(data.get(key))

    # Charges
    charges = data.get("charges") or []
    cleaned_charges = []

    for c in charges:
        if not isinstance(c, dict):
            continue

        charge_type = str(
            c.get("type", "other")
        ).lower().strip()

        if charge_type not in CHARGE_TYPES:
            charge_type = "other"

        amount = safe_number(c.get("amount"))

        # Subsidies are deductions and must always be negative.
        # Existing negative values, such as 38209CR -> -38209,
        # are not changed.
        if (
            charge_type == "subsidy"
            and amount is not None
            and amount > 0
        ):
            amount = -amount

        # Skip missing and zero-valued lines.
        if amount is not None and amount != 0:
            cleaned_charges.append(
                {
                    "type": charge_type,
                    "amount": amount,
                }
            )

    out["charges"] = cleaned_charges

    # Taxes
    taxes = data.get("taxes") or []
    cleaned_taxes = []
    for t in taxes:
        if not isinstance(t, dict):
            continue
        typ = str(t.get("type", "other_tax")).lower().strip()
        if typ not in TAX_TYPES:
            typ = "other_tax"
        amt = safe_number(t.get("amount"))
        if amt is not None and amt != 0:
            cleaned_taxes.append({"type": typ, "amount": amt})
    out["taxes"] = cleaned_taxes

    return out


def to_json_string(data: dict) -> str:
    """Compact JSON for CSV cell (no extra spaces)."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

def normalize_full_date(value: Any) -> str | None:
    """
    Return YYYY-MM-DD only when a complete date is explicitly present.
    Never infer a missing day or year.
    """
    ...
    
def normalize_bill_month(value: Any) -> str | None:
    """
    Return YYYY-MM only when month and year are explicitly present.
    """
    ... 
    