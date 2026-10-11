"""
Level-2 Question Answering Engine.

Uses deterministic Python calculations for common numeric questions,
falls back to LLM for everything else.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from llm_client import LLMClient

logger = logging.getLogger(__name__)

QA_PROMPT = """You are a helpful electricity-bill assistant for Pakistani consumers (KE, LESCO, IESCO).

Answer the customer's question in clear, plain English using ONLY the information from this bill.
Do not invent numbers or use external tariff rates.
If a value needed for the answer is missing, say so clearly instead of guessing.
When you calculate or estimate, state that it is an estimate and briefly show the arithmetic.
Exclude any personal identifiers (names, addresses, account numbers, CNIC, meter numbers).

Bill data (structured JSON):
```json
{bill_json}
```

Additional OCR text from the bill (may help with history tables, notes, etc.):
---
{ocr_text}
---

Customer question:
{question}

Write a concise, accurate answer (2–6 sentences). Return only the answer text.
"""


def format_pkr(value) -> str:
    if value is None:
        return "PKR amount unavailable"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "PKR amount unavailable"
    if value.is_integer():
        return f"PKR {value:,.0f}"
    return f"PKR {value:,.2f}"


def deterministic_answer(
    question: str,
    bill_data: dict,
    usage_history: list[dict] | None = None,
) -> str | None:
    """
    Answer common numeric questions using Python.
    Return None when LLM fallback is needed.
    """
    q = question.lower().strip()
    usage_history = usage_history or []

    # Tax percentage / how much of bill is taxes
    if "tax" in q and (
        "percentage" in q or "percent" in q or "how much of my bill" in q
    ):
        total_taxes = bill_data.get("total_taxes")
        current_bill = bill_data.get("current_bill")
        if total_taxes is None or current_bill is None:
            return (
                "The bill does not show enough information to "
                "calculate the tax percentage."
            )
        if current_bill == 0:
            return "The tax percentage cannot be calculated because the current bill is zero."
        percentage = round((float(total_taxes) / float(current_bill)) * 100, 1)
        return (
            f"Taxes and duties total {format_pkr(total_taxes)} "
            f"out of the current bill of {format_pkr(current_bill)}, "
            f"which is approximately {percentage}%."
        )

    # Difference between payable amounts
    if "difference" in q and "payable" in q:
        payable_within = bill_data.get("payable_within_due_date")
        payable_after = bill_data.get("payable_after_due_date")
        if payable_within is None or payable_after is None:
            return (
                "The bill does not show both payable amounts "
                "needed to calculate the difference."
            )
        difference = float(payable_after) - float(payable_within)
        return (
            f"The difference between the amount payable after the "
            f"due date and within the due date is {format_pkr(difference)}."
        )

    # Units consumed
    if (
        "units consumed" in q
        or "units did i consume" in q
        or "how many units" in q
    ):
        units = bill_data.get("units_consumed")
        if units is None:
            return "The bill does not show the units consumed."
        return f"The bill shows {units} units consumed."

    # Payable after due date
    if (
        "payable after" in q
        or "after the due date" in q
        or "after due date" in q
    ):
        amount = bill_data.get("payable_after_due_date")
        if amount is None:
            return (
                "The bill does not show the amount payable "
                "after the due date."
            )
        return (
            f"The amount payable after the due date is "
            f"{format_pkr(amount)}."
        )

    # Due date
    if "due date" in q and "after" not in q and "difference" not in q:
        due = bill_data.get("due_date")
        if due is None:
            return "The bill does not show the due date."
        return f"The due date for payment without surcharge is {due}."

    # Arrears / previous dues
    if "arrears" in q or "previous dues" in q or "previous due" in q:
        arrears = bill_data.get("arrears")
        if arrears is None:
            return "The bill does not show previous dues / arrears."
        return f"Previous dues / arrears on this bill are {format_pkr(arrears)}."

    # Usage history above threshold
    if "history" in q and "above" in q and "unit" in q:
        match = re.search(r"above\s+(\d+(?:\.\d+)?)", q)
        if match:
            threshold = float(match.group(1))
            if not usage_history:
                return (
                    "The bill does not show enough readable usage "
                    "history to calculate this."
                )
            count = 0
            for item in usage_history:
                units = item.get("units")
                if isinstance(units, (int, float)) and units > threshold:
                    count += 1
            return (
                f"{count} month(s) in the available usage history "
                f"went above {threshold:g} units."
            )

    return None


def answer_question(
    question: str,
    bill_data: dict[str, Any],
    ocr_text: str = "",
    llm: LLMClient | None = None,
    usage_history: list[dict] | None = None,
) -> str:
    """Generate one Level-2 answer grounded in the bill."""

    # Unwrap if nested
    if isinstance(bill_data, dict) and "level1" in bill_data:
        usage_history = usage_history or bill_data.get("usage_history")
        bill_data = bill_data["level1"]

    # Try deterministic first
    direct = deterministic_answer(
        question=question,
        bill_data=bill_data,
        usage_history=usage_history,
    )
    if direct is not None:
        return direct

    # LLM fallback
    if llm is None:
        llm = LLMClient()

    prompt = QA_PROMPT.format(
        bill_json=json.dumps(bill_data, ensure_ascii=False, indent=2),
        ocr_text=(ocr_text or "(none)")[:4000],
        question=question.strip(),
    )

    try:
        answer = llm.generate(prompt, temperature=0.2, max_tokens=512)
        answer = (answer or "").strip()
        if not answer:
            return (
                "The bill does not provide enough readable "
                "information to answer this question."
            )
        return answer
    except Exception as e:
        logger.error("QA failed: %s", e)
        return (
            "The bill does not provide enough readable "
            "information to answer this question."
        )


def answer_all(
    questions: list[dict],
    bill_data: dict,
    ocr_text: str = "",
    llm: LLMClient | None = None,
) -> list[str]:
    if llm is None:
        llm = LLMClient()
    answers = []
    for q in questions:
        ans = answer_question(
            q["question"], bill_data, ocr_text=ocr_text, llm=llm
        )
        answers.append(ans)
    return answers