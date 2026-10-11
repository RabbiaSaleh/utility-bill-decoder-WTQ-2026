"""
Deterministic calculations for Level-2 answers.

Printed bill values are never changed.
This module only calculates derived facts such as percentages,
differences, averages, and history counts.
"""

from __future__ import annotations

import re
from typing import Any


def format_pkr(value: int | float | None) -> str:
    """Format a number as PKR without unnecessary decimals."""
    if value is None:
        return "PKR amount unavailable"

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "PKR amount unavailable"

    if number.is_integer():
        return f"PKR {number:,.0f}"

    return f"PKR {number:,.2f}"


def build_calculated_facts(
    bill_data: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build derived facts without modifying the original bill data.
    """
    context = context or {}
    facts: dict[str, Any] = {}

    total_taxes = bill_data.get("total_taxes")
    current_bill = bill_data.get("current_bill")

    if (
        isinstance(total_taxes, (int, float))
        and isinstance(current_bill, (int, float))
        and current_bill != 0
    ):
        facts["tax_percentage"] = round(
            (total_taxes / current_bill) * 100,
            1,
        )

    within_due = bill_data.get("payable_within_due_date")
    after_due = bill_data.get("payable_after_due_date")

    if (
        isinstance(within_due, (int, float))
        and isinstance(after_due, (int, float))
    ):
        facts["due_date_difference"] = after_due - within_due

    previous_reading = bill_data.get("previous_reading")
    current_reading = bill_data.get("current_reading")

    if (
        isinstance(previous_reading, (int, float))
        and isinstance(current_reading, (int, float))
    ):
        facts["reading_difference"] = current_reading - previous_reading

    usage_history = context.get("usage_history") or []

    clean_history = []

    for item in usage_history:
        if not isinstance(item, dict):
            continue

        period = item.get("period")
        units = item.get("units")

        if isinstance(units, (int, float)):
            clean_history.append(
                {
                    "period": period,
                    "units": units,
                }
            )

    facts["usage_history"] = clean_history

    if clean_history:
        units = [item["units"] for item in clean_history]

        facts["history_average_units"] = round(
            sum(units) / len(units),
            2,
        )

        facts["history_max_units"] = max(units)
        facts["history_min_units"] = min(units)

    return facts


def count_history_above(
    context: dict[str, Any],
    threshold: float,
) -> int | None:
    """Count usage-history months above a threshold."""
    history = context.get("usage_history") or []

    valid_units = []

    for item in history:
        if not isinstance(item, dict):
            continue

        units = item.get("units")

        if isinstance(units, (int, float)):
            valid_units.append(units)

    if not valid_units:
        return None

    return sum(units > threshold for units in valid_units)