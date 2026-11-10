"""Shared value formatters for display.

One home for currency / ratio / percent / number rendering so every panel (and
the domain-layer detail rows) agree on precision and on how a missing value is
shown. ``blank`` is what to render for a missing value (None or NaN); it varies
by context, so it is a parameter: table cells use ``""``, inline text uses
``"—"``, metrics use ``"Undefined"``.

This module is pure string formatting with no Qt dependency, so both the UI and
the domain layer can use it. Colours live in ``ui.colors``.
"""

from __future__ import annotations

import math


def _missing(value) -> bool:
    """True for None or NaN (NaN is the only value not equal to itself)."""
    return value is None or value != value


def money(value, *, decimals: int = 0, blank: str = "—") -> str:
    """Dollar amount, e.g. ``$1,250`` or ``−$300``; ``blank`` when missing."""
    if _missing(value):
        return blank
    return f"−${-value:,.{decimals}f}" if value < 0 else f"${value:,.{decimals}f}"


def ratio(value, *, blank: str = "") -> str:
    """Ratio to three decimals; ``blank`` when missing or infinite."""
    if _missing(value) or math.isinf(value):
        return blank
    return f"{value:.3f}"


def number(value, *, blank: str = "") -> str:
    """Number with thousands separators and trailing zeros trimmed."""
    if _missing(value):
        return blank
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def percent(value, *, signed: bool = False, blank: str = "—") -> str:
    """Percentage to one decimal, optionally with an explicit sign."""
    if _missing(value):
        return blank
    return f"{value * 100:+.1f}%" if signed else f"{value * 100:.1f}%"


def span(text: str, color: str) -> str:
    """Wrap ``text`` in a coloured HTML span (colours from ``ui.colors``)."""
    return f"<span style='color:{color}'>{text}</span>"
