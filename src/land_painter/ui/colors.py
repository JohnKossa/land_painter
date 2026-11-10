"""Shared colour palette (One Dark), so the panels agree on good/bad/muted.

Panels import the semantic names (``GOOD``/``BAD``/``MUTED``) or the raw palette
(``GREEN``/``AMBER``/``RED``/``GREY``) rather than re-declaring hex codes.
"""

from __future__ import annotations

# Raw palette
GREEN = "#98c379"
AMBER = "#e5c07b"
RED = "#e06c75"
GREY = "#cccccc"
MUTED = "#888"

# Semantic roles
GOOD = GREEN   # within target / known-good / info
BAD = AMBER    # outside target / warning
ERROR = RED    # error / disqualified
