"""Shared colour palette (One Dark), so the panels agree on good/bad/muted.

Panels import the semantic names (``GOOD``/``BAD``/``MUTED``) or the raw palette
(``GREEN``/``AMBER``/``RED``/``GREY``) rather than re-declaring hex codes.
"""

from __future__ import annotations

import numpy as np

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


# --- Sale-ratio overlay -----------------------------------------------------------------
# A diverging ramp (ColorBrewer RdBu endpoints) for colouring sale points by how far
# their land ratio sits from the local median: blue = land value below the market
# (under-valued), red = above (over-valued), near-white at the median. The spread is
# measured in log space because ratios are multiplicative.

RATIO_UNDER = (33, 102, 172)      # blue: land value below the local market
RATIO_MID = (247, 247, 247)       # near-white: at the median ratio
RATIO_OVER = (178, 24, 43)        # red: land value above the local market
RATIO_NODATA = (150, 150, 150, 120)  # translucent grey: no usable ratio
RATIO_CLAMP_PERCENTILE = 90       # |log deviation| at this percentile maps to full saturation


def ratio_colors(ratios: np.ndarray) -> tuple[bytes, float | None, int]:
    """RGBA bytes colouring each ratio by its log-deviation from the median.

    Returns ``(rgba_bytes, center, n_valued)``. ``center`` is the median of the
    usable ratios (None when there are none). Colours diverge blue↔red around it,
    clamped so the ``RATIO_CLAMP_PERCENTILE``-th percentile of the absolute log
    deviation reaches full saturation. Undefined or non-positive ratios get
    ``RATIO_NODATA``.
    """
    ratios = np.asarray(ratios, dtype=float)
    rgba = np.empty((len(ratios), 4), dtype=np.uint8)
    rgba[:] = RATIO_NODATA
    valued = ~np.isnan(ratios) & (ratios > 0)
    k = int(valued.sum())
    if not k:
        return rgba.tobytes(), None, 0
    center = float(np.median(ratios[valued]))
    deviation = np.log(ratios[valued] / center)
    scale = float(np.percentile(np.abs(deviation), RATIO_CLAMP_PERCENTILE)) if k > 1 else 0.0
    t = np.clip(deviation / scale, -1.0, 1.0) if scale > 0 else np.zeros_like(deviation)
    mid, under, over = (np.asarray(c, dtype=float) for c in (RATIO_MID, RATIO_UNDER, RATIO_OVER))
    tt = t[:, None]
    rgb = np.where(tt < 0, mid + (under - mid) * -tt, mid + (over - mid) * tt)
    block = np.empty((k, 4), dtype=np.uint8)
    block[:, :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    block[:, 3] = 255
    rgba[valued] = block
    return rgba.tobytes(), center, k


def css_hex(rgb: tuple[int, int, int]) -> str:
    """``(r, g, b)`` as a CSS hex string, for stylesheet gradients."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)
