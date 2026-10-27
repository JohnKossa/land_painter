"""Sale flags.

Each sale is unflagged, known good, or disqualified. Disqualified sales are
excluded from all metrics and evidence; known good and unflagged sales are
used (with equal weight, for now). Flags are indexed by row of
``Dataset.sales``, whose order is fixed by the input file and import config.
"""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl

UNFLAGGED, KNOWN_GOOD, DISQUALIFIED = 0, 1, 2
FLAG_LABELS = {UNFLAGGED: "Unflagged", KNOWN_GOOD: "Known good", DISQUALIFIED: "Disqualified"}


def sales_fingerprint(sales: pl.DataFrame | None) -> str:
    """Identifies the sales rows, so flags are never re-applied to different sales."""
    if sales is None:
        return "none"
    rows = sales.select(pl.concat_str([pl.col("key"), pl.col("price"), pl.col("date")],
                                      separator="|", ignore_nulls=True)).to_series()
    digest = hashlib.sha256("\n".join(rows.to_list()).encode())
    return f"{sales.height}:{digest.hexdigest()}"


def usable(flags: np.ndarray, sales: pl.DataFrame) -> np.ndarray:
    """Sales that metrics may use: not disqualified, with a price and a vacant/improved value."""
    return (flags != DISQUALIFIED) & sales["price"].is_not_null().to_numpy() & sales["vacant"].is_not_null().to_numpy()
