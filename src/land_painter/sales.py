"""Sale flags.

Each sale is unflagged, known good, or disqualified. Disqualified sales are
excluded from all metrics and evidence; known good and unflagged sales are
used (with equal weight, for now). Flags are indexed by row of
``Dataset.sales``, whose order is fixed by the input file and import config.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

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


@dataclass(frozen=True)
class SalesInScope:
    """The per-sale arrays metrics share, for the usable sales within a parcel mask.

    ``ok`` is the base filter (usable, in the mask, positive price); callers
    narrow it further (vacant vs. improved, a land value, a cost, controls).
    """

    parcel: np.ndarray   # parcel index of each sale
    price: np.ndarray    # price of each sale (float, NaN where missing)
    vacant: np.ndarray   # True for a vacant sale
    ok: np.ndarray       # base mask: usable & in the parcel mask & price > 0

    @property
    def vacant_sales(self) -> np.ndarray:
        return np.flatnonzero(self.ok & self.vacant)

    @property
    def improved_sales(self) -> np.ndarray:
        return np.flatnonzero(self.ok & ~self.vacant)


def sales_in_scope(flags: np.ndarray, sales: pl.DataFrame,
                   parcel_mask: np.ndarray | None = None) -> SalesInScope:
    """Extract the shared per-sale arrays for sales in ``parcel_mask`` (None = county-wide).

    ``usable`` already drops sales with a null vacant/improved value, so filling
    the missing ``vacant`` with False never affects a sale that passes ``ok``.
    """
    parcel = sales["parcel"].to_numpy()
    price = sales["price"].fill_null(np.nan).to_numpy().astype(float)
    vacant = sales["vacant"].fill_null(False).to_numpy()
    with np.errstate(invalid="ignore"):
        ok = usable(flags, sales) & (price > 0)
    if parcel_mask is not None:
        ok = ok & parcel_mask[parcel]
    return SalesInScope(parcel, price, vacant, ok)
