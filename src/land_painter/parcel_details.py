"""Per-parcel detail view: every imported attribute beside its neighborhood norm.

Built to run down difficult ratios. For one parcel it lists each value next to the
neighborhood's mean (numeric fields) or value distribution (categorical fields), and
flags where the parcel is unusual enough to be worth a land table adjustment.

The neighborhood is the parcel's assigned neighborhood, scoped to the parcels directly
in it (as the metrics panel scopes things). No Qt here: the UI renders the sections.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .data.config import (
    CATEGORICAL,
    CONTINUOUS,
    DEPRECIATED_COST,
    FLAG,
    NEIGHBORHOOD,
    REPLACEMENT_COST,
    ROLL_BUILDING,
    ROLL_LAND,
)
from .format import money, number

Z_FLAG = 2.0        # numeric: |(value - mean) / std| at or above this is flagged as unusual
RARE_SHARE = 0.10   # categorical: a value held by this share or less of the neighborhood is flagged
MIN_FOR_FLAG = 5    # neighborhood parcels needed before anything is flagged

UP, DOWN = "▲", "▼"  # ▲ above the norm, ▼ below it


@dataclass
class DetailRow:
    label: str
    value: str           # this parcel, formatted ("—" when missing)
    comparison: str      # neighborhood summary (mean, or value shares)
    flagged: bool = False
    direction: str = ""  # UP / DOWN / "" — only on flagged rows


@dataclass
class DetailSection:
    title: str
    rows: list[DetailRow] = field(default_factory=list)


@dataclass
class ParcelDetails:
    key: str
    neighborhood: int    # assigned neighborhood index
    n_members: int       # parcels directly in that neighborhood
    sections: list[DetailSection]


def _num(value: float | None) -> str:
    return number(value, blank="—")


def _pct(share: float) -> str:
    if share <= 0:
        return "0%"
    if share < 0.005:
        return "<1%"
    return f"{share * 100:.0f}%"


def _text_array(series: pl.Series) -> np.ndarray:
    """Object array of text with None for missing (matching Dataset.characteristic)."""
    return np.array(series.cast(pl.String).to_list(), dtype=object)


def parcel_details(dataset, land_values: np.ndarray, assignment: np.ndarray, parcel: int, *,
                   z_flag: float = Z_FLAG, rare_share: float = RARE_SHARE,
                   min_for_flag: int = MIN_FOR_FLAG) -> ParcelDetails:
    """Build the detail sections for ``parcel``.

    ``land_values`` is one value per parcel (NaN where none); ``assignment`` is the
    neighborhood index per parcel.
    """
    parcel = int(parcel)
    nbhd = int(assignment[parcel])
    members = np.flatnonzero(assignment == nbhd)
    columns = dataset.parcels.columns

    def col(name: str) -> np.ndarray:
        return dataset.float_column(name)

    def numeric_row(label: str, values: np.ndarray, fmt) -> DetailRow:
        v = float(values[parcel])
        here = values[members]
        here = here[np.isfinite(here)]
        mean = float(here.mean()) if len(here) else None
        row = DetailRow(label, fmt(v if np.isfinite(v) else None),
                        f"{fmt(mean)} avg" if mean is not None else "—")
        if np.isfinite(v) and len(here) >= min_for_flag:
            std = float(here.std())
            if std > 0 and abs(v - mean) / std >= z_flag:
                row.flagged, row.direction = True, (UP if v > mean else DOWN)
        return row

    def categorical_row(label: str, values: np.ndarray, flag_rare: bool = True) -> DetailRow:
        raw = values[parcel]
        pv = None if raw is None or (isinstance(raw, float) and np.isnan(raw)) else str(raw)
        known = [str(x) for x in values[members]
                 if x is not None and not (isinstance(x, float) and np.isnan(x))]
        row = DetailRow(label, pv if pv is not None else "—", "—")
        if not known:
            return row
        uniq, counts = np.unique(np.array(known, dtype=object), return_counts=True)
        total = len(known)
        top = int(np.argmax(counts))
        mode, mode_share = str(uniq[top]), counts[top] / total
        if pv is None:
            row.comparison = f"{mode} {_pct(mode_share)}"
            return row
        share = int(dict(zip(uniq.tolist(), counts.tolist())).get(pv, 0)) / total
        row.comparison = (f"{_pct(share)} {pv} (most common)" if pv == mode
                          else f"{_pct(share)} {pv} · {mode} {_pct(mode_share)}")
        if flag_rare and total >= min_for_flag and share <= rare_share:
            row.flagged, row.direction = True, UP
        return row

    valuation = DetailSection("Valuation")
    valuation.rows.append(numeric_row("Land value", land_values, money))
    for name, label in ((REPLACEMENT_COST, "Replacement cost"), (DEPRECIATED_COST, "Depreciated cost"),
                        (ROLL_BUILDING, "Roll building value"), (ROLL_LAND, "Roll land value")):
        if name in columns:
            valuation.rows.append(numeric_row(label, col(name), money))

    units = DetailSection("Units")
    for name in dataset.config.units:
        units.rows.append(numeric_row(name, dataset.unit_values(name), _num))

    chars = DetailSection("Characteristics")
    if NEIGHBORHOOD in columns:
        chars.rows.append(categorical_row("Neighborhood (imported)", _text_array(dataset.parcels[NEIGHBORHOOD]),
                                          flag_rare=False))
    for key, label, kind in dataset.detail_fields():
        values = dataset.characteristic(key)
        if kind == CONTINUOUS:
            chars.rows.append(numeric_row(label, values, _num))
        else:
            chars.rows.append(categorical_row(label, values, flag_rare=kind in (CATEGORICAL, FLAG)))

    sections = [s for s in (valuation, units, chars) if s.rows]
    return ParcelDetails(str(dataset.parcels["key"][parcel]), nbhd, len(members), sections)
