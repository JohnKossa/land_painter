"""Metrics, tests, and evidence for land tables.

A statistic is ``None`` ("Undefined" in the UI) only when it cannot be
computed for lack of sales (or a zero divisor that follows from it).

Sales used: not disqualified, with a positive price and a known vacant /
improved value. Allocation ratios use the depreciated cost of improvements.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data.config import DEPRECIATED_COST
from .land_table import LOT
from .sales import usable


# --- Statistics -----------------------------------------------------------------------


@dataclass
class RatioStudy:
    n: int
    median: float | None
    cod: float | None
    prd: float | None
    prb: float | None


def cod(values: np.ndarray) -> float | None:
    """Coefficient of dispersion about the median, in percent."""
    if len(values) == 0:
        return None
    median = np.median(values)
    if median == 0:
        return None
    return float(100 * np.mean(np.abs(values - median)) / median)


def cov(values: np.ndarray) -> float | None:
    """Coefficient of variation (sample standard deviation / mean), in percent."""
    if len(values) < 2:
        return None
    mean = np.mean(values)
    if mean == 0:
        return None
    return float(100 * np.std(values, ddof=1) / mean)


def ratio_study(assessed: np.ndarray, price: np.ndarray) -> RatioStudy:
    """IAAO ratio statistics for assessed (land) values against sale prices."""
    n = len(price)
    if n == 0:
        return RatioStudy(0, None, None, None, None)
    ratios = assessed / price
    median = float(np.median(ratios))
    total_assessed = assessed.sum()
    prd = float(np.mean(ratios) / (total_assessed / price.sum())) if total_assessed != 0 else None
    prb = None
    if n >= 2 and median != 0:
        proxy = 0.5 * price + 0.5 * assessed / median
        if np.all(proxy > 0):
            x = np.log2(proxy)
            if np.ptp(x) > 0:
                y = (ratios - median) / median
                prb = float(np.polyfit(x, y, 1)[0])
    return RatioStudy(n, median, cod(ratios), prd, prb)


@dataclass
class Range:
    n: int
    minimum: float
    mean: float
    maximum: float


def value_range(values: np.ndarray) -> Range | None:
    if len(values) == 0:
        return None
    return Range(len(values), float(values.min()), float(values.mean()), float(values.max()))


def _median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if len(values) else None


# --- Neighborhood metrics ------------------------------------------------------------------


@dataclass
class UnitEvidence:
    unit: str
    vacant: Range | None  # sale price / units, usable vacant sales
    residual: Range | None  # (sale price − depreciated cost) / units, usable improved sales
    cov: float | None  # COV of the vacant sale prices per unit


@dataclass
class Metrics:
    n_parcels: int
    n_vacant: int  # usable vacant sales in scope
    n_improved: int  # usable improved sales in scope
    ratio: RatioStudy  # vacant sales with a land value
    spread_unit: str | None  # unit used for the land value spread
    spread_cod: float | None  # COD of land value per unit across parcels
    allocation_table: float | None  # median land allocation of built parcels, land from the land table
    allocation_vacant: float | None  # same, land from vacant sales' mean price per unit
    allocation_sales: float | None  # median (price − cost) / price of improved sales
    over_price: int  # usable sales whose land value exceeds the sale price
    over_price_checked: int  # usable sales with a land value
    evidence: list[UnitEvidence] = field(default_factory=list)
    best_unit: str | None = None  # lowest COV of vacant sale prices per unit

    @property
    def has_cost(self) -> bool:
        return self.allocation_table is not None or self.allocation_sales is not None


def compute(session, parcel_mask: np.ndarray, unit: str | None = None, land: np.ndarray | None = None) -> Metrics:
    """Metrics for the parcels in ``parcel_mask`` (and the sales on them).

    ``unit`` is the land table's unit of measure, used for the land value
    spread and the vacant-sales allocation; None picks the best unit.
    ``land`` overrides the session's land values (e.g. to preview a draft).
    """
    dataset = session.dataset
    land = session.land_values if land is None else land
    has_cost = DEPRECIATED_COST in dataset.parcels.columns
    cost = dataset.parcels[DEPRECIATED_COST].cast(float).fill_null(np.nan).to_numpy() if has_cost else None

    # Sales in scope
    sales = dataset.sales
    if sales is not None:
        sale_parcel = sales["parcel"].to_numpy()
        price = sales["price"].fill_null(np.nan).to_numpy().astype(float)
        ok = usable(session.sale_flags, sales) & parcel_mask[sale_parcel] & (price > 0)
        vacant = sales["vacant"].fill_null(False).to_numpy()
        vacant_sales, improved_sales = np.flatnonzero(ok & vacant), np.flatnonzero(ok & ~vacant)
    else:
        sale_parcel = price = None
        vacant_sales = improved_sales = np.zeros(0, dtype=np.int64)

    # Evidence per unit, and the best unit by COV
    evidence = []
    for name in dataset.unit_names():
        units = dataset.unit_values(name)
        v_units = units[sale_parcel[vacant_sales]] if len(vacant_sales) else np.zeros(0)
        v_ok = v_units > 0
        per_unit = price[vacant_sales][v_ok] / v_units[v_ok] if len(vacant_sales) else np.zeros(0)
        residual = None
        if has_cost and len(improved_sales):
            i_units = units[sale_parcel[improved_sales]]
            i_cost = cost[sale_parcel[improved_sales]]
            i_ok = (i_units > 0) & ~np.isnan(i_cost)
            residual = value_range((price[improved_sales][i_ok] - i_cost[i_ok]) / i_units[i_ok])
        evidence.append(UnitEvidence(name, value_range(per_unit), residual, cov(per_unit)))
    scored = [e for e in evidence if e.cov is not None]
    best_unit = min(scored, key=lambda e: e.cov).unit if scored else None
    unit = unit or best_unit or LOT

    # Ratio study: vacant sales whose parcel has a land value
    if len(vacant_sales):
        v_land = land[sale_parcel[vacant_sales]]
        valued = ~np.isnan(v_land)
        ratio = ratio_study(v_land[valued], price[vacant_sales][valued])
    else:
        ratio = ratio_study(np.zeros(0), np.zeros(0))

    # Spread: COD of land value per unit across parcels
    parcels = np.flatnonzero(parcel_mask)
    units = dataset.unit_values(unit)[parcels]
    p_land = land[parcels]
    spread_ok = ~np.isnan(p_land) & (units > 0)
    spread_cod = cod(p_land[spread_ok] / units[spread_ok])

    # Allocation ratios
    allocation_table = allocation_vacant = allocation_sales = None
    if has_cost:
        p_cost = cost[parcels]
        built = p_cost > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            table_ok = built & ~np.isnan(p_land)
            allocation_table = _median(p_land[table_ok] / (p_land[table_ok] + p_cost[table_ok]))
            vacant_units = dataset.unit_values(unit)[sale_parcel[vacant_sales]] if len(vacant_sales) else np.zeros(0)
            rate_ok = vacant_units > 0
            if rate_ok.any():
                rate = np.mean(price[vacant_sales][rate_ok] / vacant_units[rate_ok])
                implied = rate * units
                implied_ok = built & ~np.isnan(implied)
                allocation_vacant = _median(implied[implied_ok] / (implied[implied_ok] + p_cost[implied_ok]))
            if len(improved_sales):
                s_cost = cost[sale_parcel[improved_sales]]
                s_ok = s_cost > 0
                allocation_sales = _median(1 - s_cost[s_ok] / price[improved_sales][s_ok])

    # Land value in excess of sale price
    over = checked = 0
    if sales is not None:
        all_sales = np.concatenate([vacant_sales, improved_sales])
        s_land = land[sale_parcel[all_sales]]
        valued = ~np.isnan(s_land)
        checked = int(valued.sum())
        over = int((s_land[valued] > price[all_sales][valued]).sum())

    return Metrics(len(parcels), len(vacant_sales), len(improved_sales), ratio, unit, spread_cod,
                   allocation_table, allocation_vacant, allocation_sales, over, checked, evidence, best_unit)


def per_unit_land_evidence(session, parcel_mask: np.ndarray, unit: str) -> np.ndarray:
    """Per-unit land-value observations from sales on the masked parcels.

    Vacant sales contribute ``price / units``; improved sales contribute the land
    residual ``(price − depreciated cost) / units``. Used to calibrate a
    multiplier to market evidence. Only usable sales with a positive price and
    positive units count; improved sales also need a depreciated cost.
    """
    dataset = session.dataset
    sales = dataset.sales
    if sales is None:
        return np.zeros(0)
    sale_parcel = sales["parcel"].to_numpy()
    price = sales["price"].fill_null(np.nan).to_numpy().astype(float)
    ok = usable(session.sale_flags, sales) & parcel_mask[sale_parcel] & (price > 0)
    vacant = sales["vacant"].fill_null(False).to_numpy()
    units_all = dataset.unit_values(unit)
    samples = []
    v = np.flatnonzero(ok & vacant)
    if len(v):
        u = units_all[sale_parcel[v]]
        m = u > 0
        samples.append(price[v][m] / u[m])
    if DEPRECIATED_COST in dataset.parcels.columns:
        i = np.flatnonzero(ok & ~vacant)
        if len(i):
            cost = dataset.parcels[DEPRECIATED_COST].cast(float).fill_null(np.nan).to_numpy()
            u = units_all[sale_parcel[i]]
            c = cost[sale_parcel[i]]
            m = (u > 0) & ~np.isnan(c)
            samples.append((price[i][m] - c[m]) / u[m])
    return np.concatenate(samples) if samples else np.zeros(0)


def over_price_sales(session, parcel_mask: np.ndarray) -> np.ndarray:
    """Indices of usable sales in scope whose land value exceeds the sale price."""
    sales = session.dataset.sales
    if sales is None:
        return np.zeros(0, dtype=np.int64)
    sale_parcel = sales["parcel"].to_numpy()
    price = sales["price"].fill_null(np.nan).to_numpy().astype(float)
    land = session.land_values[sale_parcel]
    with np.errstate(invalid="ignore"):
        hit = usable(session.sale_flags, sales) & parcel_mask[sale_parcel] & (price > 0) & (land > price)
    return np.flatnonzero(hit)
