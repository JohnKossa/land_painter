"""Metrics, tests, and evidence for land tables.

A statistic is ``None`` ("Undefined" in the UI) only when it cannot be
computed for lack of sales (or a zero divisor that follows from it).

Sales used: not disqualified, with a positive price and a known vacant /
improved value. Allocation ratios use the depreciated cost of improvements.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data.config import DEPRECIATED_COST, ROLL_LAND
from .land_table import LOT
from .sales import sales_in_scope

# --- Statistics -----------------------------------------------------------------------


# Horizontal uniformity (CHD), after the Center for Land Economics' "Mass Appraisal for
# the Masses": no official standard, so these are its rules of thumb.
CHD_TARGET = 15.0  # median CHD at or below this is good
CHD_MIN_PARCELS = 15  # fewer parcels make a CHD unreliable


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


# --- Unit of measure scoring --------------------------------------------------------------
# Spread is measured on log(value per unit): scale-free like COV (its SD is
# roughly the COV for small spreads) and tolerant of the usual right skew.
# Non-positive cost residuals have no log and are left out.

MIN_SALES = 5  # per group (vacant / improved) for that group's spread to count
NOISE_FLOOR = 1e-6  # smallest log variance used as a group's noise, so weights stay finite


def log_variance(values: np.ndarray) -> float | None:
    """Sample variance of log(values); None with fewer than MIN_SALES values."""
    if len(values) < MIN_SALES:
        return None
    return float(np.var(np.log(values), ddof=1))


def as_percent(variance: float | None) -> float | None:
    """A log variance as its SD in percent, comparable to a COV."""
    return None if variance is None else float(100 * np.sqrt(variance))


def weighted_log_spread(groups: list[tuple[np.ndarray, float]]) -> float:
    """Weighted SD (in percent) of log(values) around one shared weighted mean.

    ``groups`` is (values, weight per value). Weights are reliability weights,
    so the variance uses the matching small-sample correction. Because every
    group is measured against the same mean, a level gap between groups adds
    to the spread.
    """
    x = np.concatenate([np.log(values) for values, _ in groups])
    w = np.concatenate([np.full(len(values), weight) for values, weight in groups])
    total = w.sum()
    mean = (w * x).sum() / total
    variance = (w * (x - mean) ** 2).sum() / (total - (w ** 2).sum() / total)
    return float(100 * np.sqrt(variance))


# --- Neighborhood metrics ------------------------------------------------------------------


@dataclass
class UnitEvidence:
    unit: str
    vacant: Range | None  # sale price / units, usable vacant sales
    residual: Range | None  # (sale price − depreciated cost) / units, usable improved sales
    vacant_spread: float | None = None  # SD of log(vacant price per unit), percent; None below MIN_SALES
    residual_spread: float | None = None  # same for positive residuals per unit
    residuals_dropped: int = 0  # improved sales with a residual ≤ 0 (no log), left out of the spreads
    score: float | None = None  # precision-weighted spread of both groups, percent; lowest is best
    roll: Range | None = None  # previous roll land value / units across the parcels; a reference, not scored


# Previous roll comparison. A parcel whose land value moved more than this
# share either way counts as a big mover. One constant for now; meant to
# become a user setting.
MOVE_THRESHOLD = 0.25


@dataclass
class RollComparison:
    """This neighborhood's land values against the previous roll's (when roll land value is mapped)."""
    n_compared: int  # parcels with both a positive roll value and a new land value
    n_no_roll: int  # parcels with a new land value but no positive roll value
    n_no_value: int  # parcels with a positive roll value but no new land value
    median_change: float | None  # median of new / roll − 1
    q25: float | None
    q75: float | None
    threshold: float
    n_up: int  # change > +threshold
    n_down: int  # change < −threshold
    roll_ratio: RatioStudy  # the roll's land values against the same vacant sales


@dataclass
class UnitWeights:
    """Per-sale weights for the unit score, fixed per neighborhood (the same for every unit).

    A group's noise is its lowest log variance across the candidate units: the
    spread left even under the best-fitting unit. Each sale is weighted by the
    inverse of its group's noise. None means the group has too few sales.
    """
    vacant_noise: float | None = None  # as SD in percent
    improved_noise: float | None = None
    vacant_weight: float | None = None  # per sale
    improved_weight: float | None = None


@dataclass
class Metrics:
    n_parcels: int
    n_vacant: int  # usable vacant sales in scope
    n_improved: int  # usable improved sales in scope
    ratio: RatioStudy  # vacant sales with a land value
    spread_unit: str | None  # unit used for the CHD (and the vacant-sales allocation)
    chd: float | None  # coefficient of horizontal dispersion: COD of land value per unit across parcels
    allocation_table: float | None  # median land allocation of built parcels, land from the land table
    allocation_vacant: float | None  # same, land from vacant sales' mean price per unit
    allocation_sales: float | None  # median (price − cost) / price of improved sales
    over_price: int  # usable sales whose land value exceeds the sale price
    over_price_checked: int  # usable sales with a land value
    evidence: list[UnitEvidence] = field(default_factory=list)
    best_unit: str | None = None  # lowest precision-weighted spread (UnitEvidence.score)
    weights: UnitWeights = field(default_factory=UnitWeights)
    chd_n: int = 0  # parcels in the CHD (a land value and positive units)
    roll: RollComparison | None = None  # None when roll land value isn't mapped

    @property
    def has_cost(self) -> bool:
        return self.allocation_table is not None or self.allocation_sales is not None


def _unit_evidence(dataset, sale_parcel, price, cost, vacant_sales: np.ndarray,
                   improved_sales: np.ndarray) -> tuple[list[UnitEvidence], str | None, UnitWeights]:
    """Evidence for every unit, the best unit, and the weights used to pick it.

    Scoring: per group, the noise is the lowest log variance across units
    (fixed per neighborhood, so all units are scored with the same weights);
    each sale is weighted by 1 / its group's noise; the score is the weighted
    spread of log(value per unit) of both groups around one shared mean. A
    unit is scored only if every group with a noise estimate has at least
    MIN_SALES usable values under it, so the scores stay comparable.
    """
    rows = []  # (name, vacant per unit, all residuals per unit, positive residuals per unit)
    for name in dataset.unit_names():
        units = dataset.unit_values(name)
        v_units = units[sale_parcel[vacant_sales]] if len(vacant_sales) else np.zeros(0)
        v_ok = v_units > 0
        per_unit = price[vacant_sales][v_ok] / v_units[v_ok] if len(vacant_sales) else np.zeros(0)
        residual = np.zeros(0)
        if cost is not None and len(improved_sales):
            i_units = units[sale_parcel[improved_sales]]
            i_cost = cost[sale_parcel[improved_sales]]
            i_ok = (i_units > 0) & ~np.isnan(i_cost)
            residual = (price[improved_sales][i_ok] - i_cost[i_ok]) / i_units[i_ok]
        rows.append((name, per_unit, residual, residual[residual > 0]))

    v_vars = [log_variance(v) for _, v, _, _ in rows]
    r_vars = [log_variance(r) for _, _, _, r in rows]
    known_v, known_r = [v for v in v_vars if v is not None], [v for v in r_vars if v is not None]
    v_noise = max(min(known_v), NOISE_FLOOR) if known_v else None
    r_noise = max(min(known_r), NOISE_FLOOR) if known_r else None
    weights = UnitWeights(as_percent(v_noise), as_percent(r_noise),
                          1 / v_noise if v_noise else None, 1 / r_noise if r_noise else None)

    evidence = []
    for (name, per_unit, residual, positive), v_var, r_var in zip(rows, v_vars, r_vars):
        groups = []
        complete = v_noise is not None or r_noise is not None
        if v_noise is not None:
            complete &= v_var is not None
            groups.append((per_unit, weights.vacant_weight))
        if r_noise is not None:
            complete &= r_var is not None
            groups.append((positive, weights.improved_weight))
        score = weighted_log_spread(groups) if complete else None
        evidence.append(UnitEvidence(
            name, value_range(per_unit), value_range(residual) if cost is not None else None,
            as_percent(v_var), as_percent(r_var), len(residual) - len(positive), score))
    scored = [e for e in evidence if e.score is not None]
    best_unit = min(scored, key=lambda e: e.score).unit if scored else None
    return evidence, best_unit, weights


def roll_land_values(dataset) -> np.ndarray | None:
    """The previous roll's land value per parcel (NaN where missing), or None if not mapped."""
    if ROLL_LAND not in dataset.parcels.columns:
        return None
    return dataset.float_column(ROLL_LAND)


def roll_comparison(roll: np.ndarray, land: np.ndarray, parcels: np.ndarray, vacant_land_roll: np.ndarray,
                    vacant_price: np.ndarray, threshold: float = MOVE_THRESHOLD) -> RollComparison:
    """Compare new land values with the roll's on ``parcels``; the roll's ratio study on the given sales."""
    old, new = roll[parcels], land[parcels]
    with np.errstate(invalid="ignore"):
        has_old, has_new = old > 0, ~np.isnan(new)
        both = has_old & has_new  # a new value of 0 is a real −100% move
    change = new[both] / old[both] - 1
    q25, median, q75 = (float(q) for q in np.percentile(change, [25, 50, 75])) if len(change) else (None,) * 3
    with np.errstate(invalid="ignore"):
        priced = vacant_land_roll > 0
    return RollComparison(int(both.sum()), int((has_new & ~has_old).sum()), int((has_old & ~has_new).sum()),
                          median, q25, q75, threshold, int((change > threshold).sum()),
                          int((change < -threshold).sum()),
                          ratio_study(vacant_land_roll[priced], vacant_price[priced]))


def _sales_arrays(session, parcel_mask: np.ndarray):
    """(sale parcel, price, vacant-sale indices, improved-sale indices) in scope.

    The arrays are None / empty when the session has no sales, so every metric
    block can treat "no sales" uniformly.
    """
    sales = session.dataset.sales
    if sales is None:
        empty = np.zeros(0, dtype=np.int64)
        return None, None, empty, empty
    scope = sales_in_scope(session.sale_flags, sales, parcel_mask)
    return scope.parcel, scope.price, scope.vacant_sales, scope.improved_sales


def _vacant_ratio_study(land, sale_parcel, price, vacant_sales) -> RatioStudy:
    """Ratio study over the vacant sales whose parcel has a land value."""
    if not len(vacant_sales):
        return ratio_study(np.zeros(0), np.zeros(0))
    v_land = land[sale_parcel[vacant_sales]]
    valued = ~np.isnan(v_land)
    return ratio_study(v_land[valued], price[vacant_sales][valued])


def _chd(dataset, unit, parcels, land) -> tuple[float | None, int]:
    """Horizontal uniformity: (COD of land value per unit across parcels, parcel count)."""
    units = dataset.unit_values(unit)[parcels]
    p_land = land[parcels]
    ok = ~np.isnan(p_land) & (units > 0)
    return cod(p_land[ok] / units[ok]), int(ok.sum())


def _allocations(dataset, unit, parcels, land, cost, sale_parcel, price,
                 vacant_sales, improved_sales) -> tuple[float | None, float | None, float | None]:
    """Median land allocation of built parcels, from the land table, vacant sales, and improved sales."""
    if cost is None:
        return None, None, None
    p_land, p_cost = land[parcels], cost[parcels]
    units = dataset.unit_values(unit)[parcels]
    built = p_cost > 0
    allocation_vacant = allocation_sales = None
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
    return allocation_table, allocation_vacant, allocation_sales


def _over_price(land, sale_parcel, price, vacant_sales, improved_sales) -> tuple[int, int]:
    """(sales whose land value exceeds their price, sales with a land value) over all usable sales."""
    all_sales = np.concatenate([vacant_sales, improved_sales])
    if not len(all_sales):
        return 0, 0
    s_land = land[sale_parcel[all_sales]]
    valued = ~np.isnan(s_land)
    return int((s_land[valued] > price[all_sales][valued]).sum()), int(valued.sum())


def _roll_evidence_and_comparison(dataset, evidence, parcels, land, sale_parcel, price, vacant_sales,
                                  move_threshold) -> RollComparison | None:
    """Fill each evidence row's roll range and return the roll comparison; None if roll land isn't mapped."""
    roll = roll_land_values(dataset)
    if roll is None:
        return None
    p_roll = roll[parcels]
    for e in evidence:
        e_units = dataset.unit_values(e.unit)[parcels]
        with np.errstate(invalid="ignore"):
            ok = (p_roll > 0) & (e_units > 0)
        e.roll = value_range(p_roll[ok] / e_units[ok])
    v_parcels = sale_parcel[vacant_sales] if len(vacant_sales) else np.zeros(0, dtype=np.int64)
    v_price = price[vacant_sales] if len(vacant_sales) else np.zeros(0)
    return roll_comparison(roll, land, parcels, roll[v_parcels], v_price, move_threshold)


def compute(session, parcel_mask: np.ndarray, unit: str | None = None, land: np.ndarray | None = None,
            move_threshold: float = MOVE_THRESHOLD) -> Metrics:
    """Metrics for the parcels in ``parcel_mask`` (and the sales on them).

    ``unit`` is the land table's unit of measure, used for the land value
    spread and the vacant-sales allocation; None picks the best unit.
    ``land`` overrides the session's land values (e.g. to preview a draft).
    ``move_threshold`` is the change from the previous roll that counts as a big mover.
    """
    dataset = session.dataset
    land = session.land_values if land is None else land
    cost = dataset.float_column(DEPRECIATED_COST) if DEPRECIATED_COST in dataset.parcels.columns else None
    parcels = np.flatnonzero(parcel_mask)
    sale_parcel, price, vacant_sales, improved_sales = _sales_arrays(session, parcel_mask)

    evidence, best_unit, weights = _unit_evidence(dataset, sale_parcel, price, cost, vacant_sales, improved_sales)
    unit = unit or best_unit or LOT

    ratio = _vacant_ratio_study(land, sale_parcel, price, vacant_sales)
    chd, chd_n = _chd(dataset, unit, parcels, land)
    allocation_table, allocation_vacant, allocation_sales = _allocations(
        dataset, unit, parcels, land, cost, sale_parcel, price, vacant_sales, improved_sales)
    over, checked = _over_price(land, sale_parcel, price, vacant_sales, improved_sales)
    comparison = _roll_evidence_and_comparison(
        dataset, evidence, parcels, land, sale_parcel, price, vacant_sales, move_threshold)

    return Metrics(len(parcels), len(vacant_sales), len(improved_sales), ratio, unit, chd,
                   allocation_table, allocation_vacant, allocation_sales, over, checked, evidence, best_unit,
                   weights, chd_n, comparison)


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
    scope = sales_in_scope(session.sale_flags, sales, parcel_mask)
    sale_parcel, price = scope.parcel, scope.price
    units_all = dataset.unit_values(unit)
    samples = []
    v = scope.vacant_sales
    if len(v):
        u = units_all[sale_parcel[v]]
        m = u > 0
        samples.append(price[v][m] / u[m])
    if DEPRECIATED_COST in dataset.parcels.columns:
        i = scope.improved_sales
        if len(i):
            cost = dataset.float_column(DEPRECIATED_COST)
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
    scope = sales_in_scope(session.sale_flags, sales, parcel_mask)
    land = session.land_values[scope.parcel]
    with np.errstate(invalid="ignore"):
        hit = scope.ok & (land > scope.price)
    return np.flatnonzero(hit)
