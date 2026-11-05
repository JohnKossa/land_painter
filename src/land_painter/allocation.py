"""Amenity allocation: is an amenity's market premium booked as land or as building?

For one amenity (an adjustment's condition), three log regressions are fitted
on the same improved sales, each with the same structural controls and, when
more than one neighborhood is in scope, neighborhood fixed effects:

    ln(sale price)          → market premium
    ln(land value)          → land premium     (the land tables, draft included)
    ln(depreciated cost)    → building premium (the app's building value)

Each coefficient β becomes a percentage (e^β − 1) and dollars at the median
sale (percentage × the sample's median of that quantity). The building share is
building $ ÷ (land $ + building $), so the land and building shares add to
100%. A building share well above 0 means the cost side is taking location
value that the land table should carry; comparing land $ + building $ with the
market $ shows whether the amenity is over- or under-valued overall.

Fixed effects are fitted by demeaning within neighborhood (equivalent to one
dummy per neighborhood). Controls enter as ln(x) when every value in the
sample is positive, otherwise linearly (e.g. an age that can be 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data.config import DEPRECIATED_COST
from .land_table import Condition
from .sales import usable


@dataclass
class Premium:
    beta: float  # coefficient on the amenity indicator, in log points
    se: float  # conventional OLS standard error (fixed-effect degrees of freedom accounted for)
    median: float  # median of the quantity over the sample
    percent: float  # e^β − 1
    dollars: float  # percent × median

    @property
    def t(self) -> float:
        return self.beta / self.se if self.se > 0 else float("inf")


@dataclass
class AmenityCheck:
    n_sales: int = 0  # improved sales in the regressions
    n_amenity: int = 0  # of which on parcels with the amenity
    n_groups: int = 1  # neighborhoods (fixed effects) among those sales
    controls: list[str] = field(default_factory=list)  # as entered, e.g. "ln(Area)", "Age"
    market: Premium | None = None
    land: Premium | None = None
    building: Premium | None = None  # None without a depreciated cost
    problem: str | None = None  # why nothing could be estimated

    @property
    def assessed_dollars(self) -> float | None:
        if self.land is None:
            return None
        return self.land.dollars + (self.building.dollars if self.building else 0.0)

    @property
    def building_share(self) -> float | None:
        """Building $ ÷ (land $ + building $); None when the two have opposite signs or sum to 0."""
        if self.land is None or self.building is None:
            return None
        land, building = self.land.dollars, self.building.dollars
        if land * building < 0 or land + building == 0:
            return None
        return building / (land + building)


def _demean(x: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """Subtract each group's mean (column-wise for 2-D ``x``)."""
    counts = np.bincount(groups, minlength=n_groups)
    if x.ndim == 1:
        return x - (np.bincount(groups, x, n_groups) / counts)[groups]
    return np.column_stack([_demean(col, groups, n_groups) for col in x.T])


def _fit(y: np.ndarray, X: np.ndarray, groups: np.ndarray, n_groups: int) -> tuple[float, float] | None:
    """OLS of y on X (first column = amenity) after removing group means; (β, SE) of the first column."""
    yd, Xd = _demean(y, groups, n_groups), _demean(X, groups, n_groups)
    coef, _, rank, _ = np.linalg.lstsq(Xd, yd, rcond=None)
    df = len(y) - X.shape[1] - n_groups
    if rank < X.shape[1] or df <= 0:
        return None
    resid = yd - Xd @ coef
    sigma2 = resid @ resid / df
    cov = sigma2 * np.linalg.inv(Xd.T @ Xd)
    return float(coef[0]), float(np.sqrt(max(cov[0, 0], 0.0)))


def _premium(level: np.ndarray, X: np.ndarray, groups: np.ndarray, n_groups: int) -> Premium | None:
    fit = _fit(np.log(level), X, groups, n_groups)
    if fit is None:
        return None
    beta, se = fit
    median = float(np.median(level))
    percent = float(np.expm1(beta))
    return Premium(beta, se, median, percent, percent * median)


def amenity_allocation(session, condition: Condition, parcel_mask: np.ndarray, controls: dict[str, np.ndarray],
                       land: np.ndarray | None = None) -> AmenityCheck:
    """Run the check on the improved sales of the parcels in ``parcel_mask``.

    ``controls`` maps a label to a per-parcel array (NaN where missing).
    ``land`` overrides the session's land values (e.g. a draft table's preview).
    Fixed effects use the parcels' current neighborhoods.
    """
    dataset = session.dataset
    land = session.land_values if land is None else land
    sales = dataset.sales
    if sales is None:
        return AmenityCheck(problem="No sales are loaded.")
    has_cost = DEPRECIATED_COST in dataset.parcels.columns
    cost = dataset.parcels[DEPRECIATED_COST].cast(float).fill_null(np.nan).to_numpy() if has_cost else None

    parcel = sales["parcel"].to_numpy()
    price = sales["price"].fill_null(np.nan).to_numpy().astype(float)
    vacant = sales["vacant"].fill_null(True).to_numpy()
    with np.errstate(invalid="ignore"):
        ok = usable(session.sale_flags, sales) & parcel_mask[parcel] & ~vacant & (price > 0) & (land[parcel] > 0)
        if has_cost:
            ok &= cost[parcel] > 0
    for values in controls.values():
        ok &= np.isfinite(values[parcel])
    chosen = np.flatnonzero(ok)
    p = parcel[chosen]
    amenity = condition.mask(dataset.characteristic(condition.field)[p]).astype(float)

    result = AmenityCheck(len(chosen), int(amenity.sum()))
    if result.n_sales == 0:
        needs = "a land value" + (", a depreciated cost" if has_cost else "") + " and every chosen control"
        result.problem = f"No usable improved sales with {needs}."
        return result
    if result.n_amenity == 0:
        result.problem = "None of the usable improved sales have the amenity."
        return result
    if result.n_amenity == result.n_sales:
        result.problem = "Every usable improved sale has the amenity, so there's nothing to compare against."
        return result

    columns = [amenity]
    for label, values in controls.items():
        x = values[p]
        if np.all(x > 0):
            columns.append(np.log(x))
            result.controls.append(f"ln({label})")
        else:
            columns.append(x)
            result.controls.append(label)
    X = np.column_stack(columns)
    groups_raw = session.neighborhoods.assignment[p]
    _, groups = np.unique(groups_raw, return_inverse=True)
    n_groups = int(groups.max()) + 1
    result.n_groups = n_groups

    result.market = _premium(price[chosen], X, groups, n_groups)
    if result.market is None:
        result.problem = ("The amenity can't be separated from the neighborhoods and controls: it may not vary "
                          "within any neighborhood's sales, or there are too few sales for the number of controls.")
        return result
    result.land = _premium(land[p], X, groups, n_groups)
    if has_cost:
        result.building = _premium(cost[p], X, groups, n_groups)
    return result
