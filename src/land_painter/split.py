"""Land/building split check on the current roll (tests 5a and 5b).

5a. Building ÷ location-free cost. For each improved parcel,
    ratio_b = roll building value ÷ depreciated cost. A building's cost
    shouldn't depend on location, so how much neighborhood explains the
    variance of ln(ratio_b) measures location value booked as building.

5b. Leakage comparison. The same decomposition for ln(land value per unit),
    on the same parcels. Neighborhood explaining building about as well as
    land (or better) means location is split across both components.

Variance is decomposed by one-way ANOVA over groups (neighborhoods):

    η²  = SS_between ÷ SS_total (biased upward when groups are many and small)
    ω²  = (SS_between − (k − 1)·MS_within) ÷ (SS_total + MS_within), the adjusted effect size
    ICC = (MS_between − MS_within) ÷ (MS_between + (n₀ − 1)·MS_within), one-way random effects,
          n₀ = (N − Σnᵢ²/N) ÷ (k − 1)

Groups with fewer than MIN_GROUP parcels are left out, since they inflate η².

Only meaningful when depreciated cost is location-free (cost book only); if it
already carries a neighborhood or market factor, ratio_b sits near 1 and 5a
finds nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data.config import DEPRECIATED_COST, ROLL_BUILDING, ROLL_LAND

MIN_GROUP = 5  # parcels per neighborhood to be included

# Cohen's conventions for η²-type effect sizes, used for highlighting
SMALL, MEDIUM, LARGE = 0.01, 0.06, 0.14


@dataclass
class Variance:
    n: int  # parcels
    k: int  # groups
    eta2: float
    omega2: float
    icc: float


def decompose(y: np.ndarray, groups: np.ndarray) -> Variance | None:
    """One-way ANOVA of ``y`` by integer ``groups``; None with under 2 groups or no within-group freedom."""
    _, g = np.unique(groups, return_inverse=True)
    k, n = int(g.max()) + 1 if len(g) else 0, len(y)
    if k < 2 or n - k < 1:
        return None
    counts = np.bincount(g, minlength=k)
    means = np.bincount(g, y, k) / counts
    grand = y.mean()
    ss_total = float(((y - grand) ** 2).sum())
    if ss_total == 0:
        return None
    ss_between = float((counts * (means - grand) ** 2).sum())
    ms_between = ss_between / (k - 1)
    ms_within = (ss_total - ss_between) / (n - k)
    n0 = (n - (counts ** 2).sum() / n) / (k - 1)
    icc_denom = ms_between + (n0 - 1) * ms_within
    return Variance(n, k, ss_between / ss_total,
                    (ss_between - (k - 1) * ms_within) / (ss_total + ms_within),
                    (ms_between - ms_within) / icc_denom if icc_denom > 0 else 0.0)


@dataclass
class GroupRow:
    name: str
    n: int
    building_ratio: float  # median roll building ÷ depreciated cost
    roll_land_per_unit: float | None  # median
    app_land_per_unit: float | None  # median


@dataclass
class SplitCheck:
    unit: str
    n_improved: int = 0  # parcels with a positive cost and roll building value
    excluded_groups: int = 0  # groups under MIN_GROUP
    excluded_parcels: int = 0
    median_ratio: float | None = None
    building: Variance | None = None  # 5a
    roll_land: Variance | None = None  # 5b, roll land per unit
    app_land: Variance | None = None  # 5b, the app's land tables per unit (for reference)
    groups: list[GroupRow] = field(default_factory=list)
    problem: str | None = None

    @property
    def leakage(self) -> float | None:
        """Building ω² ÷ roll land ω²: near or above 1 means location is split across both components."""
        if self.building is None or self.roll_land is None or self.roll_land.omega2 <= 0:
            return None
        return self.building.omega2 / self.roll_land.omega2


def split_check(dataset, app_land: np.ndarray, group_ids: np.ndarray, group_names: dict[int, str],
                unit: str) -> SplitCheck:
    """Run 5a and 5b over ``group_ids`` (one integer per parcel, negative = no group).

    Land per unit uses ``unit``. ``app_land`` is the app's land values (NaN where none).
    """
    result = SplitCheck(unit)
    columns = dataset.parcels.columns
    if ROLL_BUILDING not in columns or DEPRECIATED_COST not in columns:
        missing = [name for name, col in (("roll building value", ROLL_BUILDING),
                                          ("depreciated cost", DEPRECIATED_COST)) if col not in columns]
        result.problem = f"Needs {' and '.join(missing)} mapped when the session is created."
        return result

    def column(name: str) -> np.ndarray:
        return dataset.float_column(name)

    building, cost = column(ROLL_BUILDING), column(DEPRECIATED_COST)
    roll_land = column(ROLL_LAND) if ROLL_LAND in columns else None
    units = dataset.unit_values(unit)
    with np.errstate(invalid="ignore"):
        improved = (building > 0) & (cost > 0) & (group_ids >= 0)
    result.n_improved = int(improved.sum())

    # Leave out small groups
    ids, counts = np.unique(group_ids[improved], return_counts=True)
    small = set(ids[counts < MIN_GROUP].tolist())
    result.excluded_groups = len(small)
    keep = improved & ~np.isin(group_ids, list(small))
    result.excluded_parcels = int(improved.sum() - keep.sum())
    if not keep.any():
        result.problem = (f"No neighborhood has {MIN_GROUP}+ improved parcels with a positive roll building "
                          "value and depreciated cost.")
        return result

    ratio = building[keep] / cost[keep]
    groups = group_ids[keep]
    result.median_ratio = float(np.median(ratio))
    result.building = decompose(np.log(ratio), groups)

    def land_per_unit(land: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where((land[keep] > 0) & (units[keep] > 0), land[keep] / units[keep], np.nan)

    roll_pu = land_per_unit(roll_land) if roll_land is not None else None
    app_pu = land_per_unit(app_land)
    for pu, attr in ((roll_pu, "roll_land"), (app_pu, "app_land")):
        if pu is not None:
            ok = np.isfinite(pu)
            if ok.any():
                setattr(result, attr, decompose(np.log(pu[ok]), groups[ok]))

    for gid in np.unique(groups):
        in_g = groups == gid

        def median(values, in_g=in_g):
            if values is None:
                return None
            v = values[in_g]
            v = v[np.isfinite(v)]
            return float(np.median(v)) if len(v) else None

        result.groups.append(GroupRow(group_names.get(int(gid), str(gid)), int(in_g.sum()),
                                      float(np.median(ratio[in_g])), median(roll_pu), median(app_pu)))
    result.groups.sort(key=lambda r: r.building_ratio, reverse=True)
    return result
