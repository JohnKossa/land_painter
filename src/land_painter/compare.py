"""Comparing an attribute's distribution across neighborhoods.

The user picks neighborhoods, an attribute and a population:

* parcels: each neighborhood's parcels (directly in it, as everywhere else in the app);
* sales: each neighborhood's usable sales (a parcel attribute is read off the sale's parcel);
* sales vs. parcels: both, per neighborhood, to see whether the sales represent the parcels.

Continuous attributes get histograms on shared bins (shares of each group, so
groups of different sizes compare), clipped to the pooled 1st–99th percentile
with off-scale counts kept, optionally on a log scale. Categorical attributes
get shares of the most common categories, the rest folded into "Other".

"Meaningfully different" is reported as effect sizes, not p-values: with
thousands of parcels nearly any difference is significant, so what matters is
how big it is.

* Continuous: Cliff's δ = P(B > A) − P(B < A) (rank-based, so skew and outliers
  don't dominate), the Kolmogorov–Smirnov distance (largest gap between the two
  cumulative distributions), and the change in median.
* Categorical: Cramér's V of the two groups' category mix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data.config import CONTINUOUS, DEPRECIATED_COST, ROLL_BUILDING, ROLL_LAND
from .land_table import LOT
from .sales import sales_in_scope

PARCELS, SALES, BOTH = "parcels", "sales", "sales vs. parcels"
POPULATIONS = (PARCELS, SALES, BOTH)
CLIP = (1.0, 99.0)  # percentiles of the pooled values that set the histogram's range…
FAR_TAIL = 0.5  # …unless the min/max lie within this share of that range beyond it, then they're kept
MIN_BINS, MAX_BINS = 8, 40
MAX_CATEGORIES = 12  # the rest fold into "Other"
OTHER = "Other"

# Number formats for axis labels and tooltips
MONEY, NUMBER, PERCENT, RATIO = "money", "number", "percent", "ratio"

# Magnitude thresholds (lower bounds of small, medium, large)
CLIFF_LEVELS = (0.147, 0.33, 0.474)  # Romano et al. (2006)
CRAMER_LEVELS = (0.1, 0.3, 0.5)  # Cohen, for two groups (df = 1)
MAGNITUDES = ("negligible", "small", "medium", "large")


def magnitude(value: float, levels: tuple[float, float, float]) -> str:
    return MAGNITUDES[int(np.searchsorted(levels, abs(value), side="right"))]


# --- Attributes -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Attribute:
    key: str
    label: str
    categorical: bool
    sale_level: bool  # only defined per sale (e.g. sale price); not available for parcels
    fmt: str = NUMBER
    group: str = ""  # heading in the picker


def attributes(session) -> list[Attribute]:
    """Every attribute the comparison can show, in picker order."""
    dataset = session.dataset
    columns = dataset.parcels.columns
    units = [u for u in dataset.unit_names() if u != LOT]
    out = [Attribute("land", "Land value", False, False, MONEY, "Land values")]
    out += [Attribute(f"land_per:{u}", f"Land value per {u.lower()}", False, False, MONEY, "Land values")
            for u in units]
    if ROLL_LAND in columns:
        out.append(Attribute("roll_change", "Change from previous roll", False, False, PERCENT, "Land values"))
    if dataset.sales is not None:
        out.append(Attribute("sale_ratio", "Sale ratio (land value ÷ price), vacant sales", False, True, RATIO,
                             "Sales"))
        out.append(Attribute("price", "Sale price", False, True, MONEY, "Sales"))
        out += [Attribute(f"price_per:{u}", f"Sale price per {u.lower()}", False, True, MONEY, "Sales")
                for u in units]
    out += [Attribute(f"unit:{u}", u, False, False, NUMBER, "Units of measure") for u in units]
    for column, label in ((DEPRECIATED_COST, "Depreciated cost"), (ROLL_BUILDING, "Roll building value"),
                          (ROLL_LAND, "Roll land value")):
        if column in columns:
            out.append(Attribute(f"col:{column}", label, False, False, MONEY, "Costs and roll"))
    if ROLL_BUILDING in columns and DEPRECIATED_COST in columns:
        out.append(Attribute("bldg_ratio", "Roll building ÷ depreciated cost", False, False, RATIO,
                             "Costs and roll"))
    for key, label, kind in dataset.filter_fields():
        out.append(Attribute(f"char:{key}", label, kind != CONTINUOUS, False, NUMBER, "Characteristics"))
    return out


def _column(dataset, name: str) -> np.ndarray:
    return dataset.float_column(name)


def parcel_values(session, attr: Attribute) -> np.ndarray:
    """Per-parcel values of a parcel-level attribute (NaN / None where missing)."""
    dataset, key = session.dataset, attr.key
    land = session.land_values
    with np.errstate(divide="ignore", invalid="ignore"):
        if key == "land":
            return land.copy()
        if key.startswith("land_per:"):
            units = dataset.unit_values(key.split(":", 1)[1])
            return np.where(units > 0, land / units, np.nan)
        if key == "roll_change":
            roll = _column(dataset, ROLL_LAND)
            return np.where(roll > 0, land / roll - 1, np.nan)
        if key.startswith("unit:"):
            return dataset.unit_values(key.split(":", 1)[1]).copy()
        if key.startswith("col:"):
            return _column(dataset, key.split(":", 1)[1])
        if key == "bldg_ratio":
            building, cost = _column(dataset, ROLL_BUILDING), _column(dataset, DEPRECIATED_COST)
            return np.where((building > 0) & (cost > 0), building / cost, np.nan)
    if key.startswith("char:"):
        return dataset.characteristic(key.split(":", 1)[1])
    raise KeyError(key)


def sale_values(session, attr: Attribute) -> tuple[np.ndarray, np.ndarray]:
    """(sale indices, values) for the usable sales; parcel attributes are read off each sale's parcel."""
    scope = sales_in_scope(session.sale_flags, session.dataset.sales)  # county-wide
    parcel, price = scope.parcel, scope.price
    ok = scope.ok
    if attr.key == "sale_ratio":
        ok = ok & scope.vacant
    index = np.flatnonzero(ok)
    p, pr = parcel[index], price[index]
    if not attr.sale_level:
        return index, parcel_values(session, attr)[p]
    with np.errstate(divide="ignore", invalid="ignore"):
        if attr.key == "price":
            return index, pr
        if attr.key.startswith("price_per:"):
            units = session.dataset.unit_values(attr.key.split(":", 1)[1])[p]
            return index, np.where(units > 0, pr / units, np.nan)
        if attr.key == "sale_ratio":
            return index, session.land_values[p] / pr
    raise KeyError(attr.key)


# --- Groups -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """A set of parcels to compare: one neighborhood, or a combination built from the tree."""
    key: tuple  # identifies it in the picker, e.g. ("hood", 3), ("subtree", 3), ("rest", 3)
    label: str
    parcels: np.ndarray = field(compare=False, hash=False)  # bool mask over parcels


def neighborhood_selection(model, index: int) -> Selection:
    """The parcels directly in a neighborhood (as everywhere else in the app)."""
    return Selection(("hood", index), model.neighborhoods[index].name, model.assignment == index)


def subtree_selection(model, index: int) -> Selection:
    """A neighborhood together with all its sub-neighborhoods."""
    name = model.neighborhoods[index].name
    return Selection(("subtree", index), f"{name} + sub-neighborhoods",
                     model.subtree_mask(index)[model.assignment])


def has_descendants(model, index: int) -> bool:
    return bool(model.children(index))


def rest_of_parent_selection(model, index: int) -> Selection | None:
    """Everything under ``index``'s parent except ``index`` and its sub-neighborhoods; None at the root."""
    parent = model.neighborhoods[index].parent
    if parent is None:
        return None
    in_parent = model.subtree_mask(parent)[model.assignment]
    in_child = model.subtree_mask(index)[model.assignment]
    return Selection(("rest", index), f"Rest of {model.neighborhoods[parent].name} "
                                      f"(without {model.neighborhoods[index].name})", in_parent & ~in_child)


def selection_for(model, key: tuple) -> Selection | None:
    """Rebuild a selection from its key against the current tree; None if it no longer applies."""
    kind, index = key
    if index >= len(model.neighborhoods) or model.neighborhoods[index].deleted:
        return None
    if kind == "hood":
        return neighborhood_selection(model, index)
    if kind == "subtree":
        return subtree_selection(model, index)
    if kind == "rest":
        return rest_of_parent_selection(model, index)
    raise KeyError(kind)


@dataclass
class Group:
    label: str
    key: tuple  # the Selection's key
    population: str  # PARCELS or SALES
    values: np.ndarray  # float (NaN = missing) or object (None = missing)
    parcels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))  # parcel of each value

    @property
    def present_mask(self) -> np.ndarray:
        if self.values.dtype == object:
            return np.array([v is not None for v in self.values], dtype=bool)
        return np.isfinite(self.values)

    @property
    def present(self) -> np.ndarray:
        return self.values[self.present_mask] if len(self.values) else self.values

    @property
    def missing(self) -> int:
        return len(self.values) - len(self.present)


def build_groups(session, attr: Attribute, selections: list[Selection], population: str) -> list[Group]:
    """One group per selection (two in BOTH mode: its parcels, then its sales)."""
    groups = []
    need_parcels = population in (PARCELS, BOTH) and not attr.sale_level
    need_sales = population in (SALES, BOTH) and session.dataset.sales is not None
    p_values = parcel_values(session, attr) if need_parcels else None
    if need_sales:
        s_index, s_values = sale_values(session, attr)
        s_parcel = session.dataset.sales["parcel"].to_numpy()[s_index]
    for s in selections:
        if need_parcels:
            label = s.label + (" — parcels" if population == BOTH else "")
            groups.append(Group(label, s.key, PARCELS, p_values[s.parcels], np.flatnonzero(s.parcels)))
        if need_sales:
            label = s.label + (" — sales" if population == BOTH else "")
            in_s = s.parcels[s_parcel]
            groups.append(Group(label, s.key, SALES, s_values[in_s], s_parcel[in_s]))
    return groups


# --- Summaries ----------------------------------------------------------------------------------


@dataclass
class Summary:
    n: int
    missing: int
    median: float | None = None
    q25: float | None = None
    q75: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    top: str | None = None  # categorical: most common category
    top_share: float | None = None


def summarize(group: Group, categorical: bool) -> Summary:
    x = group.present
    s = Summary(len(x), group.missing)
    if not len(x):
        return s
    if categorical:
        names, counts = np.unique(x.astype(str), return_counts=True)
        i = int(np.argmax(counts))
        s.top, s.top_share = str(names[i]), float(counts[i] / len(x))
    else:
        s.q25, s.median, s.q75 = (float(v) for v in np.percentile(x, [25, 50, 75]))
        s.minimum, s.maximum = float(x.min()), float(x.max())
    return s


# --- Histograms -----------------------------------------------------------------------------------


@dataclass
class Histogram:
    edges: np.ndarray  # bin edges in the attribute's units (log-spaced when log)
    shares: list[np.ndarray]  # per group: share of its present values in each bin
    counts: list[np.ndarray]
    below: list[int]  # per group: values left of the range (incl. non-positive ones on a log scale)
    above: list[int]
    log: bool
    t_edges: np.ndarray = field(default_factory=lambda: np.zeros(0))  # edges in binning space (log10 when log)


def bin_index(t: np.ndarray, t_edges: np.ndarray) -> np.ndarray:
    """Bin of each value in binning space; −1 outside. The last bin includes its right edge.

    Shared by the histogram and map linking, so a bar's count and its parcels always agree.
    """
    n = len(t_edges) - 1
    index = np.searchsorted(t_edges, t, side="right") - 1
    index[t == t_edges[-1]] = n - 1
    index[(t < t_edges[0]) | (t > t_edges[-1])] = -1
    return index


def can_log(groups: list[Group]) -> bool:
    pooled = np.concatenate([g.present for g in groups]) if groups else np.zeros(0)
    return len(pooled) > 0 and bool(np.any(pooled > 0))


def histogram(groups: list[Group], log: bool = False) -> Histogram | None:
    """Shared bins over the pooled values' clipped range; None if no group has a value."""
    present = [g.present.astype(float) for g in groups]
    pooled = np.concatenate(present) if present else np.zeros(0)
    if log:
        pooled = pooled[pooled > 0]
    if not len(pooled):
        return None
    t = np.log10(pooled) if log else pooled
    low, high = (float(v) for v in np.percentile(t, CLIP))
    # Clip a tail only when it's really far out; otherwise show everything
    reach = FAR_TAIL * (high - low)
    if t.min() >= low - reach:
        low = float(t.min())
    if t.max() <= high + reach:
        high = float(t.max())
    if high <= low:  # (nearly) one value: a single bin around it
        pad = abs(low) * 0.05 or 0.5
        low, high = low - pad, high + pad
    inside = t[(t >= low) & (t <= high)]
    iqr = float(np.subtract(*np.percentile(inside, [75, 25])))
    width = 2 * iqr / len(inside) ** (1 / 3) if iqr > 0 else 0  # Freedman–Diaconis
    bins = int(np.clip(np.ceil((high - low) / width) if width else MIN_BINS, MIN_BINS, MAX_BINS))
    t_edges = np.linspace(low, high, bins + 1)
    edges = 10 ** t_edges if log else t_edges

    shares, counts, below, above = [], [], [], []
    for x in present:
        if log:
            nonpositive = int((x <= 0).sum())
            tx = np.log10(x[x > 0])
        else:
            nonpositive, tx = 0, x
        index = bin_index(tx, t_edges)
        c = np.bincount(index[index >= 0], minlength=bins)
        counts.append(c)
        shares.append(c / len(x) if len(x) else np.zeros(bins))
        below.append(int((tx < low).sum()) + nonpositive)
        above.append(int((tx > high).sum()))
    return Histogram(edges, shares, counts, below, above, log, t_edges)


@dataclass
class Categories:
    names: list[str]  # most common first, OTHER last if anything was folded
    shares: list[np.ndarray]
    counts: list[np.ndarray]


def categories(groups: list[Group]) -> Categories | None:
    texts = [g.present.astype(str) for g in groups]
    pooled = np.concatenate(texts) if texts else np.zeros(0, dtype=str)
    if not len(pooled):
        return None
    names, counts = np.unique(pooled, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    kept = [str(n) for n in names[order][:MAX_CATEGORIES]]
    folded = len(names) > MAX_CATEGORIES
    labels = kept + ([OTHER] if folded else [])
    index = {name: i for i, name in enumerate(kept)}
    shares, all_counts = [], []
    for x in texts:
        c = np.zeros(len(labels), dtype=int)
        for value, k in zip(*np.unique(x, return_counts=True)):
            c[index.get(str(value), len(kept))] += k
        all_counts.append(c)
        shares.append(c / len(x) if len(x) else np.zeros(len(labels)))
    return Categories(labels, shares, all_counts)


# --- Effect sizes -----------------------------------------------------------------------------------


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float | None:
    """P(b > a) − P(b < a), via ranks (ties count half)."""
    if not len(a) or not len(b):
        return None
    pooled = np.concatenate([a, b])
    _, inverse, counts = np.unique(pooled, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    ranks = (upper - (counts - 1) / 2)[inverse]  # average ranks, 1-based
    u_b = ranks[len(a):].sum() - len(b) * (len(b) + 1) / 2  # pairs where b > a, ties half
    return float(2 * u_b / (len(a) * len(b)) - 1)


def ks_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    if not len(a) or not len(b):
        return None
    grid = np.concatenate([a, b])
    fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.abs(fa - fb).max())


def cramers_v(counts_a: np.ndarray, counts_b: np.ndarray) -> float | None:
    table = np.vstack([counts_a, counts_b]).astype(float)
    table = table[:, table.sum(axis=0) > 0]
    n = table.sum()
    if table.shape[1] < 2 or table[0].sum() == 0 or table[1].sum() == 0:
        return None
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    return float(np.sqrt(chi2 / n))  # min(rows, cols) − 1 = 1 for two groups


@dataclass
class Effect:
    """Group ``b`` compared with group ``a``."""
    a: int
    b: int
    size: float | None  # Cliff's δ (continuous, signed) or Cramér's V (categorical)
    magnitude: str | None
    ks: float | None = None
    median_change: float | None = None  # b's median ÷ a's − 1


def effects(groups: list[Group], categorical: bool, cats: Categories | None = None) -> list[Effect]:
    out = []
    present = [g.present for g in groups]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if categorical:
                v = cramers_v(cats.counts[i], cats.counts[j]) if cats else None
                out.append(Effect(i, j, v, None if v is None else magnitude(v, CRAMER_LEVELS)))
                continue
            a, b = present[i].astype(float), present[j].astype(float)
            d = cliffs_delta(a, b)
            change = None
            if len(a) and len(b):
                ma = float(np.median(a))
                change = float(np.median(b)) / ma - 1 if ma != 0 else None
            out.append(Effect(i, j, d, None if d is None else magnitude(d, CLIFF_LEVELS), ks_distance(a, b),
                              change))
    return out


# --- Everything for one view ----------------------------------------------------------------------


@dataclass
class Comparison:
    attribute: Attribute
    population: str
    groups: list[Group] = field(default_factory=list)
    summaries: list[Summary] = field(default_factory=list)
    histogram: Histogram | None = None
    categories: Categories | None = None
    effects: list[Effect] = field(default_factory=list)
    log_available: bool = False


def compare(session, attr: Attribute, selections: list[Selection], population: str,
            log: bool = False) -> Comparison:
    groups = build_groups(session, attr, selections, population)
    result = Comparison(attr, population, groups, [summarize(g, attr.categorical) for g in groups])
    if attr.categorical:
        result.categories = categories(groups)
    else:
        result.log_available = can_log(groups)
        result.histogram = histogram(groups, log and result.log_available)
    result.effects = effects(groups, attr.categorical, result.categories)
    return result


# --- Map linking -------------------------------------------------------------------------------------


def bin_parcels(result: Comparison, row: int, b: int) -> np.ndarray:
    """Parcels behind one bar (group ``row``, bin or category ``b``), each once.

    For a sales group these are the sold parcels. Uses the same bins and
    clipping as the drawn histogram, so the parcels match the bar's count.
    """
    group = result.groups[row]
    present = group.present_mask
    values, parcels = group.values[present], group.parcels[present]
    if result.categories is not None:
        names = result.categories.names
        text = values.astype(str)
        hit = ~np.isin(text, names[:-1]) if names[b] == OTHER else text == names[b]
        return np.unique(parcels[hit])
    h = result.histogram
    x = values.astype(float)
    if h.log:
        keep = x > 0
        x, parcels = np.log10(x[keep]), parcels[keep]
    return np.unique(parcels[bin_index(x, h.t_edges) == b])


def group_parcels(result: Comparison, row: int) -> np.ndarray:
    """Every parcel of a group that has a value (for a sales group, the sold parcels)."""
    group = result.groups[row]
    return np.unique(group.parcels[group.present_mask])
