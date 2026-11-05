"""Residual spatial autocorrelation of land ratios within one neighborhood.

Asks whether sale ratios cluster in space after the neighborhood's level is
removed, i.e. whether the land table misses a location gradient inside it.

1. Residual per sale: r = ln(land value / price) − median ln(ratio) over the
   neighborhood's sales (the ratio study's sales: usable vacant sales with a
   positive land value).
2. Neighbors: each sale's K nearest other sales in the neighborhood, never on
   the same parcel (a resale would be its own neighbor by construction).
3. Spatial lag: the mean of the neighbors' residuals.
4. Statistic: ρ = corr(r, lag), Moran's I with row-standardized weights in
   correlation form (so not on Moran's scale).
5. Null: residuals permuted among the neighborhood's sales; the p-value is
   one-sided, for positive clustering.
6. Time control: the same, with neighbors within WINDOW_DAYS of a sale's date
   excluded and refilled with the next-nearest eligible sales, so a surviving
   ρ is not nearby sales sharing a market moment. Pairs with a missing date are
   kept, since their timing can't be checked.

Computed only for the selected neighborhood (never county-wide, never for the
land table draft preview), so its cost stays at tens of milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .sales import usable

K = 8  # neighbors per sale (fewer if the neighborhood has fewer eligible sales)
WINDOW_DAYS = 90  # time control: neighbors sold within this many days are excluded
PERMUTATIONS = 999
SEED = 0  # fixed, so the same neighborhood always reports the same p-value
_ROW_CHUNK = 1024  # rows of the distance matrix held at once
_PERM_CHUNK = 128  # permutations evaluated at once


@dataclass
class Autocorrelation:
    n: int  # sales with at least one neighbor
    rho: float | None
    null_mean: float | None  # mean ρ over the permutations
    p_value: float | None  # one-sided: share of permutations with ρ ≥ observed


@dataclass
class SpatialTest:
    n_sales: int  # sales in the test (ratio study sales)
    k: int  # neighbors per sale used (K, or fewer for small neighborhoods)
    all_pairs: Autocorrelation
    time_controlled: Autocorrelation  # neighbors within WINDOW_DAYS excluded


def _neighbors(xy: np.ndarray, parcel: np.ndarray, days: np.ndarray | None, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Nearest eligible neighbors: (n, k) indices and a validity mask.

    Pairs on the same parcel are never neighbors; with ``days`` given, neither
    are pairs sold within WINDOW_DAYS of each other (unless a date is missing).
    """
    n = len(xy)
    index = np.zeros((n, k), dtype=np.int64)
    valid = np.zeros((n, k), dtype=bool)
    for start in range(0, n, _ROW_CHUNK):
        rows = slice(start, min(start + _ROW_CHUNK, n))
        d = ((xy[rows, None, :] - xy[None, :, :]) ** 2).sum(-1)
        blocked = parcel[rows, None] == parcel[None, :]  # includes the sale itself
        if days is not None:
            with np.errstate(invalid="ignore"):
                blocked |= np.abs(days[rows, None] - days[None, :]) <= WINDOW_DAYS  # NaN compares False
        d[blocked] = np.inf
        nearest = np.argpartition(d, k - 1, axis=1)[:, :k] if k < n else np.argsort(d, axis=1)[:, :k]
        index[rows] = nearest
        valid[rows] = np.isfinite(np.take_along_axis(d, nearest, axis=1))
    return index, valid


def _rows_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of each row of ``a`` with the same row of ``b`` (NaN if constant)."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(axis=1) / denom


def _test(r: np.ndarray, index: np.ndarray, valid: np.ndarray) -> Autocorrelation:
    has = valid.any(axis=1)
    if has.sum() < 3:  # a correlation needs 3 points to mean anything
        return Autocorrelation(int(has.sum()), None, None, None)
    index, weights = index[has], valid[has].astype(float)
    count = weights.sum(axis=1)

    def rho(residuals: np.ndarray) -> np.ndarray:  # residuals: (m, n_all)
        lag = (residuals[:, index] * weights).sum(axis=2) / count
        return _rows_corr(residuals[:, has], lag)

    observed = rho(r[None, :])[0]
    if np.isnan(observed):  # all residuals (or lags) equal
        return Autocorrelation(int(has.sum()), None, None, None)
    rng = np.random.default_rng(SEED)
    null = []
    for start in range(0, PERMUTATIONS, _PERM_CHUNK):
        m = min(_PERM_CHUNK, PERMUTATIONS - start)
        null.append(rho(rng.permuted(np.broadcast_to(r, (m, len(r))), axis=1)))
    null = np.concatenate(null)
    null = null[~np.isnan(null)]
    p = (1 + int((null >= observed).sum())) / (1 + len(null))
    return Autocorrelation(int(has.sum()), float(observed), float(null.mean()) if len(null) else None, p)


def residual_autocorrelation(session, parcel_mask: np.ndarray) -> SpatialTest:
    """The test for the parcels in ``parcel_mask`` (one neighborhood) and their sales."""
    empty = SpatialTest(0, 0, Autocorrelation(0, None, None, None), Autocorrelation(0, None, None, None))
    sales = session.dataset.sales
    if sales is None:
        return empty
    sale_parcel = sales["parcel"].to_numpy()
    price = sales["price"].fill_null(np.nan).to_numpy().astype(float)
    vacant = sales["vacant"].fill_null(False).to_numpy()
    land = session.land_values[sale_parcel]
    with np.errstate(invalid="ignore"):
        ok = usable(session.sale_flags, sales) & parcel_mask[sale_parcel] & vacant & (price > 0) & (land > 0)
    chosen = np.flatnonzero(ok)
    if len(chosen) < 2:
        return SpatialTest(len(chosen), 0, empty.all_pairs, empty.time_controlled)

    log_ratio = np.log(land[chosen] / price[chosen])
    r = log_ratio - np.median(log_ratio)
    parcel = sale_parcel[chosen]
    lng_lat = session.layer.parcel_points(parcel)
    # Equirectangular projection: fine for ranking distances within a neighborhood
    xy = np.column_stack([lng_lat[:, 0] * np.cos(np.radians(lng_lat[:, 1].mean())), lng_lat[:, 1]])
    dates = sales["date"][chosen]
    days = (dates.cast(pl.Date).cast(pl.Int32).cast(pl.Float64).fill_null(np.nan).to_numpy()
            if dates.dtype.is_temporal() else None)  # days since epoch, NaN where missing

    k = min(K, len(chosen) - 1)
    all_pairs = _test(r, *_neighbors(xy, parcel, None, k))
    timed = _test(r, *_neighbors(xy, parcel, days, k)) if days is not None else Autocorrelation(0, None, None, None)
    return SpatialTest(len(chosen), k, all_pairs, timed)
