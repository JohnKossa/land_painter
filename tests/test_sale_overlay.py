"""The sale-ratio map overlay: per-sale land ratios and their diverging colours."""

from __future__ import annotations

import numpy as np

from land_painter.sales import sale_land_ratios
from land_painter.ui.colors import RATIO_MID, RATIO_NODATA, ratio_colors


def _pixels(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8).reshape(-1, 4)


def test_sale_land_ratios_vacant_uses_price_improved_uses_residual():
    land = np.array([100.0, 200.0, 400.0])
    parcel = np.array([0, 1, 2])
    price = np.array([100.0, 1000.0, 50.0])
    vacant = np.array([True, False, True])
    known = np.array([True, True, True])
    cost = np.array([np.nan, 800.0, np.nan])
    r = sale_land_ratios(land, parcel, price, vacant, known, cost)
    assert r[0] == 1.0          # vacant: 100 / 100
    assert r[1] == 1.0          # improved: 200 / (1000 - 800)
    assert r[2] == 8.0          # vacant: 400 / 50


def test_sale_land_ratios_undefined_cases():
    land = np.array([np.nan, 100.0, 100.0, 100.0])
    parcel = np.array([0, 1, 2, 3])
    price = np.array([100.0, 100.0, 0.0, 500.0])
    vacant = np.array([True, True, True, False])
    known = np.array([True, False, True, True])
    cost = np.array([np.nan, np.nan, np.nan, 600.0])  # last: price - cost < 0
    r = sale_land_ratios(land, parcel, price, vacant, known, cost)
    assert np.isnan(r[0])       # no land value
    assert np.isnan(r[1])       # sale type unknown
    assert np.isnan(r[2])       # non-positive price
    assert np.isnan(r[3])       # improved residual <= 0


def test_sale_land_ratios_improved_without_cost_is_undefined():
    r = sale_land_ratios(np.array([100.0]), np.array([0]), np.array([500.0]),
                         np.array([False]), np.array([True]), np.array([np.nan]))
    assert np.isnan(r[0])


def test_ratio_colors_centres_on_the_median():
    ratios = np.array([np.nan, 0.5, 1.0, 2.0])  # usable median = 1.0
    data, center, n = ratio_colors(ratios)
    assert center == 1.0
    assert n == 3
    px = _pixels(data)
    assert tuple(px[0]) == RATIO_NODATA           # NaN -> no-data grey
    assert tuple(px[2][:3]) == RATIO_MID          # the median sits at the midpoint colour
    assert px[1][2] > px[3][2]                     # below median is bluer than above
    assert px[3][0] > px[1][0]                     # above median is redder than below
    assert (px[:, 3] == np.array([RATIO_NODATA[3], 255, 255, 255])).all()


def test_ratio_colors_symmetric_log_spread():
    # 0.5 and 2.0 are symmetric in log space about 1.0, so they saturate equally.
    data, _, _ = ratio_colors(np.array([0.5, 1.0, 2.0]))
    px = _pixels(data)
    assert tuple(px[0][:3]) == (33, 102, 172)      # fully under-valued (RATIO_UNDER)
    assert tuple(px[2][:3]) == (178, 24, 43)       # fully over-valued (RATIO_OVER)


def test_ratio_colors_no_usable_ratios():
    data, center, n = ratio_colors(np.array([np.nan, -1.0, 0.0]))
    assert center is None
    assert n == 0
    assert all(tuple(p) == RATIO_NODATA for p in _pixels(data))


def test_ratio_colors_single_sale_is_neutral():
    data, center, n = ratio_colors(np.array([1.37]))
    assert center == 1.37
    assert n == 1
    assert tuple(_pixels(data)[0][:3]) == RATIO_MID  # nothing to diverge against
