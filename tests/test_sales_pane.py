"""SalesTableModel: the per-sale land-value/ratio arrays are cached and invalidated."""

import os
from datetime import date

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def make_model():
    from land_painter.ui.sales_pane import SalesTableModel

    sales = pl.DataFrame({
        "parcel": np.array([0, 1, 2], dtype=np.uint32),
        "key": ["P0", "P1", "P2"],
        "price": [1000.0, 2000.0, None],
        "date": [date(2024, 1, 1), date(2024, 2, 1), None],
        "vacant": [True, False, None],
    })
    land_values = np.array([100.0, 200.0, 300.0])
    flags = np.zeros(3, dtype=np.uint8)
    neighborhood_of = lambda p: np.array(["A"] * len(p), dtype=object)  # noqa: E731
    return SalesTableModel(sales, flags, land_values, {}, neighborhood_of), land_values


def test_sale_land_values_and_ratios_are_cached(qapp):
    model, _ = make_model()
    first = model.sale_land_values()
    assert model.sale_land_values() is first  # same object: not reallocated per call
    assert list(first) == [100.0, 200.0, 300.0]
    r = model.ratios()
    assert model.ratios() is r
    assert r[0] == pytest.approx(0.1)  # 100 / 1000


def test_refresh_invalidates_the_cache_after_land_values_change(qapp):
    model, land_values = make_model()
    first = model.sale_land_values()

    land_values[0] = 999.0  # the session mutates land_values in place
    assert model.sale_land_values() is first  # stale until refresh
    assert model.ratios()[0] == pytest.approx(0.1)

    model.refresh()
    refreshed = model.sale_land_values()
    assert refreshed is not first and refreshed[0] == 999.0
    assert model.ratios()[0] == pytest.approx(0.999)


def test_filtering_and_sorting_keep_the_cache(qapp):
    from PySide6.QtCore import Qt

    model, _ = make_model()
    first = model.sale_land_values()
    model.set_mask(np.array([True, False, True]))  # filter: narrows rows, not values
    model.sort(model.columns.index(next(c for c in model.columns if c.header == "Land value")),
               Qt.SortOrder.AscendingOrder)
    assert model.sale_land_values() is first  # per-sale arrays are absolute-indexed, so unaffected
