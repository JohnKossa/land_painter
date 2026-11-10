"""MainWindow smoke test (offscreen).

Constructs the real main window against a small session and exercises the map
data producers, the change fan-outs, selection and a neighborhood edit. It is a
regression guard for refactors of app.py: it does not assert pixels, only that
the wiring holds and the map buffers stay consistent.

Map buffers are read through the scheme handler by name, so the guard does not
care which object actually produces them.
"""

import os

import numpy as np
import pytest

from land_painter.data.config import ROLE_SALES, ColumnRef, FileSpec, SalesMapping, parcel_only_config
from land_painter.data.dataset import build_dataset
from land_painter.data.sources import inspect
from land_painter.neighborhoods import ROOT
from land_painter.sales import DISQUALIFIED
from land_painter.session import Session

from test_core import synthetic


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def session(tmp_path):
    gdf = synthetic.make_parcels(6)
    gdf["area"] = [100.0, 200.0, 100.0, 400.0, 100.0, 0.0]
    gdf["cost"] = [0.0, 0.0, 50.0, 100.0, 300.0, 0.0]
    parcels = tmp_path / "parcels.parquet"
    gdf.to_parquet(parcels)
    sales = tmp_path / "sales.csv"
    sales.write_text("key,price,date,vacant\nP0,1000,2024-01-01,Y\nP1,2400,2024-01-02,Y\nP2,250,2024-01-03,N\n")
    config = parcel_only_config(inspect(parcels))
    config.files.append(FileSpec(str(sales), [ROLE_SALES]))
    config.sales = SalesMapping("key", "price", "date", "vacant", ["Y"])
    config.units = {"Area": ColumnRef(config.parcel_file, "area")}
    config.depreciated_cost = [ColumnRef(config.parcel_file, "cost")]
    return Session.new(build_dataset(config))


def test_mainwindow_smoke(qapp, session):
    from land_painter.app import MainWindow

    w = MainWindow(session)
    produce = w.map.scheme_handler._data  # name -> bytes producer, wherever it lives

    # Map data producers return buffers, and sale points/colors stay in step.
    assert len(produce["meta"]()) > 0
    assert len(produce["assignment"]()) > 0
    points, colors = produce["sale_points"](), produce["sale_colors"]()
    assert len(points) > 0 and len(points) % 24 == 0  # float64[3] per shown sale
    assert len(points) // 24 == len(colors) // 4       # one RGBA per shown sale

    # The always-visible banner is populated.
    assert w.banner.text()

    # Change fan-outs run cleanly.
    w._values_changed()
    w._assignment_changed()

    # Selection updates state; panel metrics are skipped while the popup is hidden.
    w._select(ROOT)
    assert w.selected == ROOT
    assert not w.popup_dock.isVisible()

    # A neighborhood edit is reflected in the per-part assignment buffer.
    child = session.create_neighborhood(ROOT, np.array([0, 1, 2]))
    assert child is not None
    w._assignment_changed()
    parts = np.frombuffer(produce["assignment"](), dtype=session.neighborhoods.assignment.dtype)
    assert parts.max() == child

    # Flagging a sale runs the flag fan-out without error and keeps buffers consistent.
    w._flag_sales(np.array([0]), DISQUALIFIED)
    assert len(produce["sale_points"]()) // 24 == len(produce["sale_colors"]()) // 4

    # Map focus / zoom / highlight / fly paths run and the focus buffer tracks parcels.
    w._focus_parcels(np.array([0, 1]))
    assert len(produce["focus"]()) > 0
    w._focus_parcels(None)
    assert produce["focus"]() == b""
    w._zoom_to_parcels(np.array([0, 1]))
    w._zoom_to_parcels(None)
    w._highlight_sale(0)
    w._zoom_to_sale(0)
    w._sales_changed()
