from datetime import date, timedelta

import numpy as np
import pytest
from test_core import synthetic

from land_painter.data.config import ROLE_SALES, FileSpec, SalesMapping, parcel_only_config
from land_painter.data.dataset import build_dataset
from land_painter.data.sources import inspect
from land_painter.land_table import LOT, FlatRate, LandTable
from land_painter.neighborhoods import ROOT
from land_painter.sales import DISQUALIFIED
from land_painter.session import Session
from land_painter.spatial import WINDOW_DAYS, _neighbors, residual_autocorrelation

N = 64  # 8 x 8 grid, one vacant sale per parcel
COLS = 8


def make_session(tmp_path, ratios, days_apart=100, extra_rows=""):
    """Every parcel valued at 1000 (flat rate per lot), sold at 1000 / ratio."""
    parcels = tmp_path / "parcels.parquet"
    synthetic.make_parcels(N).to_parquet(parcels)
    start = date(2020, 1, 1)
    rows = [f"P{i},{1000 / ratio:.4f},{start + timedelta(days=i * days_apart)},Y" for i, ratio in enumerate(ratios)]
    sales = tmp_path / "sales.csv"
    sales.write_text("key,price,date,vacant\n" + "\n".join(rows) + "\n" + extra_rows)
    config = parcel_only_config(inspect(parcels))
    config.files.append(FileSpec(str(sales), [ROLE_SALES]))
    config.sales = SalesMapping("key", "price", "date", "vacant", ["Y"])
    session = Session.new(build_dataset(config))
    session.set_land_table(ROOT, LandTable(LOT, FlatRate(1000.0)))
    return session


def everything(session):
    return np.ones(session.layer.n_parcels, dtype=bool)


def test_east_west_gradient_is_detected(tmp_path):
    column = np.arange(N) % COLS
    session = make_session(tmp_path, np.exp(0.3 * (column - column.mean()) / COLS))
    t = residual_autocorrelation(session, everything(session))
    assert t.n_sales == N and t.k == 8
    assert t.all_pairs.rho > 0.8 and t.all_pairs.p_value < 0.01
    # A sale is never in its own lag, so the shuffled null sits a little below 0 in a small sample
    assert -0.15 < t.all_pairs.null_mean < 0
    # Sales 100 days apart: the time window removes no neighbors, so nothing changes
    assert t.time_controlled.rho == pytest.approx(t.all_pairs.rho)


def test_random_ratios_are_not_clustered(tmp_path):
    session = make_session(tmp_path, np.exp(np.random.default_rng(1).normal(0, 0.2, N)))
    t = residual_autocorrelation(session, everything(session))
    assert abs(t.all_pairs.rho) < 0.3 and t.all_pairs.p_value > 0.05


def test_result_is_repeatable(tmp_path):
    session = make_session(tmp_path, np.exp(np.random.default_rng(2).normal(0, 0.2, N)))
    a = residual_autocorrelation(session, everything(session))
    b = residual_autocorrelation(session, everything(session))
    assert a == b


def test_time_control_refills_and_can_run_out(tmp_path):
    column = np.arange(N) % COLS
    session = make_session(tmp_path, np.exp(0.3 * (column - column.mean()) / COLS), days_apart=0)
    t = residual_autocorrelation(session, everything(session))
    assert t.all_pairs.rho > 0.8
    # Every sale on the same day: no neighbor is outside the window
    assert t.time_controlled.rho is None and t.time_controlled.n == 0


def test_neighbors_exclude_same_parcel_and_window():
    xy = np.array([[0.0, 0], [0, 0], [1, 0], [5, 0], [6, 0]])
    parcel = np.array([0, 0, 1, 2, 3])  # sales 0 and 1 are a resale of one parcel
    index, valid = _neighbors(xy, parcel, None, 2)
    assert set(index[0][valid[0]]) == {2, 3}  # skips its own resale at distance 0
    days = np.array([0.0, 500, WINDOW_DAYS, 1000, np.nan])
    index, valid = _neighbors(xy, parcel, days, 2)
    assert set(index[0][valid[0]]) == {3, 4}  # sale 2 is inside the window; a missing date is kept


def test_undefined_without_enough_sales(tmp_path):
    session = make_session(tmp_path, np.ones(N))
    one = np.zeros(N, dtype=bool)
    one[0] = True
    assert residual_autocorrelation(session, one).all_pairs.rho is None
    # Identical ratios everywhere: no variation to correlate
    assert residual_autocorrelation(session, everything(session)).all_pairs.rho is None


def test_disqualified_and_unvalued_sales_are_left_out(tmp_path):
    session = make_session(tmp_path, np.exp(np.random.default_rng(3).normal(0, 0.2, N)))
    session.set_sale_flags(np.arange(10), DISQUALIFIED)
    t = residual_autocorrelation(session, everything(session))
    assert t.n_sales == N - 10
