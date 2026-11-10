import numpy as np
import pytest
from test_core import synthetic

from land_painter.allocation import amenity_allocation
from land_painter.data.config import CATEGORICAL, ROLE_SALES, ColumnRef, FileSpec, SalesMapping, parcel_only_config
from land_painter.data.dataset import build_dataset
from land_painter.data.sources import inspect
from land_painter.land_table import IN, LOT, MULTIPLY, Adjustment, Condition, FlatRate, LandTable
from land_painter.neighborhoods import ROOT
from land_painter.session import Session

N = 120


def make_session(tmp_path, golf, neighborhood_b=None, market=0.20, building=0.15, level_b=2.0):
    """Improved sales whose log price, cost and land have known golf premiums.

    price = 50·area^0.8 · (1+market)^golf · level_b^(in B)
    cost  = 100·area^0.5 · (1+building)^golf
    land  = 20,000 per lot, × 1.05 for golf (the land table)
    """
    rng = np.random.default_rng(0)
    area = rng.uniform(5_000, 15_000, N)
    in_b = np.zeros(N, bool) if neighborhood_b is None else neighborhood_b
    gdf = synthetic.make_parcels(N)
    gdf["area"] = area
    gdf["golf"] = np.where(golf, "Y", "N")
    gdf["cost"] = 100 * area ** 0.5 * (1 + building) ** golf
    parcels = tmp_path / "parcels.parquet"
    gdf.to_parquet(parcels)
    price = 50 * area ** 0.8 * (1 + market) ** golf * np.where(in_b, level_b, 1.0)
    sales = tmp_path / "sales.csv"
    sales.write_text("key,price,date,vacant\n" + "\n".join(
        f"P{i},{price[i]:.6f},2024-01-01,N" for i in range(N)) + "\n")
    config = parcel_only_config(inspect(parcels))
    p = config.parcel_file
    config.files.append(FileSpec(str(sales), [ROLE_SALES]))
    config.sales = SalesMapping("key", "price", "date", "vacant", ["Y"])
    config.units = {"Area": ColumnRef(p, "area")}
    config.depreciated_cost = [ColumnRef(p, "cost")]
    config.characteristic_kinds[f"{p}::golf"] = CATEGORICAL
    session = Session.new(build_dataset(config))
    golf_condition = Condition(f"{p}::golf", IN, values=("Y",))
    table = LandTable(LOT, FlatRate(20_000.0), (Adjustment(golf_condition, MULTIPLY, 1.05),))
    session.set_land_table(ROOT, table)
    if neighborhood_b is not None:
        child = session.create_neighborhood(ROOT, np.flatnonzero(in_b))
        session.set_land_table(child, table)
    controls = {"Area": session.dataset.unit_values("Area")}
    return session, golf_condition, controls


def everything():
    return np.ones(N, dtype=bool)


def test_recovers_known_premiums(tmp_path):
    golf = np.arange(N) % 3 == 0
    session, condition, controls = make_session(tmp_path, golf)
    r = amenity_allocation(session, condition, everything(), controls)
    assert r.problem is None and r.n_sales == N and r.n_amenity == golf.sum()
    assert r.controls == ["ln(Area)"]
    assert r.market.percent == pytest.approx(0.20)
    assert r.land.percent == pytest.approx(0.05)
    assert r.building.percent == pytest.approx(0.15)
    land_d, bldg_d = 0.05 * r.land.median, 0.15 * r.building.median
    assert r.land.dollars == pytest.approx(land_d) and r.building.dollars == pytest.approx(bldg_d)
    assert r.building_share == pytest.approx(bldg_d / (land_d + bldg_d))
    assert r.assessed_dollars == pytest.approx(land_d + bldg_d)


def test_fixed_effects_remove_neighborhood_confounding(tmp_path):
    # Golf parcels are mostly in neighborhood B, which sells at double the price
    in_b = np.arange(N) >= N // 2
    golf = np.where(in_b, np.arange(N) % 4 != 0, np.arange(N) % 4 == 0)
    session, condition, controls = make_session(tmp_path, golf, neighborhood_b=in_b)
    r = amenity_allocation(session, condition, everything(), controls)
    assert r.n_groups == 2
    assert r.market.percent == pytest.approx(0.20)  # without fixed effects this would absorb B's level


def test_single_neighborhood_scope(tmp_path):
    in_b = np.arange(N) >= N // 2
    golf = np.arange(N) % 3 == 0
    session, condition, controls = make_session(tmp_path, golf, neighborhood_b=in_b)
    mask = session.neighborhoods.assignment == session.neighborhoods.assignment[N - 1]
    r = amenity_allocation(session, condition, mask, controls)
    assert r.n_groups == 1 and r.n_sales == N // 2
    assert r.market.percent == pytest.approx(0.20)


def test_amenity_confounded_with_neighborhood_is_reported(tmp_path):
    in_b = np.arange(N) >= N // 2
    session, condition, controls = make_session(tmp_path, in_b, neighborhood_b=in_b)
    r = amenity_allocation(session, condition, everything(), controls)
    assert r.market is None and "can't be separated" in r.problem


def test_no_amenity_sales(tmp_path):
    session, condition, controls = make_session(tmp_path, np.zeros(N, bool))
    r = amenity_allocation(session, condition, everything(), controls)
    assert r.market is None and "None of the usable" in r.problem


def test_opposite_signs_have_no_share(tmp_path):
    golf = np.arange(N) % 3 == 0
    session, condition, controls = make_session(tmp_path, golf, market=-0.1, building=-0.2)
    r = amenity_allocation(session, condition, everything(), controls)
    assert r.building.dollars < 0 < r.land.dollars and r.building_share is None
