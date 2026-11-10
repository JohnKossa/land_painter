import numpy as np
import pytest
from test_core import synthetic

from land_painter.data.config import ColumnRef, parcel_only_config
from land_painter.data.dataset import build_dataset
from land_painter.data.sources import inspect
from land_painter.split import MIN_GROUP, decompose, split_check

N = 200
K = 10  # neighborhoods of 20


def test_decompose_known_values():
    y = np.array([1.0, 2, 3, 4, 5, 6])
    g = np.array([0, 0, 0, 1, 1, 1])
    v = decompose(y, g)
    # group means 2 and 5, grand mean 3.5: SSB = 3·1.5²·2 = 13.5, SST = 17.5, SSW = 4
    assert v.eta2 == pytest.approx(13.5 / 17.5)
    msw = 4 / 4
    assert v.omega2 == pytest.approx((13.5 - 1 * msw) / (17.5 + msw))
    assert v.icc == pytest.approx((13.5 - msw) / (13.5 + 2 * msw))  # balanced: n₀ = 3
    assert (v.n, v.k) == (6, 2)


def test_decompose_undefined():
    assert decompose(np.array([1.0, 2.0]), np.array([0, 0])) is None  # one group
    assert decompose(np.array([1.0, 1.0, 1.0]), np.array([0, 0, 1])) is None  # no variation


def test_random_groups_have_small_adjusted_effect():
    rng = np.random.default_rng(0)
    y, g = rng.normal(size=400), np.repeat(np.arange(80), 5)  # many small groups
    v = decompose(y, g)
    assert v.eta2 > 0.1  # η² is inflated by many small groups…
    assert abs(v.omega2) < 0.06 and abs(v.icc) < 0.1  # …ω² and the ICC are not


def make_dataset(tmp_path, building_factor, land_factor, codes=None):
    rng = np.random.default_rng(1)
    hood = np.repeat(np.arange(K), N // K)
    area = rng.uniform(5_000, 15_000, N)
    cost = rng.uniform(100_000, 300_000, N)
    gdf = synthetic.make_parcels(N)
    gdf["area"] = area
    gdf["cost"] = cost
    gdf["bldg_val"] = cost * building_factor[hood] * np.exp(rng.normal(0, 0.05, N))
    gdf["land_val"] = area * 10 * land_factor[hood] * np.exp(rng.normal(0, 0.05, N))
    gdf["nbhd"] = [f"NB{h:02d}" for h in hood] if codes is None else codes
    path = tmp_path / "parcels.parquet"
    gdf.to_parquet(path)
    config = parcel_only_config(inspect(path))
    p = config.parcel_file
    config.depreciated_cost = [ColumnRef(p, "cost")]
    config.roll_building_value = [ColumnRef(p, "bldg_val")]
    config.roll_land_value = [ColumnRef(p, "land_val")]
    config.units = {"Area": ColumnRef(p, "area")}
    config.neighborhood = ColumnRef(p, "nbhd")
    dataset = build_dataset(config)
    return dataset, hood


def run(dataset, hood):
    return split_check(dataset, np.full(N, np.nan), hood, {i: f"NB{i:02d}" for i in range(K)}, "Area")


def test_location_booked_as_building_is_detected(tmp_path):
    location = np.linspace(0.6, 1.6, K)  # building and land both rise with location
    dataset, hood = make_dataset(tmp_path, building_factor=location, land_factor=location)
    r = run(dataset, hood)
    assert r.problem is None and r.n_improved == N and r.building.k == K
    assert r.building.omega2 > 0.8 and r.roll_land.omega2 > 0.8
    assert r.leakage == pytest.approx(1, abs=0.1)  # split across both
    assert r.groups[0].name == "NB09" and r.groups[0].building_ratio > r.groups[-1].building_ratio
    assert r.app_land is None  # no land tables in this session


def test_clean_split_shows_no_building_effect(tmp_path):
    dataset, hood = make_dataset(tmp_path, building_factor=np.ones(K), land_factor=np.linspace(0.6, 1.6, K))
    r = run(dataset, hood)
    assert r.building.omega2 < 0.06 and r.roll_land.omega2 > 0.8
    assert r.leakage < 0.1


def test_small_groups_are_left_out(tmp_path):
    dataset, hood = make_dataset(tmp_path, np.linspace(0.6, 1.6, K), np.ones(K))
    hood = hood.copy()
    hood[:MIN_GROUP - 1] = 99  # a tiny neighborhood
    r = run(dataset, hood)
    assert r.excluded_groups == 1 and r.excluded_parcels == MIN_GROUP - 1


def test_needs_roll_building_and_cost(tmp_path):
    dataset, hood = make_dataset(tmp_path, np.ones(K), np.ones(K))
    dataset.config.roll_building_value = []
    dataset.parcels = dataset.parcels.drop("roll_building_value")
    assert "roll building value" in run(dataset, hood).problem
