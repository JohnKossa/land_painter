import numpy as np
import pytest

from land_painter.data.config import ROLE_SALES, ColumnRef, FileSpec, SalesMapping, parcel_only_config
from land_painter.data.dataset import build_dataset
from land_painter.data.sources import inspect
from land_painter.land_table import FlatRate, LandTable
from land_painter.metrics import MOVE_THRESHOLD, compute
from land_painter.neighborhoods import ROOT
from land_painter.session import Session

from test_core import synthetic


def make_session(tmp_path, with_roll=True):
    gdf = synthetic.make_parcels(6)
    gdf["area"] = [100.0, 200.0, 100.0, 400.0, 100.0, 0.0]
    gdf["roll_land"] = [800.0, 2000.0, 1000.0, 2000.0, None, 500.0]
    parcels = tmp_path / "parcels.parquet"
    gdf.to_parquet(parcels)
    sales = tmp_path / "sales.csv"
    sales.write_text("key,price,date,vacant\nP0,1000,2024-01-01,Y\nP1,2400,2024-01-02,Y\n")
    config = parcel_only_config(inspect(parcels))
    p = config.parcel_file
    config.files.append(FileSpec(str(sales), [ROLE_SALES]))
    config.sales = SalesMapping("key", "price", "date", "vacant", ["Y"])
    config.units = {"Area": ColumnRef(p, "area")}
    if with_roll:
        config.roll_land_value = [ColumnRef(p, "roll_land")]
    session = Session.new(build_dataset(config))
    session.set_land_table(ROOT, LandTable("Area", FlatRate(10.0)))  # land: 1000, 2000, 1000, 4000, 1000, 0
    return session


def everything():
    return np.ones(6, dtype=bool)


def test_change_from_roll(tmp_path):
    r = compute(make_session(tmp_path), everything(), "Area").roll
    # Compared: parcels 0, 1, 2, 3, 5 → +25%, 0%, 0%, +100%, −100% (new value 0); parcel 4 has no roll value
    assert r.n_compared == 5 and r.n_no_roll == 1 and r.n_no_value == 0
    assert r.median_change == pytest.approx(0.0)
    assert (r.q25, r.q75) == (pytest.approx(0.0), pytest.approx(0.25))
    assert r.threshold == MOVE_THRESHOLD
    assert (r.n_up, r.n_down) == (1, 1)  # +25% exactly is not beyond the threshold


def test_threshold_is_a_parameter(tmp_path):
    r = compute(make_session(tmp_path), everything(), "Area", move_threshold=0.2).roll
    assert r.threshold == 0.2 and (r.n_up, r.n_down) == (2, 1)


def test_roll_ratio_study_uses_the_same_vacant_sales(tmp_path):
    m = compute(make_session(tmp_path), everything(), "Area")
    assert m.roll.roll_ratio.n == 2
    assert m.roll.roll_ratio.median == pytest.approx(np.median([800 / 1000, 2000 / 2400]))
    assert m.ratio.n == 2  # the land tables' study on the same sales


def test_roll_per_unit_is_a_reference_range(tmp_path):
    m = compute(make_session(tmp_path), everything(), "Area")
    area = next(e for e in m.evidence if e.unit == "Area")
    # roll / area where both are positive: 8, 10, 10, 5 (parcel 5 has no area, parcel 4 no roll)
    assert (area.roll.n, area.roll.minimum, area.roll.mean, area.roll.maximum) == (4, 5.0, 8.25, 10.0)
    lot = next(e for e in m.evidence if e.unit == "Lot")
    assert lot.roll.n == 5 and lot.roll.maximum == 2000.0


def test_no_roll_mapped(tmp_path):
    m = compute(make_session(tmp_path, with_roll=False), everything(), "Area")
    assert m.roll is None and all(e.roll is None for e in m.evidence)
