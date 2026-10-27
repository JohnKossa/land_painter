import numpy as np
import pytest

from land_painter.data.config import CATEGORICAL, CONTINUOUS, FLAG, ColumnRef, parcel_only_config
from land_painter.data.dataset import build_dataset
from land_painter.data.sources import inspect
from land_painter.land_table import (
    ADD, IN, LOT, MULTIPLY, Adjustment, Condition, FlatRate, LandTable, evaluate,
)
from land_painter.neighborhoods import ROOT
from land_painter.session import Session

from test_core import synthetic


def test_condition_masks():
    numbers = np.array([1.0, 5.0, np.nan, 10.0])
    assert Condition("f", ">", 4).mask(numbers).tolist() == [False, True, False, True]
    assert Condition("f", "≠", 5).mask(numbers).tolist() == [True, False, False, True]  # missing never matches
    text = np.array(["R1", None, "C", "R2"], dtype=object)
    assert Condition("f", IN, values=("R1", "R2")).mask(text).tolist() == [True, False, False, True]


def test_evaluate_multiplies_before_adding():
    fields = {"zone": np.array(["A", "B", "A", "C"], dtype=object), "acres": np.array([1.0, 2.0, 3.0, 4.0])}
    table = LandTable("Area", FlatRate(10.0), (
        Adjustment(Condition("zone", IN, values=("A",)), MULTIPLY, 2.0),
        Adjustment(Condition("acres", ">=", 3), ADD, 5.0),
        Adjustment(Condition("acres", ">", 3.5), MULTIPLY, 0.5),
    ))
    units = np.array([100.0, 100.0, 100.0, np.nan])
    values = evaluate(table, units, fields.__getitem__, np.arange(4))
    # parcel 0: 1000 × 2 ; parcel 1: 1000 ; parcel 2: 1000 × 2 + 5 ; parcel 3: units missing
    assert values[:3].tolist() == [2000.0, 1000.0, 2005.0]
    assert np.isnan(values[3])


def test_table_round_trip():
    table = LandTable("Road frontage", FlatRate(125.5), (
        Adjustment(Condition("p::zone", IN, values=("R1", "R2")), MULTIPLY, 1.1, "residential"),
        Adjustment(Condition("p::acres", "<", 0.25), ADD, -2000.0),
    ))
    assert LandTable.from_dict(table.to_dict()) == table
    assert LandTable.from_dict(None) is None


@pytest.fixture
def session(tmp_path):
    gdf = synthetic.make_parcels(10)
    gdf["area"] = [100.0] * 9 + [None]
    gdf["zone"] = ["R1", "R2", "C"] * 3 + ["R1"]
    gdf["acres"] = np.arange(10, dtype=float)
    gdf["nbhd"] = ["A"] * 5 + ["B"] * 5
    path = tmp_path / "parcels.parquet"
    gdf.to_parquet(path)
    config = parcel_only_config(inspect(path))
    p = config.parcel_file
    config.units = {"Area": ColumnRef(p, "area")}
    config.characteristic_kinds = {f"{p}::zone": CATEGORICAL, f"{p}::acres": CONTINUOUS, f"{p}::nbhd": FLAG}
    return Session.new(build_dataset(config))


def values(session):
    return [None if np.isnan(v) else v for v in session.land_values]


def test_dataset_units_and_characteristics(session):
    d = session.dataset
    assert d.unit_names() == [LOT, "Area"]
    assert d.unit_values(LOT).tolist() == [1.0] * 10
    assert np.isnan(d.unit_values("Area")[9])
    p = d.config.parcel_file
    assert d.characteristic(f"{p}::acres")[:3].tolist() == [0.0, 1.0, 2.0]
    assert d.characteristic(f"{p}::zone")[:3].tolist() == ["R1", "R2", "C"]
    assert [f[1] for f in d.filter_fields()] == ["acres", "nbhd", "zone"]
    assert d.distinct_values(f"{p}::zone")[0] == "R1"  # most common first


def test_children_copy_parent_table_then_diverge(session):
    root_table = LandTable(LOT, FlatRate(5000.0))
    session.set_land_table(ROOT, root_table)
    assert values(session) == [5000.0] * 10

    child = session.create_neighborhood(ROOT, np.array([0, 1, 2]))
    assert session.neighborhoods.neighborhoods[child].table == root_table
    session.set_land_table(child, LandTable("Area", FlatRate(3.0)))
    assert values(session)[:4] == [300.0, 300.0, 300.0, 5000.0]

    session.set_land_table(ROOT, LandTable(LOT, FlatRate(7000.0)))  # parent edit does not reach child
    assert values(session)[:4] == [300.0, 300.0, 300.0, 7000.0]

    session.undo()
    session.undo()
    assert values(session)[:4] == [5000.0] * 4


def test_import_copies_parent_table(session):
    p = session.dataset.config.parcel_file
    session.set_land_table(ROOT, LandTable(LOT, FlatRate(1.0)))
    session.import_neighborhoods(session.dataset.characteristic(f"{p}::nbhd"))
    tables = [n.table for n in session.neighborhoods.neighborhoods]
    assert tables == [LandTable(LOT, FlatRate(1.0))] * 3


def test_apply_table_to_children(session):
    a = session.create_neighborhood(ROOT, np.array([0, 1]))
    b = session.create_neighborhood(ROOT, np.array([2, 3]))
    session.set_land_table(a, LandTable(LOT, FlatRate(10.0)))
    table = LandTable("Area", FlatRate(2.0), (
        Adjustment(Condition(f"{session.dataset.config.parcel_file}::zone", IN, values=("C",)), ADD, 1.0),))
    session.set_land_table(ROOT, table)
    assert session.apply_table_to_children(ROOT) == 2
    assert values(session) == [200.0, 200.0, 201.0, 200.0, 200.0, 201.0, 200.0, 200.0, 201.0, None]
    assert session.apply_table_to_children(ROOT) == 0  # nothing left to change

    session.undo()
    assert session.neighborhoods.neighborhoods[a].table == LandTable(LOT, FlatRate(10.0))
    assert session.neighborhoods.neighborhoods[b].table is None
    assert values(session)[:4] == [10.0, 10.0, None, None]


def test_values_follow_parcels_and_survive_reopen(session, tmp_path):
    session.set_land_table(ROOT, LandTable(LOT, FlatRate(100.0)))
    a = session.create_neighborhood(ROOT, np.array([5, 6]))
    session.set_land_table(a, LandTable(LOT, FlatRate(1.0)))
    expected = values(session)
    assert expected[4:8] == [100.0, 1.0, 1.0, 100.0]

    session.save_as(tmp_path / "s.lpsession")
    reopened = Session.open(tmp_path / "s.lpsession")
    assert values(reopened) == expected
    assert reopened.undo_label() == "Edit Land Table"


def test_draft_preview_does_not_change_session(session):
    session.set_land_table(ROOT, LandTable(LOT, FlatRate(100.0)))
    parcels, preview = session.values_for(ROOT, LandTable(LOT, FlatRate(1.0)))
    assert len(parcels) == 10 and preview.tolist() == [1.0] * 10
    assert values(session) == [100.0] * 10
    assert not session.set_land_table(ROOT, LandTable(LOT, FlatRate(100.0)))  # unchanged: nothing journaled
