import os

import numpy as np
import pytest
from test_core import synthetic

from land_painter.data.config import CATEGORICAL, CONTINUOUS, ColumnRef, parcel_only_config
from land_painter.data.dataset import build_dataset
from land_painter.data.sources import inspect
from land_painter.land_table import ADD, IN, MULTIPLY, Adjustment, Condition, FlatRate, LandTable, split_tables
from land_painter.neighborhoods import ROOT
from land_painter.session import Session


@pytest.fixture
def session(tmp_path):
    gdf = synthetic.make_parcels(10)
    gdf["area"] = [100.0] * 9 + [None]
    gdf["zone"] = ["R1", "R2", "C"] * 3 + ["R1"]
    gdf["acres"] = np.arange(10, dtype=float)
    path = tmp_path / "parcels.parquet"
    gdf.to_parquet(path)
    config = parcel_only_config(inspect(path))
    p = config.parcel_file
    config.units = {"Area": ColumnRef(p, "area")}
    config.characteristic_kinds = {f"{p}::zone": CATEGORICAL, f"{p}::acres": CONTINUOUS}
    return Session.new(build_dataset(config))


def zone(session, *values):
    return Condition(f"{session.dataset.config.parcel_file}::zone", IN, values=values)


def acres(session, op, value):
    return Condition(f"{session.dataset.config.parcel_file}::acres", op, value)


def values(session):
    return [None if np.isnan(v) else v for v in session.land_values]


def members(session, index):
    return np.flatnonzero(session.neighborhoods.assignment == index).tolist()


# --- Folding --------------------------------------------------------------------------------


def test_split_tables_folds_a_multiplier_into_the_rate():
    corner = Adjustment(Condition("f", ">", 1), ADD, 50.0)
    lake = Adjustment(Condition("g", IN, values=("Y",)), MULTIPLY, 1.25, "Lake")
    table = LandTable("Area", FlatRate(8.0), (corner, lake))
    parent, child = split_tables(table, 1)
    assert parent == LandTable("Area", FlatRate(8.0), (corner,))
    assert child == LandTable("Area", FlatRate(10.0), (corner,))  # 8 × 1.25, other adjustments carried over


def test_split_tables_keeps_an_addition_in_the_child():
    corner = Adjustment(Condition("f", ">", 1), ADD, 50.0)
    table = LandTable("Area", FlatRate(8.0), (corner,))
    parent, child = split_tables(table, 0)
    assert parent == LandTable("Area", FlatRate(8.0), ())
    assert child == table


# --- Convert an adjustment ------------------------------------------------------------------------


def test_convert_multiplier_keeps_values_and_is_one_undo_step(session, tmp_path):
    table = LandTable("Area", FlatRate(2.0), (
        Adjustment(zone(session, "C"), MULTIPLY, 1.5, "Commercial"),
        Adjustment(acres(session, ">=", 5), ADD, 10.0),
    ))
    session.set_land_table(ROOT, table)
    before = values(session)
    n_before = len(session.neighborhoods.neighborhoods)

    child = session.convert_adjustment(ROOT, 0)
    model = session.neighborhoods
    assert members(session, child) == [2, 5, 8]
    assert model.neighborhoods[child].name == "Commercial"  # the adjustment's name
    assert model.neighborhoods[child].table == LandTable("Area", FlatRate(3.0), (table.adjustments[1],))
    assert model.neighborhoods[ROOT].table == LandTable("Area", FlatRate(2.0), (table.adjustments[1],))
    assert values(session) == before
    assert session.undo_label() == "Convert Adjustment to Neighborhood"

    session.save_as(tmp_path / "s.lpsession")
    reopened = Session.open(tmp_path / "s.lpsession")
    assert members(reopened, child) == [2, 5, 8]
    assert reopened.neighborhoods.neighborhoods[child].table == model.neighborhoods[child].table
    assert values(reopened) == values(session)

    session.undo()
    assert len(model.neighborhoods) == n_before
    assert model.neighborhoods[ROOT].table == table
    assert members(session, ROOT) == list(range(10))
    session.redo()
    assert members(session, child) == [2, 5, 8]


def test_convert_addition_keeps_it_in_the_child(session):
    table = LandTable("Area", FlatRate(2.0), (Adjustment(acres(session, ">=", 7), ADD, 25.0),))
    session.set_land_table(ROOT, table)
    before = values(session)
    child = session.convert_adjustment(ROOT, 0)
    assert members(session, child) == [7, 8, 9]
    assert session.neighborhoods.neighborhoods[child].name == "acres >= 7"  # unnamed: the condition
    assert session.neighborhoods.neighborhoods[child].table == table
    assert session.neighborhoods.neighborhoods[ROOT].table == LandTable("Area", FlatRate(2.0), ())
    assert values(session) == before


def test_convert_with_no_matches_records_nothing(session):
    session.set_land_table(ROOT, LandTable("Area", FlatRate(2.0), (Adjustment(zone(session, "X"), ADD, 1.0),)))
    label = session.undo_label()
    assert session.convert_adjustment(ROOT, 0) is None
    assert session.undo_label() == label
    with pytest.raises(ValueError):
        session.convert_adjustment(ROOT, 5)


# --- Split by filter ---------------------------------------------------------------------------


def test_split_moves_only_direct_members_and_copies_the_table(session):
    table = LandTable("Area", FlatRate(4.0))
    session.set_land_table(ROOT, table)
    existing = session.create_neighborhood(ROOT, np.array([8]))
    session.set_land_table(existing, LandTable("Area", FlatRate(1.0)))
    before = values(session)

    child = session.split_neighborhood(ROOT, acres(session, ">=", 7))
    assert members(session, child) == [7, 9]  # parcel 8 stays in its existing child
    assert members(session, existing) == [8]
    model = session.neighborhoods
    assert model.neighborhoods[child].parent == ROOT
    assert model.neighborhoods[child].table == table and model.neighborhoods[ROOT].table == table
    assert model.neighborhoods[child].name == "acres >= 7"
    assert values(session) == before
    assert session.undo_label() == "Split Neighborhood"

    session.undo()
    assert members(session, ROOT) == [0, 1, 2, 3, 4, 5, 6, 7, 9]


def test_split_uses_a_given_name_and_ignores_no_match(session):
    child = session.split_neighborhood(ROOT, zone(session, "R2"), "  Residential 2 ")
    assert session.neighborhoods.neighborhoods[child].name == "Residential 2"
    assert session.split_neighborhood(ROOT, zone(session, "nope")) is None


# --- Dialogs (offscreen smoke test) ---------------------------------------------------------------


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_dialogs_build_conditions(qapp, session):
    from PySide6.QtCore import Qt

    from land_painter.ui.land_table_panel import AdjustmentDialog, SplitDialog

    parcels = np.arange(10)
    adjustment = Adjustment(zone(session, "C"), MULTIPLY, 1.5, "Commercial")
    dialog = AdjustmentDialog(session, parcels, adjustment, base_table=LandTable("Area", FlatRate(2.0)))
    assert dialog.result_adjustment() == adjustment
    assert dialog.picker.match_count() == 3 and dialog._ok.isEnabled()

    split = SplitDialog(session, ROOT)
    picker = split.picker
    picker.field.setCurrentIndex(picker.field.findText("zone"))
    assert not split._ok.isEnabled()  # no value ticked yet
    item = next(picker.values.item(i) for i in range(picker.values.count()) if picker.values.item(i).text() == "R2")
    item.setCheckState(Qt.CheckState.Checked)
    assert split._ok.isEnabled()
    assert split.condition_value() == zone(session, "R2")
    assert split.name_value() == "zone is R2"
    split.name.textEdited.emit("Custom")
    split.name.setText("Custom")
    item.setCheckState(Qt.CheckState.Unchecked)
    assert split.name_value() == "Custom"  # an edited name is kept
