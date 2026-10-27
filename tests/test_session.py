import sqlite3

import numpy as np
import pytest

from land_painter.actions import ImportNeighborhoods
from land_painter.data.config import ColumnRef, parcel_only_config
from land_painter.data.dataset import build_dataset
from land_painter.data.sources import inspect
from land_painter.neighborhoods import ROOT
from land_painter.session import Session, SessionError, read_config, relocate_inputs

from test_core import synthetic


@pytest.fixture
def parcel_file(tmp_path):
    path = tmp_path / "parcels.parquet"
    synthetic.make_parcels(100).to_parquet(path)
    return path


def new_session(parcel_file, **config_changes):
    config = parcel_only_config(inspect(parcel_file))
    for name, value in config_changes.items():
        setattr(config, name, value)
    return Session.new(build_dataset(config))


@pytest.fixture
def session(parcel_file):
    return new_session(parcel_file)


def assignment(session):
    return session.neighborhoods.assignment.tolist()


def test_create_neighborhood_ignores_parcels_outside_parent(session):
    a = session.create_neighborhood(ROOT, np.array([0, 1, 2]))
    b = session.create_neighborhood(ROOT, np.array([2, 3]))  # 2 belongs to a
    assert assignment(session)[:5] == [a, a, a, b, 0]
    assert session.create_neighborhood(a, np.array([3, 4])) is None  # nothing qualifies
    assert session.journal.length == 2  # the no-op was not journaled


def test_undo_redo(session):
    a = session.create_neighborhood(ROOT, np.array([0, 1, 2]))
    child = session.create_neighborhood(a, np.array([1]))
    after = assignment(session)

    assert session.undo_label() == "Create Neighborhood"
    session.undo()
    assert assignment(session)[:3] == [a, a, a]
    session.undo()
    assert set(assignment(session)) == {ROOT}
    assert session.undo() is None
    assert len(session.neighborhoods.neighborhoods) == 1

    session.redo()
    session.redo()
    assert session.redo() is None
    assert assignment(session) == after
    assert session.neighborhoods.neighborhoods[child].parent == a


def test_new_action_after_undo_discards_redo(session):
    session.create_neighborhood(ROOT, np.array([0, 1]))
    session.create_neighborhood(ROOT, np.array([5, 6]))
    session.undo()
    session.create_neighborhood(ROOT, np.array([8]))
    assert session.journal.length == 2
    assert session.redo_label() is None
    assert assignment(session)[5:9] == [0, 0, 0, 2]


def test_save_and_reopen_replays_journal(session, tmp_path):
    a = session.create_neighborhood(ROOT, np.arange(10))
    session.create_neighborhood(a, np.array([3, 4]))
    session.create_neighborhood(ROOT, np.array([50]))
    session.undo()  # undone actions stay redoable after reopening

    path = tmp_path / "s.lpsession"
    session.save_as(path)
    session.create_neighborhood(ROOT, np.array([60]))  # written straight to the file
    expected = assignment(session)

    reopened = Session.open(path)
    assert assignment(reopened) == expected
    assert reopened.redo_label() is None  # the new action discarded the old redo
    reopened.undo()
    reopened.undo()
    assert assignment(reopened)[:11] == [a] * 10 + [0]


def test_neighborhood_import_on_new_session(tmp_path):
    gdf = synthetic.make_parcels(10)
    gdf["nbhd"] = ["B", "A", None, "B", " ", "A", "C", "C", "B", None]
    path = tmp_path / "parcels.parquet"
    gdf.to_parquet(path)
    session = new_session(path, neighborhood=ColumnRef(str(path.resolve()), "nbhd"))

    names = [n.name for n in session.neighborhoods.neighborhoods]
    assert names == ["Everything", "A", "B", "C"]
    assert assignment(session) == [2, 1, 0, 2, 0, 1, 3, 3, 2, 0]
    assert session.undo_label() == ImportNeighborhoods.label

    session.undo()
    assert set(assignment(session)) == {0} and len(session.neighborhoods.neighborhoods) == 1
    session.redo()
    session.create_neighborhood(2, np.array([0, 3, 6]))  # 6 is in C, ignored
    session.save_as(tmp_path / "s.lpsession")
    reopened = Session.open(tmp_path / "s.lpsession")
    assert assignment(reopened) == assignment(session)


def test_import_only_takes_parcels_from_parent(session):
    a = session.create_neighborhood(ROOT, np.array([0, 1]))
    assert session.import_neighborhoods(["X"] * 100) == 1
    assert assignment(session)[:3] == [a, a, a + 1]
    session.undo()
    assert assignment(session)[:3] == [a, a, 0]


def test_relocated_inputs_are_found_beside_session(session, parcel_file, tmp_path):
    session.create_neighborhood(ROOT, np.array([1]))
    moved_dir = tmp_path / "moved"
    moved_dir.mkdir()
    session.save_as(moved_dir / "s.lpsession")
    session.journal.conn.close()
    parcel_file.rename(moved_dir / parcel_file.name)

    config = read_config(moved_dir / "s.lpsession")
    assert relocate_inputs(config, moved_dir / "s.lpsession")
    reopened = Session.open(moved_dir / "s.lpsession", config)
    assert assignment(reopened)[1] == 1
    assert read_config(moved_dir / "s.lpsession").parcel_file == str((moved_dir / parcel_file.name).resolve())


def test_missing_input_uses_locate(session, parcel_file, tmp_path):
    path = tmp_path / "s.lpsession"
    session.save_as(path)
    session.journal.conn.close()
    elsewhere = tmp_path / "elsewhere" / "renamed.parquet"
    elsewhere.parent.mkdir()
    parcel_file.rename(elsewhere)

    with pytest.raises(SessionError, match="not found"):
        relocate_inputs(read_config(path), path)
    config = read_config(path)
    relocate_inputs(config, path, locate=lambda missing: elsewhere)
    assert config.parcel_file == str(elsewhere.resolve())
    Session.open(path, config)


def test_open_rejects_different_parcel_file(session, parcel_file, tmp_path):
    path = tmp_path / "s.lpsession"
    session.save_as(path)
    session.journal.conn.close()
    synthetic.make_parcels(99).to_parquet(parcel_file)
    with pytest.raises(SessionError, match="does not match"):
        Session.open(path)


def test_session_file_is_plain_sqlite(session, tmp_path):
    session.create_neighborhood(ROOT, np.array([1, 2]))
    path = tmp_path / "s.lpsession"
    session.save_as(path)
    rows = sqlite3.connect(path).execute("SELECT seq, kind FROM actions").fetchall()
    assert rows == [(1, "create_neighborhood")]


# --- Sale flags ------------------------------------------------------------------

from land_painter.data.config import ROLE_SALES, FileSpec, SalesMapping  # noqa: E402
from land_painter.sales import DISQUALIFIED, KNOWN_GOOD, UNFLAGGED, usable  # noqa: E402


@pytest.fixture
def sales_session(parcel_file, tmp_path):
    sales = tmp_path / "sales.csv"
    sales.write_text("key,price,date,vacant\n" + "\n".join(
        f"P{i},{1000 * (i + 1)},2024-01-{i + 1:02d},{'Y' if i % 2 else 'N'}" for i in range(6)))
    config = parcel_only_config(inspect(parcel_file))
    config.files.append(FileSpec(str(sales), [ROLE_SALES]))
    config.sales = SalesMapping("key", "price", "date", "vacant", ["Y"])
    return Session.new(build_dataset(config))


def test_sale_flags_undo_redo(sales_session):
    s = sales_session
    assert s.sale_flags.tolist() == [UNFLAGGED] * 6
    assert s.set_sale_flags([1, 3], KNOWN_GOOD)
    assert s.set_sale_flags([3, 4], DISQUALIFIED)
    assert not s.set_sale_flags([4], DISQUALIFIED)  # no change, nothing journaled
    assert s.sale_flags.tolist() == [0, 1, 0, 2, 2, 0]
    assert s.undo_label() == "Flag Sales"

    s.undo()
    assert s.sale_flags.tolist() == [0, 1, 0, 1, 0, 0]  # 3 goes back to known good, not unflagged
    s.undo()
    assert s.sale_flags.tolist() == [0] * 6
    s.redo()
    s.redo()
    assert s.sale_flags.tolist() == [0, 1, 0, 2, 2, 0]
    assert usable(s.sale_flags, s.dataset.sales).tolist() == [True, True, True, False, False, True]


def test_sale_flags_survive_reopen(sales_session, tmp_path):
    s = sales_session
    s.set_sale_flags([0, 5], DISQUALIFIED)
    s.save_as(tmp_path / "s.lpsession")
    reopened = Session.open(tmp_path / "s.lpsession")
    assert reopened.sale_flags.tolist() == [2, 0, 0, 0, 0, 2]


def test_open_refuses_changed_sales(sales_session, tmp_path):
    s = sales_session
    s.set_sale_flags([0], KNOWN_GOOD)
    s.save_as(tmp_path / "s.lpsession")
    s.journal.conn.close()
    sales = tmp_path / "sales.csv"
    sales.write_text(sales.read_text() + "\nP9,5,2024-02-01,N")  # a sale was added
    with pytest.raises(SessionError, match="sales data does not match"):
        Session.open(tmp_path / "s.lpsession")
