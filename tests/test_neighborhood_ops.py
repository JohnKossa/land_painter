import geopandas as gpd
import numpy as np
import pytest
import shapely

from land_painter.data.config import parcel_only_config
from land_painter.data.dataset import build_dataset
from land_painter.data.sources import inspect
from land_painter.land_table import LOT, FlatRate, LandTable
from land_painter.neighborhoods import ROOT
from land_painter.session import Session

D = 0.0002  # parcel size in degrees (~20 m); the 2-parcel gap is ~40 m, within adjacency distance


@pytest.fixture
def session(tmp_path):
    """A row of 8 parcels: a road-sized gap between parcels 3 and 4, a wide gap before 7."""
    xs = [0, 1, 2, 3, 6, 7, 8, 30]
    geoms = [shapely.box(-80 + x * D, 36, -80 + (x + 1) * D, 36 + D) for x in xs]
    gdf = gpd.GeoDataFrame({"key": [f"K{i}" for i in range(8)]}, geometry=geoms, crs="EPSG:4326")
    path = tmp_path / "parcels.parquet"
    gdf.to_parquet(path)
    return Session.new(build_dataset(parcel_only_config(inspect(path))))


def names(session):
    return [n.name for n in session.neighborhoods.neighborhoods]


def test_rename_is_undoable(session):
    a = session.create_neighborhood(ROOT, np.array([0, 1]))
    assert session.rename_neighborhood(a, "  Lakeside  ")
    assert names(session)[a] == "Lakeside"
    assert not session.rename_neighborhood(a, "Lakeside") and not session.rename_neighborhood(a, " ")
    session.undo()
    assert names(session)[a] == f"N{a}"


def test_adjacency_spans_road_gaps_but_not_far_parcels(session):
    a = session.create_neighborhood(ROOT, np.array([0, 1]))
    b = session.create_neighborhood(ROOT, np.array([4, 5]))
    c = session.create_neighborhood(ROOT, np.array([7]))
    assert session.adjacent_neighborhoods(a) == [ROOT]  # parcel 2 is in root
    assert sorted(session.adjacent_neighborhoods(b)) == [ROOT]  # parcel 3 across the road; 6 in root
    assert session.adjacent_neighborhoods(c) == []  # isolated
    session.reassign(a, np.array([2, 3]))
    assert sorted(session.adjacent_neighborhoods(b)) == sorted([ROOT, a])  # a now reaches across the road


def test_delete_moves_parcels_and_children(session):
    session.set_land_table(ROOT, LandTable(LOT, FlatRate(1.0)))
    a = session.create_neighborhood(ROOT, np.array([0, 1, 2, 3]))
    session.set_land_table(a, LandTable(LOT, FlatRate(5.0)))
    child = session.create_neighborhood(a, np.array([0]))
    b = session.create_neighborhood(ROOT, np.array([4, 5]))
    session.set_land_table(b, LandTable(LOT, FlatRate(9.0)))

    session.delete_neighborhood(a, b)
    model = session.neighborhoods
    assert model.neighborhoods[a].deleted
    assert model.assignment.tolist()[:6] == [child, b, b, b, b, b]
    assert model.neighborhoods[child].parent == ROOT  # moved up to a's parent
    assert session.land_values.tolist()[1:4] == [9.0, 9.0, 9.0]  # recipient's table applies
    assert a not in model.live() and a not in model.children(ROOT)

    session.undo()
    assert not model.neighborhoods[a].deleted and model.neighborhoods[child].parent == a
    assert model.assignment.tolist()[:4] == [child, a, a, a]
    assert session.land_values.tolist()[1:4] == [5.0, 5.0, 5.0]
    with pytest.raises(ValueError):
        session.delete_neighborhood(ROOT, a)


def test_merge_moves_parcels_and_children_under_destination(session):
    session.set_land_table(ROOT, LandTable(LOT, FlatRate(1.0)))
    a = session.create_neighborhood(ROOT, np.array([0, 1, 2, 3]))
    session.set_land_table(a, LandTable(LOT, FlatRate(5.0)))
    child = session.create_neighborhood(a, np.array([0]))
    b = session.create_neighborhood(ROOT, np.array([4, 5]))
    session.set_land_table(b, LandTable(LOT, FlatRate(9.0)))

    session.merge_neighborhood(a, b)
    model = session.neighborhoods
    assert model.neighborhoods[a].deleted
    assert model.assignment.tolist()[:6] == [child, b, b, b, b, b]
    assert model.neighborhoods[child].parent == b  # reparented under the destination, not up to root
    assert session.land_values.tolist()[1:4] == [9.0, 9.0, 9.0]  # destination's table applies
    assert a not in model.live() and a not in model.children(ROOT)
    assert session.undo_label() == "Merge Neighborhood"

    session.undo()  # one undo restores the parcels, the neighborhood and the child's parent
    assert not model.neighborhoods[a].deleted and model.neighborhoods[child].parent == a
    assert model.assignment.tolist()[:4] == [child, a, a, a]
    assert session.land_values.tolist()[1:4] == [5.0, 5.0, 5.0]


def test_merge_rejects_root_self_and_descendant(session):
    a = session.create_neighborhood(ROOT, np.array([0, 1, 2]))
    child = session.create_neighborhood(a, np.array([0]))
    with pytest.raises(ValueError):
        session.merge_neighborhood(ROOT, a)
    with pytest.raises(ValueError):
        session.merge_neighborhood(a, a)
    with pytest.raises(ValueError):
        session.merge_neighborhood(a, child)  # can't merge into its own descendant


def test_merge_survives_reopen(session, tmp_path):
    a = session.create_neighborhood(ROOT, np.array([0, 1]))
    b = session.create_neighborhood(ROOT, np.array([2, 3]))
    session.rename_neighborhood(b, "East")
    session.merge_neighborhood(a, b)
    session.save_as(tmp_path / "s.lpsession")
    reopened = Session.open(tmp_path / "s.lpsession")
    assert reopened.neighborhoods.assignment.tolist() == session.neighborhoods.assignment.tolist()
    assert [n.deleted for n in reopened.neighborhoods.neighborhoods] == [False, True, False]
    assert reopened.neighborhoods.neighborhoods[b].name == "East"


def test_reassign_takes_from_any_neighborhood(session):
    a = session.create_neighborhood(ROOT, np.array([0, 1]))
    b = session.create_neighborhood(ROOT, np.array([4, 5, 6]))
    moving, emptied = session.plan_reassign(a, np.array([1, 2, 4]))  # 1 is already in a
    assert moving.tolist() == [2, 4] and emptied == []
    assert session.reassign(a, np.array([1, 2, 4])) == []
    assert session.neighborhoods.assignment.tolist()[:7] == [a, a, a, 0, a, b, b]


def test_reassign_that_empties_a_neighborhood_deletes_it_in_one_action(session):
    a = session.create_neighborhood(ROOT, np.array([0, 1]))
    b = session.create_neighborhood(ROOT, np.array([4, 5]))
    _, emptied = session.plan_reassign(a, np.array([4, 5, 6]))
    assert emptied == [b]
    assert session.neighborhoods.assignment[4] == b  # planning changes nothing
    assert session.reassign(a, np.array([4, 5, 6])) == [b]
    assert session.neighborhoods.neighborhoods[b].deleted
    assert session.undo_label() == "Reassign Parcels"

    session.undo()  # one undo restores both the parcels and the neighborhood
    assert not session.neighborhoods.neighborhoods[b].deleted
    assert session.neighborhoods.assignment.tolist()[4:7] == [b, b, 0]


def test_root_and_parents_with_children_are_never_emptied(session):
    a = session.create_neighborhood(ROOT, np.arange(8))  # root now has no parcels of its own
    child = session.create_neighborhood(a, np.array([0]))
    _, emptied = session.plan_reassign(child, np.arange(1, 8))
    assert emptied == []  # a still has a child; root is never deleted


def test_deletions_survive_reopen(session, tmp_path):
    a = session.create_neighborhood(ROOT, np.array([0, 1]))
    b = session.create_neighborhood(ROOT, np.array([2, 3]))
    session.rename_neighborhood(b, "East")
    session.delete_neighborhood(a, b)
    session.reassign(ROOT, np.array([2]))
    session.save_as(tmp_path / "s.lpsession")
    reopened = Session.open(tmp_path / "s.lpsession")
    assert reopened.neighborhoods.assignment.tolist() == session.neighborhoods.assignment.tolist()
    assert [n.deleted for n in reopened.neighborhoods.neighborhoods] == [False, True, False]
    assert reopened.neighborhoods.neighborhoods[b].name == "East"
