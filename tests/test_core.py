import importlib.util
from pathlib import Path

import numpy as np
import pytest

from land_painter.neighborhoods import ROOT, NeighborhoodModel
from land_painter.parcels import build_layer

_spec = importlib.util.spec_from_file_location(
    "make_synthetic_parcels", Path(__file__).parents[1] / "scripts" / "make_synthetic_parcels.py"
)
synthetic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synthetic)


@pytest.fixture(scope="module")
def layer():
    return build_layer(synthetic.make_parcels(100))  # 10 x 10 grid


def test_buffers_are_consistent(layer):
    assert layer.n_parcels == layer.n_parts == 100
    assert layer.starts[0] == 0
    assert layer.starts[-1] == len(layer.positions) // 2
    assert np.all(np.diff(layer.starts) == 9)  # 8 vertices + closing vertex
    minx, miny, maxx, maxy = layer.bounds
    assert -82.5 < minx < maxx < -81.5 and 26 < miny < maxy < 27  # reprojected to lng/lat


def test_rings_are_counter_clockwise(layer):
    xy = layer.positions.reshape(-1, 2)
    for p in range(layer.n_parts):
        ring = xy[layer.starts[p]:layer.starts[p + 1]]
        signed_area = np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1])
        assert signed_area > 0


def test_lasso_includes_any_touched_parcel(layer):
    # A tiny triangle around one parcel's interior point should hit exactly that parcel.
    import shapely

    point = shapely.point_on_surface(layer.parts[55])
    x, y = point.x, point.y
    d = 1e-6
    hits = layer.query_polygon([[x - d, y - d], [x + d, y - d], [x, y + d]])
    assert hits.tolist() == [55]

    # A box over the whole extent hits everything.
    minx, miny, maxx, maxy = layer.bounds
    hits = layer.query_polygon([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]])
    assert len(hits) == 100


def test_self_intersecting_lasso_does_not_fail(layer):
    minx, miny, maxx, maxy = layer.bounds
    bowtie = [[minx, miny], [maxx, maxy], [maxx, miny], [minx, maxy]]
    assert len(layer.query_polygon(bowtie)) > 0


def test_create_child_only_takes_parcels_from_parent():
    model = NeighborhoodModel(10)
    a = model.create_child(ROOT, np.array([0, 1, 2, 3]))
    b = model.create_child(ROOT, np.array([3, 4, 5]))  # 3 already belongs to a
    assert model.assignment.tolist() == [a, a, a, a, b, b, 0, 0, 0, 0]

    grandchild = model.create_child(a, np.array([1, 2, 4, 6]))  # 4 is b's, 6 is root's
    assert model.assignment.tolist() == [a, grandchild, grandchild, a, b, b, 0, 0, 0, 0]
    assert model.neighborhoods[grandchild].parent == a


def test_create_child_with_no_qualifying_parcels_is_noop():
    model = NeighborhoodModel(5)
    a = model.create_child(ROOT, np.array([0, 1]))
    assert model.create_child(a, np.array([2, 3])) is None
    assert len(model.neighborhoods) == 2
