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
    return build_layer(synthetic.make_parcels(100), "key")  # 10 x 10 grid


def test_buffers_are_consistent(layer):
    assert layer.n_parcels == layer.n_parts == 100
    assert layer.starts[0] == 0
    assert layer.starts[-1] == len(layer.positions) // 2
    assert np.all(np.diff(layer.starts) == 9)  # 8 vertices + closing vertex
    minx, miny, maxx, maxy = layer.bounds
    assert -82.5 < minx < maxx < -81.5 and 26 < miny < maxy < 27  # reprojected to lng/lat


def _triangle_areas(layer):
    xy = layer.positions.reshape(-1, 2)
    tri = xy[layer.indices.reshape(-1, 3)]
    a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def test_triangles_cover_each_part(layer):
    assert _triangle_areas(layer).sum() == pytest.approx(sum(p.area for p in layer.parts), rel=1e-9)


def test_holes_are_kept_and_not_filled():
    import geopandas as gpd
    import shapely

    donut = shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(3, 3), (3, 7), (7, 7), (7, 3)]])
    inner = shapely.box(3, 3, 7, 7)
    layer = build_layer(gpd.GeoDataFrame({"key": ["d", "i"]}, geometry=[donut, inner], crs="EPSG:4326"), "key")
    assert len(layer.ring_starts) - 1 == 3  # donut exterior + hole, inner exterior
    areas = _triangle_areas(layer)
    owner = np.searchsorted(layer.starts, layer.indices.reshape(-1, 3)[:, 0], side="right") - 1
    assert areas[owner == 0].sum() == pytest.approx(84)  # 100 - 16: the hole is not filled
    assert areas[owner == 1].sum() == pytest.approx(16)


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


def test_members_of_filters_to_one_neighborhood():
    model = NeighborhoodModel(6)
    a = model.add_child(ROOT, np.array([0, 1, 2]), "A")
    assert model.members_of(ROOT, np.arange(6)).tolist() == [3, 4, 5]
    assert model.members_of(a, np.array([1, 4])).tolist() == [1]


def test_remove_last_returns_parcels_to_parent():
    model = NeighborhoodModel(6)
    a = model.add_child(ROOT, np.array([0, 1, 2]), "A")
    model.add_child(a, np.array([1]), "B")
    model.remove_last(np.array([1]))
    assert model.assignment.tolist() == [a, a, a, 0, 0, 0]
    assert len(model.neighborhoods) == 2
