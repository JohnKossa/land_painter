import numpy as np
import pytest

from land_painter.compare import (
    BOTH, OTHER, PARCELS, SALES, Group, attributes, categories, cliffs_delta, compare, cramers_v, histogram,
    ks_distance, magnitude, neighborhood_selection, rest_of_parent_selection, selection_for, subtree_selection,
    CLIFF_LEVELS, MAX_CATEGORIES,
)
from land_painter.data.config import CATEGORICAL, ROLE_SALES, ColumnRef, FileSpec, SalesMapping, parcel_only_config
from land_painter.data.dataset import build_dataset
from land_painter.data.sources import inspect
from land_painter.land_table import FlatRate, LandTable
from land_painter.neighborhoods import ROOT
from land_painter.session import Session

from test_core import synthetic


def brute_delta(a, b):
    return float(np.mean([np.sign(y - x) for x in a for y in b]))


def test_cliffs_delta_matches_definition():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 10, 40).astype(float), rng.integers(2, 12, 30).astype(float)  # many ties
    assert cliffs_delta(a, b) == pytest.approx(brute_delta(a, b))
    assert cliffs_delta(np.array([1.0, 2]), np.array([3.0, 4])) == 1.0
    assert cliffs_delta(np.array([3.0, 4]), np.array([1.0, 2])) == -1.0
    assert cliffs_delta(np.zeros(0), np.array([1.0])) is None


def test_ks_distance():
    assert ks_distance(np.array([1.0, 2, 3]), np.array([1.0, 2, 3])) == 0.0
    assert ks_distance(np.array([1.0, 2]), np.array([3.0, 4])) == 1.0
    assert ks_distance(np.array([1.0, 2, 3, 4]), np.array([3.0, 4, 5, 6])) == pytest.approx(0.5)


def test_cramers_v():
    assert cramers_v(np.array([10, 0]), np.array([0, 10])) == pytest.approx(1.0)
    assert cramers_v(np.array([5, 5]), np.array([10, 10])) == pytest.approx(0.0)
    assert cramers_v(np.array([10, 0]), np.array([10, 0])) is None  # one category in use


def test_magnitude_labels():
    assert [magnitude(v, CLIFF_LEVELS) for v in (0.1, -0.2, 0.4, 0.9)] == ["negligible", "small", "medium", "large"]


def group(values, label="g"):
    return Group(label, ("hood", 0), PARCELS, np.asarray(values, dtype=float))


def test_histogram_shares_and_off_scale():
    a = group(np.r_[np.linspace(10, 20, 98), 1000.0, -5.0])  # one far outlier each side
    b = group(np.linspace(15, 25, 50))
    h = histogram([a, b])
    assert len(h.shares) == 2 and len(h.edges) - 1 == len(h.shares[0])
    for g, shares, below, above in zip((a, b), h.shares, h.below, h.above):
        assert shares.sum() + (below + above) / len(g.present) == pytest.approx(1.0)
    assert h.above[0] >= 1 and h.below[0] >= 1  # the outliers are counted, not drawn


def test_histogram_log_counts_nonpositive_as_below():
    h = histogram([group([0.0, -1.0, 10, 100, 1000])], log=True)
    assert h.log and h.below[0] >= 2 and h.edges[0] > 0


def test_histogram_single_value_and_empty():
    assert histogram([group([5.0, 5.0, 5.0])]).shares[0].sum() == pytest.approx(1.0)
    assert histogram([group([np.nan])]) is None


def test_categories_fold_into_other():
    values = np.array([f"c{i}" for i in range(MAX_CATEGORIES + 3)] * 2 + ["c0"] * 5, dtype=object)
    cats = categories([Group("g", ("hood", 0), PARCELS, values)])
    assert cats.names[0] == "c0" and cats.names[-1] == OTHER and len(cats.names) == MAX_CATEGORIES + 1
    assert cats.counts[0][-1] == 6 and cats.shares[0].sum() == pytest.approx(1.0)


# --- With a session --------------------------------------------------------------------------------


@pytest.fixture
def session(tmp_path):
    """Two neighborhoods: B has bigger lots; every parcel valued at $10/sqft; sales only on small lots."""
    n = 40
    gdf = synthetic.make_parcels(n)
    in_b = np.arange(n) >= n // 2
    gdf["area"] = np.where(in_b, 2000.0, 1000.0) + np.arange(n)
    gdf["water"] = np.where(np.arange(n) % 4 == 0, "Y", "N")
    gdf.loc[3, "water"] = None
    parcels = tmp_path / "parcels.parquet"
    gdf.to_parquet(parcels)
    sales = tmp_path / "sales.csv"
    rows = [f"P{i},{10_000 + 100 * i},2024-01-01,{'Y' if i % 2 else 'N'}" for i in list(range(0, 10)) + [25]]
    sales.write_text("key,price,date,vacant\n" + "\n".join(rows) + "\n")
    config = parcel_only_config(inspect(parcels))
    p = config.parcel_file
    config.files.append(FileSpec(str(sales), [ROLE_SALES]))
    config.sales = SalesMapping("key", "price", "date", "vacant", ["Y"])
    config.units = {"Area": ColumnRef(p, "area")}
    config.characteristic_kinds[f"{p}::water"] = CATEGORICAL
    s = Session.new(build_dataset(config))
    s.set_land_table(ROOT, LandTable("Area", FlatRate(10.0)))
    child = s.create_neighborhood(ROOT, np.flatnonzero(in_b))
    s.set_land_table(child, LandTable("Area", FlatRate(10.0)))
    return s, child


def attr(session, key):
    return next(a for a in attributes(session) if a.key == key)


def hoods(s, *indices):
    return [neighborhood_selection(s.neighborhoods, i) for i in indices]


def label(s, index):
    return s.neighborhoods.neighborhoods[index].name


def test_attribute_list(session):
    s, _ = session
    keys = [a.key for a in attributes(s)]
    assert keys[:2] == ["land", "land_per:Area"] and "sale_ratio" in keys and "unit:Area" in keys
    assert "roll_change" not in keys and "bldg_ratio" not in keys  # roll values not mapped
    water = attr(s, next(k for k in keys if k.endswith("::water")))
    assert water.categorical


def test_lot_size_difference_is_large(session):
    s, child = session
    c = compare(s, attr(s, "unit:Area"), hoods(s, ROOT, child), PARCELS)
    assert [g.label for g in c.groups] == [label(s, ROOT), label(s, child)]
    assert [x.n for x in c.summaries] == [20, 20]
    e = c.effects[0]
    assert e.size == 1.0 and e.magnitude == "large" and e.ks == 1.0
    assert e.median_change == pytest.approx(np.median(np.arange(20, 40) + 2000) / np.median(np.arange(20) + 1000) - 1)


def test_land_per_unit_is_the_same(session):
    s, child = session
    e = compare(s, attr(s, "land_per:Area"), hoods(s, ROOT, child), PARCELS).effects[0]
    assert e.size == 0.0 and e.magnitude == "negligible"


def test_sales_vs_parcels(session):
    s, child = session
    c = compare(s, attr(s, "unit:Area"), hoods(s, ROOT, child), BOTH)
    r, k = label(s, ROOT), label(s, child)
    assert [g.label for g in c.groups] == [f"{r} — parcels", f"{r} — sales", f"{k} — parcels", f"{k} — sales"]
    assert [x.n for x in c.summaries] == [20, 10, 20, 1]
    # N0's sales sit on its 10 smallest lots: smaller than its parcels overall
    assert c.effects[0].size < -0.4


def test_sale_level_attributes(session):
    s, child = session
    c = compare(s, attr(s, "sale_ratio"), hoods(s, ROOT, child), PARCELS)
    assert c.groups == []  # sale-level attributes have no parcel population
    c = compare(s, attr(s, "sale_ratio"), hoods(s, ROOT, child), SALES)
    assert [x.n for x in c.summaries] == [5, 1]  # vacant sales only
    assert c.summaries[0].median == pytest.approx(np.median([10 * (1000 + i) / (10_000 + 100 * i) for i in (1, 3, 5, 7, 9)]))


def test_categorical_comparison(session):
    s, child = session
    key = next(a.key for a in attributes(s) if a.key.endswith("::water"))
    c = compare(s, attr(s, key), hoods(s, ROOT, child), PARCELS)
    assert c.categories.names[0] == "N"
    assert c.summaries[0].missing == 1 and c.summaries[0].n == 19
    assert c.effects[0].magnitude is not None


def test_histogram_keeps_near_tails():
    h = histogram([group(np.arange(1000.0, 1040.0))])  # no outliers: nothing is clipped
    assert h.below == [0] and h.above == [0] and h.edges[0] == 1000.0 and h.edges[-1] == 1039.0



def test_subtree_and_rest_of_parent(session):
    s, child = session
    model = s.neighborhoods
    # A grandchild inside the child, and a sibling of the child
    grandchild = s.create_neighborhood(child, np.arange(20, 25))
    sibling = s.create_neighborhood(ROOT, np.arange(0, 5))
    sub = subtree_selection(model, child)
    assert sub.parcels.sum() == 20  # the child's 15 direct parcels + the grandchild's 5
    rest = rest_of_parent_selection(model, child)
    assert rest.parcels.sum() == 20  # root's 15 direct parcels + the sibling's 5
    assert not (rest.parcels & sub.parcels).any()
    assert "Rest of" in rest.label and label(s, child) in rest.label
    assert rest_of_parent_selection(model, ROOT) is None  # the root has no parent
    c = compare(s, attr(s, "unit:Area"), [sub, rest], PARCELS)
    assert [x.n for x in c.summaries] == [20, 20] and c.effects[0].size == -1.0  # rest has the small lots


def test_selection_keys_rebuild_against_the_current_tree(session):
    s, child = session
    model = s.neighborhoods
    key = rest_of_parent_selection(model, child).key
    before = selection_for(model, key).parcels.sum()
    s.create_neighborhood(ROOT, np.arange(0, 5))  # a new sibling stays inside "rest of parent"
    assert selection_for(model, key).parcels.sum() == before
    assert selection_for(model, ("hood", 999)) is None


def test_bars_link_to_their_parcels(session):
    from land_painter.compare import bin_parcels, group_parcels

    s, child = session
    for key, log in (("unit:Area", False), ("unit:Area", True), ("land", False)):
        c = compare(s, attr(s, key), hoods(s, ROOT, child), PARCELS, log=log)
        for row in range(len(c.groups)):
            members = [bin_parcels(c, row, b) for b in range(len(c.histogram.edges) - 1)]
            assert [len(m) for m in members] == list(c.histogram.counts[row])  # bar count = parcels shown
            linked = np.concatenate(members)
            assert set(linked) <= set(c.groups[row].parcels)  # only the group's own parcels
            assert len(group_parcels(c, row)) == c.summaries[row].n


def test_sales_bars_link_to_sold_parcels(session):
    from land_painter.compare import bin_parcels

    s, child = session
    c = compare(s, attr(s, "unit:Area"), hoods(s, ROOT), SALES)
    linked = np.concatenate([bin_parcels(c, 0, b) for b in range(len(c.histogram.edges) - 1)])
    assert sorted(linked) == list(range(10))  # the 10 sold parcels in the root (P0–P9)


def test_categorical_bars_link_including_other(session):
    from land_painter.compare import bin_parcels

    s, child = session
    key = next(a.key for a in attributes(s) if a.key.endswith("::water"))
    c = compare(s, attr(s, key), hoods(s, ROOT), PARCELS)
    by_name = {name: bin_parcels(c, 0, b) for b, name in enumerate(c.categories.names)}
    assert len(by_name["Y"]) + len(by_name["N"]) == c.summaries[0].n  # parcel 3 (missing) is in neither
    assert 3 not in np.concatenate(list(by_name.values()))
