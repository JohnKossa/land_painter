import numpy as np
import pytest

from land_painter.data.config import CATEGORICAL, CONTINUOUS, ColumnRef, parcel_only_config
from land_painter.data.dataset import build_dataset
from land_painter.data.sources import inspect
from land_painter.neighborhoods import ROOT
from land_painter.parcel_details import parcel_details
from land_painter.session import Session

from test_core import synthetic


@pytest.fixture
def session(tmp_path):
    """10 parcels: parcel 9 is the odd one out (huge acreage, a rare zone, missing area)."""
    gdf = synthetic.make_parcels(10)
    gdf["area"] = [100.0] * 9 + [None]
    gdf["acres"] = [1.0] * 9 + [100.0]
    gdf["zone"] = ["R1"] * 9 + ["C"]
    gdf["dep"] = [50000.0] * 10
    path = tmp_path / "parcels.parquet"
    gdf.to_parquet(path)
    config = parcel_only_config(inspect(path))
    p = config.parcel_file
    config.units = {"Area": ColumnRef(p, "area")}
    config.depreciated_cost = [ColumnRef(p, "dep")]
    config.characteristic_kinds = {f"{p}::zone": CATEGORICAL, f"{p}::acres": CONTINUOUS}
    return Session.new(build_dataset(config))


def details(session, parcel):
    return parcel_details(session.dataset, session.land_values, session.neighborhoods.assignment, parcel)


def row(d, label):
    for section in d.sections:
        for r in section.rows:
            if r.label == label:
                return r
    raise KeyError(label)


def test_numeric_outlier_is_flagged_with_the_mean(session):
    d9, d0 = details(session, 9), details(session, 0)
    outlier = row(d9, "acres")
    assert outlier.value == "100" and outlier.comparison == "10.9 avg"
    assert outlier.flagged and outlier.direction == "▲"
    assert not row(d0, "acres").flagged  # a typical parcel is not flagged


def test_categorical_shows_both_shares_and_flags_rare_values(session):
    rare = row(details(session, 9), "zone")
    assert rare.value == "C" and rare.comparison == "10% C · R1 90%"
    assert rare.flagged and rare.direction == "▲"
    common = row(details(session, 0), "zone")
    assert common.value == "R1" and common.comparison == "90% R1 (most common)" and not common.flagged


def test_missing_value_shows_a_dash_and_is_never_flagged(session):
    area = row(details(session, 9), "Area")  # parcel 9 has no area
    assert area.value == "—" and area.comparison == "100 avg" and not area.flagged


def test_money_fields_are_formatted_as_dollars(session):
    assert row(details(session, 0), "Depreciated cost").value == "$50,000"


def test_means_are_scoped_to_the_assigned_neighborhood(session):
    a = session.create_neighborhood(ROOT, np.array([0, 1, 2]))  # acres all 1 here
    d = details(session, 0)
    assert d.neighborhood == a and d.n_members == 3
    assert row(d, "acres").comparison == "1 avg" and not row(d, "acres").flagged


def test_detail_fields_lists_the_characteristics(session):
    labels = {label for _, label, _ in session.dataset.detail_fields()}
    assert {"zone", "acres"} <= labels
