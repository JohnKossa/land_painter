import numpy as np
import polars as pl
import pytest
from test_core import synthetic

from land_painter.data.config import CATEGORICAL, ColumnRef, parcel_only_config
from land_painter.data.dataset import build_dataset
from land_painter.data.sources import inspect
from land_painter.export import export, land_table_tables, neighborhood_paths, output_paths, parcel_table
from land_painter.land_table import ADD, IN, LOT, MULTIPLY, Adjustment, Condition, FlatRate, LandTable
from land_painter.neighborhoods import ROOT
from land_painter.session import Session


@pytest.fixture
def session(tmp_path):
    gdf = synthetic.make_parcels(6)
    gdf["area"] = [100.0, 200.0, 300.0, None, 500.0, 600.0]
    gdf["zone"] = ["R1", "C", "R1", "C", "R1", "C"]
    path = tmp_path / "parcels.parquet"
    gdf.to_parquet(path)
    config = parcel_only_config(inspect(path))
    p = config.parcel_file
    config.units = {"Area": ColumnRef(p, "area")}
    config.characteristic_kinds = {f"{p}::zone": CATEGORICAL}
    s = Session.new(build_dataset(config))

    zone = f"{p}::zone"
    s.set_land_table(ROOT, LandTable("Area", FlatRate(2.0), (
        Adjustment(Condition(zone, IN, values=("C",)), MULTIPLY, 1.5, "commercial"),
        Adjustment(Condition(zone, IN, values=("C",)), ADD, 100.0),
    )))
    north = s.create_neighborhood(ROOT, np.array([0, 1]))
    s.rename_neighborhood(north, "North")
    lake = s.create_neighborhood(north, np.array([1]))
    s.rename_neighborhood(lake, "Lake")
    s.set_land_table(lake, LandTable(LOT, FlatRate(9000.0)))
    gone = s.create_neighborhood(ROOT, np.array([5]))
    s.delete_neighborhood(gone, ROOT)
    return s


def test_paths(session):
    assert neighborhood_paths(session.neighborhoods)[:3] == [
        "Everything", "Everything / North", "Everything / North / Lake"]


def test_parcel_table(session):
    t = parcel_table(session)
    assert t["key"].to_list() == list(session.layer.keys)
    assert t["neighborhood"].to_list() == ["North", "Lake", "Everything", "Everything", "Everything", "Everything"]
    assert t["neighborhood_path"][1] == "Everything / North / Lake"
    assert t["unit_of_measure"].to_list() == ["Area", "Lot", "Area", "Area", "Area", "Area"]
    # parcel 5 (zone C): 600 x 2 = 1200 base, x 1.5 + 100 = 1900
    row = t.row(5, named=True)
    assert (row["units"], row["rate"], row["base_value"], row["multiplier"], row["addition"], row["land_value"]) == \
        (600.0, 2.0, 1200.0, 1.5, 100.0, 1900.0)
    assert t["land_value"][3] is None  # missing area -> no value, exported as null (not NaN)
    rebuilt = (t["base_value"] * t["multiplier"] + t["addition"]).to_list()
    assert [a == b for a, b in zip(rebuilt, t["land_value"].to_list()) if b is not None] == [True] * 5


def test_land_table_tables(session):
    tables, adjustments = land_table_tables(session)
    assert tables["neighborhood"].to_list() == ["Everything", "North", "Lake"]  # deleted one left out
    assert tables["parent"].to_list() == [None, "Everything", "North"]
    assert tables["parcels"].to_list() == [4, 1, 1]
    assert tables.row(2, named=True)["unit_of_measure"] == "Lot" and tables["rate"][2] == 9000.0
    assert tables["adjustments"].to_list() == [2, 2, 0]  # North inherited Everything's table
    first = adjustments.row(0, named=True)
    assert (first["neighborhood"], first["order"], first["name"], first["field"], first["values"],
            first["effect"], first["amount"]) == ("Everything", 1, "commercial", "zone", "C", "multiply", 1.5)
    assert first["field_source"] == "parcels.parquet" and first["condition"] == "zone is C"


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_export_round_trip(session, tmp_path, suffix):
    written = export(session, tmp_path / f"out{suffix}")
    # Parcels follow the chosen format; the two reference tables are always CSV.
    assert [p.name for p in written] == [f"out{suffix}", "out_land_tables.csv", "out_adjustments.csv"]
    assert written == output_paths(tmp_path / f"out{suffix}")
    read_parcels = pl.read_csv if suffix == ".csv" else pl.read_parquet
    parcels = read_parcels(written[0])
    assert parcels.height == 6 and parcels["land_value"][5] == 1900.0
    assert pl.read_csv(written[1]).height == 3 and pl.read_csv(written[2]).height == 4


def test_parquet_is_geoparquet_with_geometry(session, tmp_path):
    import geopandas as gpd

    written = export(session, tmp_path / "out.parquet")
    gdf = gpd.read_parquet(written[0])  # fails if it isn't valid GeoParquet
    assert gdf.crs is not None and gdf.crs.to_epsg() == 4326
    assert gdf.geometry.notna().all() and (gdf.geom_type.isin(["Polygon", "MultiPolygon"])).all()
    assert len(gdf) == session.layer.n_parcels
    assert list(gdf["key"]) == list(session.layer.keys)  # geometry aligns with the attribute rows


def test_csv_keeps_keys_as_text(session, tmp_path):
    written = export(session, tmp_path / "out.csv")
    first_line = written[0].read_text().splitlines()[1]
    assert first_line.startswith('"P0","North"')
